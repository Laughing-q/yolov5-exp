# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
from utils.assigner import TaskAlignedAssigner
from utils.general import xywh2xyxy, xyxy2xywh


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        gt_score = gt_score * label
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score  # * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = F.binary_cross_entropy(pred_score.float(), label.float(), reduction="none") * weight
        return loss


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

        self.BCEcls2 = VarifocalLoss()
        self.assigner = TaskAlignedAssigner(topk=3, num_classes=80, alpha=1.0, beta=6.0)

    # def __call__(self, p, targets):  # predictions, targets
    #     lcls = torch.zeros(1, device=self.device)  # class loss
    #     lbox = torch.zeros(1, device=self.device)  # box loss
    #     lobj = torch.zeros(1, device=self.device)  # object loss
    #     tcls, tobj, tbox, indices = self.build_targets(p, targets)  # targets
    #
    #     # Losses
    #     for i, pi in enumerate(p):  # layer index, layer predictions
    #         tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype, device=self.device)  # target obj
    #
    #         n = len(tobj[i])  # number of targets
    #         shape = pi.shape
    #         if n:
    #             # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
    #             pxy, pwh, _, pcls = pi.split((2, 2, 1, self.nc), 1)  # target-subset of predictions
    #             pcls = pcls.view(shape[0], shape[1] - 5, -1)
    #
    #             # Regression
    #             # pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
    #             # pwh = (0.0 + (pwh - 1.09861).sigmoid() * 4) * anchors[i]
    #             # pwh = (0.33333 + (pwh - 1.09861).sigmoid() * 2.66667) * anchors[i]
    #             # pwh = (0.25 + (pwh - 1.38629).sigmoid() * 3.75) * anchors[i]
    #             # pwh = (0.20 + (pwh - 1.60944).sigmoid() * 4.8) * anchors[i]
    #             # pwh = (0.16667 + (pwh - 1.79175).sigmoid() * 5.83333) * anchors[i]
    #             pxy = pxy.sigmoid() * 1.6 - 0.3
    #             pwh = (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i][:, :, None, None]
    #             pbox = torch.cat((pxy, pwh), 1).view(shape[0], 4, -1)  # predicted box
    #             print(pbox.shape, indices[i].shape)
    #             iou = bbox_iou(pbox[indices[i]], tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
    #             lbox += (1.0 - iou).mean()  # iou loss
    #
    #             iou = iou.detach().clamp(0).type(tobj.dtype)
    #             # Classification
    #             if self.nc > 1:  # cls loss (only if multiple classes)
    #                 t = torch.full_like(pcls, self.cn, device=self.device)  # targets
    #                 t[range(n), tcls[i]] = self.cp
    #                 # lcls += self.BCEcls(pcls, t)  # BCE
    #                 lcls += self.BCEcls2(pred_score=pcls.sigmoid(), gt_score=iou[:, None], label=t).mean()
    #
    #             # Append targets to text file
    #             # with open('targets.txt', 'a') as file:
    #             #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
    #
    #         obji = self.BCEobj(pi[:, 4], tobj)
    #         lobj += obji * self.balance[i]  # obj loss
    #         if self.autobalance:
    #             self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
    #
    #     if self.autobalance:
    #         self.balance = [x / self.balance[self.ssi] for x in self.balance]
    #     lbox *= self.hyp["box"]
    #     lobj *= self.hyp["obj"]
    #     lcls *= self.hyp["cls"]
    #     bs = tobj.shape[0]  # batch size
    #
    #     return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
    #
    # def build_targets(self, p, targets):
    #     # p[i]: (b, a, h, w)
    #     # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    #     nt = targets.shape[0]  # number of anchors, targets
    #     tcls, tbox, tobj, indices = [], [], [], []
    #     gain = torch.ones(6, device=self.device)  # normalized to gridspace gain
    #
    #     for i in range(self.nl):
    #         anchors, shape = self.anchors[i], p[i].shape
    #         gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain
    #
    #         # Match targets to anchors
    #         t = targets * gain  # shape(n, 6)
    #         if nt:
    #             # Matches
    #             r = t[..., 4:6] / anchors  # wh ratio
    #             j = torch.max(r, 1 / r).max(1)[0] < self.hyp["anchor_t"]  # compare
    #             # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
    #             t = t[j]  # filter
    #
    #             gt_box = t[:, 2:]
    #             gt_cls = t[:, 1]
    #             bi = t[:, 0].long()
    #             pxy, pwh, _, pcls = p[i].split((2, 2, 1, self.nc), 1)  # target-subset of predictions
    #             pcls = pcls.sigmoid().view(shape[0], shape[1] - 5, -1)[bi]
    #             pxy = pxy.sigmoid() * 1.6 - 0.3
    #             pwh = (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i][:, :, None, None]
    #             pbox = torch.cat((pxy, pwh), 1).view(shape[0], 4, -1)[bi]  # predicted box
    #             gt_score, idx = self.assigner(pcls, pbox, gt_cls, gt_box)
    #
    #         else:
    #             gt_score = 0
    #             idx = 0
    #         tobj.append(gt_score)
    #         tcls.append(gt_cls)
    #         tbox.append(gt_box)
    #         indices.append(idx.bool())

    def __call__(self, p, targets):
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        for i in range(self.nl):
            b, no, h, w = p[i].shape
            tobj = torch.zeros((b, h * w), dtype=p[i].dtype, device=self.device)  # target obj
            gain[2:6] = torch.tensor([w, h, w, h])  # xyxy gain
            t = targets * gain  # shape(n, 6)
            t = self.preprocess(t, batch_size=b)  # (b, max_num_obj, 5)
            gt_labels = t[:, :, :1]   # (b, max_num_obj, 1)
            gt_bboxes = t[:, :, 1:]   # (b, max_num_obj, 4), xywh
            # filter invalid which generated by `self.preprocess`
            mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()  # (b, max_num_obj, 1)

            pxy, pwh, pobj, pcls = p[i].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

            # pboxes
            pobj = pobj.view(b, -1)   # (b, h*w)
            pcls = pcls.view(b, no - 5, -1).permute(0, 2, 1).contiguous()  # (b, h*w, 80)
            pred_bboxes, torch_bboxes = self.bbox_decode(pxy, pwh, i, b)  # (b, h*w, 4), xyyx, xywh

            # anchor_point
            shift_x = (torch.arange(end=w, device=targets.device))
            shift_y = (torch.arange(end=h, device=targets.device))
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')  # (h, w)
            anchor_points = torch.stack(
                [shift_x, shift_y], axis=-1).clone().to(p[0].dtype)  # (h, w, 2)
            anchor_points = anchor_points.reshape([-1, 2])  # (h*w, 2)

            target_labels, target_bboxes, target_scores, fg_mask = \
                self.assigner(
                    pcls.detach(),
                    pred_bboxes.detach(),
                    anchor_points,
                    gt_labels,
                    xywh2xyxy(gt_bboxes),
                    mask_gt)
            n = fg_mask.sum()
            if n:
                iou = bbox_iou(torch_bboxes[fg_mask], target_bboxes[fg_mask], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss
                iou = iou.detach().clamp(0).type(tobj.dtype)
                tobj[fg_mask] = iou
                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    lcls += self.BCEcls(pcls, target_scores)  # BCE

            obji = self.BCEobj(pobj, tobj)
            lobj += obji * self.balance[i]  # obj loss
        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"] * 2
        lcls *= self.hyp["cls"] * 5
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def preprocess(self, targets, batch_size):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :]).to(
            targets.device
        )
        batch_target = targets[:, :, 1:5]
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets

    def bbox_decode(self, pxy, pwh, i, batch_size):
        # pxy = pxy.sigmoid() * 3 - 1   # (b, 2, h, w)
        pxy = pxy.sigmoid() * 1.6 - 0.3  # (b, 2, h, w)
        pwh = (0.2 + pwh.sigmoid() * 4.8) * self.anchors[i][:, :, None, None]
        pbox = torch.cat((pxy, pwh), 1).view(batch_size, 4, -1)  # bs, 4, num_grids
        pbox = pbox.permute(0, 2, 1).contiguous()  # b, h*w, 4
        return xywh2xyxy(pbox), pbox
