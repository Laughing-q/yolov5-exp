# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel
from utils.general import xywh2xyxy
from .tal.anchor_generator import generate_anchors, dist2bbox, bbox2dist


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
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
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

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

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
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
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False, use_dfl=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
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

        self.reg_max = 16 if use_dfl else 0
        self.use_dfl = use_dfl
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)

    def __call__(self, p, targets, imgs=None):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        ldfl = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices = self.build_targets(p, targets)  # targets

        # Losses
        fs = [pi.shape[2:4] for pi in p]
        anchor_points, _ = generate_anchors(p, torch.tensor([8, 16, 32]), 5.0, 0.5, device=p[0].device, is_eval=True)
        # anchor_points = anchor_points / stride_tensor
        anchor_points = anchor_points.split((fs[0][0] * fs[0][1], fs[1][0] * fs[1][1], fs[2][0] * fs[2][1]), 0)
        strides = [8, 16, 32]
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros((pi.shape[0], pi.shape[2], pi.shape[3]), dtype=pi.dtype,
                               device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                pdist, _, pcls = pi[b, :, gj, gi].split(((self.reg_max + 1) * 4, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                an_p = anchor_points[i].view(fs[i][0], fs[i][1], 2)[gj, gi, :]
                pbox = self.bbox_decode(an_p, pdist, "xywh")
                iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # vis
                # if imgs is not None:
                #     tdist = bbox2dist(an_p, xywh2xyxy(tbox[i]), self.reg_max)
                #     self.vis_assignments(imgs, b, gj, gi, strides[i], tdist)

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, gj, gi, iou = b[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, gj, gi] = iou  # iou ratio

                # dfl
                if self.use_dfl:
                    tdist = bbox2dist(an_p, xywh2xyxy(tbox[i]), self.reg_max)
                    ldfl += self._df_loss(pdist.view([-1, 4, self.reg_max + 1]).contiguous(), tdist).mean()

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[:, (self.reg_max + 1) * 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        ldfl *= 0.5
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls + ldfl) * bs, torch.cat((lbox, lobj, ldfl)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        nt = targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices = [], [], []
        gain = torch.ones(6, device=self.device)  # normalized to gridspace gain

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(n,7)
            if nt:
                # Matches
                # r = t[..., 4:6] / self.anchors[i]  # wh ratio
                # j = torch.max(r, 1 / r).max(1)[0] < self.hyp['anchor_t']  # compare
                # # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                # t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = torch.zeros((0, 6), device=targets.device)
                offsets = 0

            # Define
            bc, gxy, gwh = t.chunk(3, 1)  # (image, class), grid xy, grid wh
            b, c = bc.long().T  # image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, grid_y, grid_x indices
            tbox.append(torch.cat((gxy, gwh), 1))  # box
            tcls.append(c)  # class

        return tcls, tbox, indices

    def bbox_decode(self, anchor_points, pred_dist, box_format='xyxy'):
        if self.use_dfl:
            n, _= pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(n, 4, self.reg_max + 1), dim=-1)\
                                .matmul(self.proj.to(pred_dist.device).to(pred_dist.dtype))
        return dist2bbox(pred_dist, anchor_points, box_format=box_format)

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = (
            F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(target_left.shape) * weight_left
        )
        loss_right = (
            F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction="none").view(target_left.shape) * weight_right
        )
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def vis_assignments(self, imgs, b, gj, gi, stride, tdist):
        imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # b, h, w, 3
        bs, h, w, _ = imgs.shape
        fg_mask = torch.zeros((bs, h // stride, w // stride), dtype=torch.uint8)
        tdist = torch.as_tensor(tdist, dtype=torch.long)
        for bi in b.unique():
            index = b == bi
            y = torch.as_tensor(gj[index], dtype=torch.long)
            x = torch.as_tensor(gi[index], dtype=torch.long)
            fg_mask[b[index], y, x] = 1

            for jj in range(len(x)):
                x1 = tdist[index][:, 0]
                fg_mask[b[index][jj], y[jj], (x - x1)[jj]:x[jj]] = 1
                y1 = tdist[index][:, 1]
                fg_mask[b[index][jj], (y - y1)[jj]:y[jj], x[jj]] = 1
                x2 = tdist[index][:, 2]
                fg_mask[b[index][jj], y[jj], x[jj]:(x + x2)[jj]] = 1
                y2 = tdist[index][:, 3]
                fg_mask[b[index][jj], y[jj]:(y + y2)[jj], x[jj]] = 1

        fg_mask = F.interpolate(fg_mask[None], (640, 640), mode="nearest")[0].cpu().numpy()
        for i in range(len(imgs)):
            img = imgs[i]
            fg = fg_mask[i]
            img[fg.astype(bool)] = img[fg.astype(bool)] * 0.35 + (np.array((0, 0, 255)) * 0.65)
            cv2.imshow("p", img)
            if cv2.waitKey(0) == ord("q"):
                exit()
