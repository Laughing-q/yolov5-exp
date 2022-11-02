import torch
import torch.nn as nn
import torch.nn.functional as F


def dist_calculator(gt_bboxes, anchor_bboxes):
    """compute center distance between all bbox and gt

    Args:
        gt_bboxes (Tensor): shape(bs*n_max_boxes, 4)
        anchor_bboxes (Tensor): shape(num_total_anchors, 4)
    Return:
        distances (Tensor): shape(bs*n_max_boxes, num_total_anchors)
        ac_points (Tensor): shape(num_total_anchors, 2)
    """
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = torch.stack([gt_cx, gt_cy], dim=1)
    ac_cx = (anchor_bboxes[:, 0] + anchor_bboxes[:, 2]) / 2.0
    ac_cy = (anchor_bboxes[:, 1] + anchor_bboxes[:, 3]) / 2.0
    ac_points = torch.stack([ac_cx, ac_cy], dim=1)

    distances = (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances, ac_points


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchors's center in gt

    Args:
        xy_centers (Tensor): shape(grids, 4)
        gt_bboxes (Tensor): shape(tn, 4)
    Return:
        (Tensor): shape(tn, grids)
    """
    n_anchors = xy_centers.size(0)
    tn, _ = gt_bboxes.size()
    xy_centers = xy_centers.unsqueeze(0).repeat(tn, 1, 1) # (tn, h*w, 4)
    gt_bboxes_lt = gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1) # (tn, grids, 2)
    gt_bboxes_rb = gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1) 

    b_lt = xy_centers - gt_bboxes_lt
    b_rb = gt_bboxes_rb - xy_centers
    bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
    bbox_deltas = bbox_deltas.reshape([tn, n_anchors, -1])
    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)


def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(tn, h*w)
        overlaps (Tensor): shape(tn, h*w)
    Return:
        target_gt_idx (Tensor): shape(b, h*w)
        fg_mask (Tensor): shape(b, h*w)
        mask_pos (Tensor): shape(b, n_max_boxes, h*w)
    """
    # (b, n_max_boxes, h*w) -> (b, h*w)
    fg_mask = mask_pos.sum(axis=-2)
    if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1]) # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.argmax(axis=1) # (b, h*w)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)  # (b, h*w, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype) # (b, n_max_boxes, h*w)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos) # (b, n_max_boxes, h*w)
        fg_mask = mask_pos.sum(axis=-2)
    # find each grid serve which gt(index)
    target_gt_idx = mask_pos.argmax(axis=-2) # (b, h*w)
    return target_gt_idx, fg_mask, mask_pos


def iou_calculator(box1, box2, eps=1e-9):
    """Calculate iou for batch

    Args:
        box1 (Tensor): shape(tn, 4), xyxy
        box2 (Tensor): shape(tn, grids, 4), xyxy
    Return:
        (Tensor): shape(tn, grids)
    """
    box1 = box1.unsqueeze(1)  # tn, 1, 4
    px1y1, px2y2 = box1[..., 0:2], box1[..., 2:4]
    gx1y1, gx2y2 = box2[..., 0:2], box2[..., 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps

    return overlap / union


class TaskAlignedAssignerV2(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssignerV2, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(nt, grids, nc)
            pd_bboxes (Tensor): shape(nt, grids, 4)
            anc_points (Tensor): shape(grids, 2)
            gt_labels (Tensor): shape(nt, 1)
            gt_bboxes (Tensor): shape(nt, 4)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.tn = pd_scores.size(0)

        if self.tn == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        # (tn, grids)
        mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, mask_pos)

        # normalize
        align_metric *= mask_pos  # tn, grids
        pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]  # (tn, )
        pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]  # (tn, )
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).unsqueeze(-1)  # (tn, grids, 1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, mask_pos.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points):

        # get anchor_align metric, (tn, grids)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask, (tn, h*w)
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask, (tn, h*w)
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts)
        # merge all mask to a final mask, (tn, h*w)
        mask_pos = mask_topk * mask_in_gts

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):

        pd_scores = pd_scores.permute(0, 2, 1)  # tn, nc, grids
        gt_labels = gt_labels.to(torch.long)  # tn, 1
        ind = torch.zeros([2, self.tn], dtype=torch.long)  # 2, tn
        ind[0] = torch.arange(end=self.tn)    # (tn, )
        ind[1] = gt_labels.squeeze(-1)  # (tn, )
        # get the scores of each grid for each gt cls
        bbox_scores = pd_scores[ind[0], ind[1]]  # tn, grids

        overlaps = iou_calculator(gt_bboxes, pd_bboxes) # tn, grids
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps


    def select_topk_candidates(self, metrics, largest=True):
        """
        Args:
            metrics: (tn, grids).
        """

        num_anchors = metrics.shape[-1]  # grids
        # (tn, topk)
        _, topk_idxs = torch.topk(metrics, self.topk, axis=-1, largest=largest)
        # (tn, topk, h*w) -> (tn, h*w)
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, fg_mask):
        """
        Args:
            gt_labels: (tn, 1)
            gt_bboxes: (tn, 4)
            fg_mask: (tn, grids)
        """

        # assigned target labels, (tn, grids)
        target_labels = torch.where(fg_mask == 1, gt_labels, fg_mask).long()
        # (tn, grids, 4)
        target_bboxes = torch.where(fg_mask[..., None].repeat(1, 1, 4) == 1, gt_bboxes[:, None], 
                                    torch.zeros((1, 1, 4), dtype=fg_mask.dtype, device=fg_mask.device))

        # assigned target scores
        target_scores = F.one_hot(target_labels, self.num_classes) # (tn, grids, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (tn, grids, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.full_like(target_scores, 0))

        return target_labels, target_bboxes, target_scores
