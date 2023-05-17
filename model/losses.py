import torch
import torch.nn.functional as F
from model.roi_align import roi_align


def fastrcnn_loss(cls_pred, bbox_pred, label, regression_target):
    cls_loss = F.cross_entropy(cls_pred, label)
    N, num_pos = label.shape[0], regression_target.shape[0]
    regression_target_reshaped = bbox_pred.view(N, -1, 4)
    regression_target_pos, label_pos = regression_target_reshaped[:num_pos], label[:num_pos]
    idx = torch.arange(0, num_pos, device=label.device)
    bbox_loss = F.l1_loss(regression_target_pos[idx, label_pos], regression_target, reduction="sum") / N
    return cls_loss, bbox_loss
    

def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx.unsqueeze(1).to(proposal)
    roi = torch.concat([matched_idx, proposal])

    M = mask_logit.shape[-1]
    gt_mask = gt_mask.unsqueeze(1).to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0] # just bunch of 0/1

    idx = torch.arange(0, label.shape[0], device=label.device)
    maskrcnn_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
    return maskrcnn_loss

    