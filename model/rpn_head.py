import torch
from torch import nn
import torch.nn.functional as F
from model.utils import Matcher, BalancePosNegSampler, Boxcoder, process_box, nms, iou

class RPNHead(nn.Module):
    def __init__(self, input_channels, num_anchors):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, input_channels, 3, 1, 1)
        self.cls_logits = nn.Conv2d(input_channels, num_anchors, 1)
        self.regression = nn.Conv2d(input_channels, 4 * num_anchors, 1)

        for name, layer in self.named_children():
            if name == "conv":
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
            else:
                nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        cls_logits = self.cls_logits(x)
        regression = self.regression(x)
        return cls_logits, regression


class RegionProposalNetwork(nn.Module):
    def __init__(self, anchor_generator, head, 
                fg_iou_thresh, bg_iou_thresh,
                num_samples, positive_fraction,
                reg_weights, pre_nms_top_n, 
                post_nms_top_n, nms_thresh):
        
        self.anchor_generator = anchor_generator
        self.head = head

        self.propose_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matchers=True)
        self.fg_bg_sampler = BalancePosNegSampler(num_samples, positive_fraction)
        self.box_coder = Boxcoder(reg_weights)

        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.min_size = 1
    
    def create_proposal(self, cls_logits, regression, image_shape, anchor):
        if self.training:
            pre_nms_top_n = self.pre_nms_top_n['training']
            post_nms_top_n = self.post_nms_top_n['training']
        else:
            pre_nms_top_n = self.pre_nms_top_n['testing']
            post_nms_top_n = self.post_nms_top_n['testing']
        
        pre_nms_top_n = min(cls_logits.shape[0], pre_nms_top_n)
        score, topk_index = cls_logits.topk(pre_nms_top_n)
        pred_box = self.box_coder.decode(regression[topk_index], anchor[topk_index])

        pred_box_clean, score = process_box(pred_box, score, image_shape, self.min_size)
        keep = nms(pred_box_clean, score, self.nms_thresh)[:post_nms_top_n]
        proposal = pred_box_clean[keep]
        return proposal

    def compute_loss(self, cls_logits, regression, anchor, gt_box):
        iou_matrix = iou(gt_box, anchor)
        label, matched_idx = self.propose_matcher(iou_matrix)
        pos_idx, neg_idx = self.fg_bg_sampler(label)
        sample_index = torch.concat([pos_idx, neg_idx])
        gt_box_scale = self.box_coder.encode(gt_box, anchor)

        cls_loss = F.binary_cross_entropy(cls_logits[sample_index], label[sample_index])
        reg_loss = F.l1_loss(regression[pos_idx], gt_box_scale[matched_idx[pos_idx]], reduction="sum") / sample_index.shape[0]
        return cls_loss, reg_loss

    def forward(self, features, image_shape, target=None):
        if target:
            gt_box = target["boxes"]
        
        anchor = self.anchor_generator(features)
        cls_logits, regression = self.head(features)
        cls_logits = cls_logits.permute(0, 2, 3, 1).flatten()
        regression = regression.permute(0, 2, 3, 1).reshape(-1, 4)

        ## Still in the same dtype and device, just detach from the graph
        proposal = self.create_proposal(cls_logits.detach(), regression.detach(), image_shape, anchor)

        if self.training:
            cls_loss, reg_loss = self.compute_loss(cls_logits, regression, anchor, gt_box)
            return proposal, dict(cls_loss=cls_loss, reg_loss=reg_loss)

        else:
            return proposal