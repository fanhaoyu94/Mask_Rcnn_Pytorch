import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import List
from model.utils import Matcher, BalancePosNegSampler, Boxcoder, process_box, nms, iou
from model.losses import fastrcnn_loss, maskrcnn_loss

class FastRcnnHead(nn.Module):
    def __init__(self, in_channels:int, mid_channels:int, num_classes:int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls = nn.Linear(mid_channels, num_classes)
        self.bbox = nn.Linear(mid_channels, num_classes * 4)

        for name, layer in self.named_children():
            if name in ["fc1", "fc2"]:
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, inputs):
        feature = inputs.flatten()
        feature = self.fc1(feature)
        feature = F.relu(feature)
        feature = self.fc2(feature)
        feature = F.relu(feature)
        cls_prob = self.cls(feature)
        bbox = self.bbox(feature)
        return cls_prob, bbox


class MaskHead(nn.Sequential):
    def __init__(self, in_channels:int, layers:List[int], out_channels:int, num_class:int):
        d = OrderedDict()
        for idx, layer_channels in enumerate(layers):
            d[f"mask_conv{idx}"] = nn.Conv2d(in_channels, layer_channels)
            d[f"mask_relu{idx}"] = nn.ReLU()
            in_channels = layer_channels
        
        size = len(layers)
        d[f"mask_conv{size}"] = nn.ConvTranspose2d(layer_channels, out_channels, 2, 2, 0)
        d[f"mask_relu{size}"] = nn.ReLU()
        d["cls_score_pixel"] = nn.Conv2d(out_channels, num_class, 3, 1, 1)
        super().__init__(d)

        for name, params in self.named_parameters():
            if "weight" in name and "cls_score_pixel" not in name:
                nn.init.kaiming_normal_(params, mode="fan_out", nonlinearity="relu") # std = (2 / ((1 + power(a,2)) * fan_in)) sqrt


class MaskRcnnNetwork(nn.Module):
    def __init__(self, fastrcnn_roi_align, fastrcnn_roi_head,
                 mask_roi_align, mask_roi_head,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights, score_thresh, nms_thresh,
                 num_detections):
    
        self.fastrcnn_roi_align = fastrcnn_roi_align
        self.fastrcnn_roi_head = fastrcnn_roi_head
        
        self.mask_roi_align = mask_roi_align
        self.mask_roi_head = mask_roi_head

        self.propose_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matchers=True)
        self.fg_bg_sampler = BalancePosNegSampler(num_samples, positive_fraction)
        self.box_coder = Boxcoder(reg_weights)

        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh
        self.num_detections = num_detections
        self.min_size = 1
    
    def generate_training_samples(self, proposals, target):
        gt_box = target["boxes"]
        gt_label = target["labels"]

        iou_pg = iou(gt_box, proposals)
        label, matched_idx = self.propose_matcher(iou_pg)
        pos_idx, neg_idx = self.fg_bg_sampler(label)
        sample_index = torch.concat([pos_idx, neg_idx])

        regression_target = self.box_coder.encoder(gt_box[matched_idx[pos_idx]], proposals[pos_idx])
        proposals = proposals[sample_index]
        label = label[sample_index]
        label[:pos_idx.shape[0]] = gt_label[matched_idx[pos_idx]]
        label[pos_idx.shape[0]:] = 0

        return proposals, matched_idx, label, regression_target

    def fastrcnn_inference(self):
        pass

    def forward(self, feature, proposals, image_shape, target):
        if self.training:
            proposals, matched_idx, label, regression_target = self.generate_training_samples(proposals, target)
        feature_fastrcnn_roi = self.fastrcnn_roi_align(feature, proposals, image_shape)
        cls_pred, bbox_pred = self.fastrcnn_roi_head(feature_fastrcnn_roi)

        result, losses = {}, {}
        if self.training:
            cls_loss, bbox_loss = fastrcnn_loss(cls_pred, bbox_pred, label, regression_target)
            losses = dict(roi_classifier_loss=cls_loss, roi_box_loss=bbox_loss)
            # For mask Rcnn
            pos_num = regression_target.shape[0]
            mask_proposals = proposals[:pos_num]
            matched_idx_pos = matched_idx[:pos_num]
            mask_label = label[:pos_num]

            if mask_proposals.shape[0] == 0:
                losses.update({"roi_mask_loss":torch.tensor(0)})
                return result, losses

        else:
            result = self.fastrcnn_inference()
            # For Mask Rcnn
            mask_proposals = result["boxes"]
            
            if mask_proposals.shape[0] == 0:
                result.update({"masks":torch.empty((0,28,28))})
                return result, losses
            
        feature_maskrcnn_roi = self.mask_roi_align(feature, mask_proposals, image_shape)
        mask_output = self.mask_roi_head(feature_maskrcnn_roi)

        if self.training:
            gt_mask = target["masks"]
            mask_loss = maskrcnn_loss(mask_output, mask_proposals, matched_idx_pos, mask_label, gt_mask)
            losses.update(dict(maskrcnn_loss=mask_loss))
        else:
            label = target["label"]
            idx = torch.arange(0, label.shape[0], device=label.device)
            mask_result = mask_output[idx, mask_label]
            mask_result_final = F.sigmoid(mask_result)
            result.update(dict(masks=mask_result_final))
        return result, losses ## when visualize, you already generate bounding box, so then you reshape the (28, 28) to bounding box size
        


