from typing import Any
import torch 

test = torch.tensor([1,2,3])
test[test>2] = 4


class Anchorgenerator:
    '''
    anchor is target for orginal image gt-box
    '''
    def __init__(self, anchor_sizes, anchor_ratio):
        self.anchor_sizes = anchor_sizes
        self.anchor_ratio = anchor_ratio
    
    def _create_anchors(self, dtype, device):
        anchor_size = torch.tensor(self.anchor_sizes, dtype=dtype, device=device)
        anchor_ratio = torch.tensor(self.anchor_sizes, dtype=dtype, device=device)
        # also can use [None, :] to increase index
        y = anchor_size.view(3,1) * torch.sqrt(anchor_ratio).view(1,3)
        x = anchor_size.view(3,1) / torch.sqrt(anchor_ratio).view(1,3)
        x, y = x.reshape(-1), y.reshape(-1)
        output = torch.concat([-x, -y, x, y], dim=1) / 2
        return output
    
    def _grid_anchors(self, orig_anchor, grid, stride):
        dtype, device = orig_anchor.dtype, orig_anchor.device
        h = torch.arange(0, grid[0], dtype=dtype, device=device) * stride[0]
        w = torch.arange(0, grid[1], dtype=dtype, device=device) * stride[1]

        x, y = torch.meshgrid(w, h)
        grid_xy = torch.concat([x, y, x, y], dim=1).reshape(-1, 1, 4)
        grid_anchor = (grid_xy + orig_anchor).reshape(-1, 4)
        return grid_anchor
    
    def __call__(self, image, features) -> Any:
        dtype, device = features.dtype, features.device
        grid = tuple(features.shape[-2:]) # N, C, H, W
        stride = tuple([i/g for i,g in zip(image.shape[-2:], features.shape[-2:])])
        anchors = self._create_anchors(dtype, device)
        grid_anchor = self._grid_anchors(anchors, grid, stride) # shape is (total_grid, 9, 4)
        return grid_anchor


class Matcher:
    def __init__(self, fg_iou_thresh, bg_iou_thresh, allow_low_quality_matchers=True):
        self.fg_iou_thresh = fg_iou_thresh 
        self.bg_iou_thresh = bg_iou_thresh
        self.allow_low_quality_matchers = allow_low_quality_matchers
    
    def __call__(self, iou):
        label = torch.full((iou.shape[-1], 1), -1, dtype=iou.dtype, device=iou.shape)
        value, matched_ids = iou.max(dim=0)
        label[value > self.fg_iou_thresh] = 1
        label[value < self.bg_iou_thresh] = 0

        if self.allow_low_quality_matchers:
            _, low_quality_match = iou.max(dim=1)
            label[low_quality_match] = 1
            ## change match_ids
            low_matching_anchor = torch.where(value <= self.fg_iou_thresh)[0]
            for anchor_indice in low_matching_anchor:
                gt_box_indice = torch.where(low_quality_match == anchor_indice)[0]
                if gt_box_indice:
                    matched_ids[anchor_indice] = gt_box_indice

        return label, matched_ids


class BalancePosNegSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction
    
    def __call__(self, label) -> Any:
        pos_size = int(self.num_samples * self.positive_fraction)
        neg_size = self.num_samples - pos_size
        label_pos = torch.where(label == 1)[0]
        label_neg = torch.where(label == 0)[0]
        indice_pos = torch.randperm(label_pos.numel())[:pos_size]
        indice_neg = torch.randperm(label_neg.numel())[:neg_size]
        return label_pos[indice_pos], label_neg[indice_neg]


class Boxcoder:
    def __init__(self) -> None:
        pass


def process_box():
    pass


def nms():
    pass

def iou():
    pass