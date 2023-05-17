import torch
from torch import nn
from model.utils import Anchorgenerator
from model.rpn_head import RPNHead, RegionProposalNetwork
from model.roi_align import roi_align
from model.roi_head import FastRcnnHead, MaskHead, MaskRcnnNetwork
from model.transform import Transformer
from torchvision import models


'''
several lession learn here:
1. when you print children for module, you actually can see the components of submodule,
moduleList, sequential
2. you can use module.layer to get layer, layer.weight and bias to get the params, more 
complicated params can get throuhgh _parameters. may called different name
3. requires_grad_ is a good way to inplace change whether train those parameteres, detach 
is not inplace, or requires_grad = True, still backward & forward pass, but no grad descent 
4. _ at the back is usually function for inplace change, without _ is usually a property
5. ModuleDict can either be a dictionary or iterabe key value pair
'''
class Resnet(nn.Module):
    def __init__(self, out_channels:int) -> None:
        super().__init__()
        body = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for name, params in body.named_parameters():
            if "layer2" not in name and "layer3" not in name and "layer4" not in name:
                params.requires_grad_(False)
        
        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = body.layer4[-1].bn3.weight.shape[0]
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        for name, layer in self.named_children():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
    
    def forward(self, input):
        x = self.body(input)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class mask_rcnn(nn.Module):
    def __init__(self, backbone, num_classes, image_shape,
                 # RPN parameters
                 anchor_sizes=(128, 256, 512), anchor_scales=(0.5, 1, 2),
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):
        super().__init__()
        self.feature_extraction = backbone
        out_channels = backbone.out_channels ## double check

        ## RPN Module
        anchor_generator = Anchorgenerator(anchor_sizes, anchor_scales, image_shape)
        rpn_heads = RPNHead(out_channels, len(anchor_sizes) * len(anchor_scales))
        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        
        self.RPN_Head = RegionProposalNetwork(
            anchor_generator, rpn_heads, 
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_num_samples, rpn_positive_fraction,
            rpn_reg_weights, image_shape,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh  
        )

        ## ROI Head
        fastrcnn_roi_align = roi_align()
        resolution = fastrcnn_roi_align.output_size[0]
        in_channels = out_channels * resolution ** 2 # this is for multilayer perceptron 
        mid_channels = 1024
        fastrcnn_roi_head = FastRcnnHead(in_channels, mid_channels, num_classes=)

        mask_roi_align = roi_align()
        mask_roi_head = MaskHead(out_channels, (256, 256, 256, 256), 256, num_classes)

        self.ROI_Head = MaskRcnnNetwork(
            fastrcnn_roi_align, fastrcnn_roi_head,
            mask_roi_align, mask_roi_head,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_num_samples, box_positive_fraction,
            box_reg_weights, box_score_thresh, box_nms_thresh,
            box_num_detections, image_shape
        )

        self.transformer = Transformer(
            min_size=800, max_size=1333, 
            image_mean=[0.485, 0.456, 0.406], 
            image_std=[0.229, 0.224, 0.225])
    
    def forward(self, inputs, target):
        ori_image_shape = inputs.shape[-2:]
        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]

        features = self.feature_extraction(image)
        #RPN stage
        proposal, rpn_losses = self.RPN_Head(features, image_shape, target)
        #ROI stage
        results, roi_losses = self.ROI_Head(features, proposal, target)

        if self.training:
            return dict(**rpn_losses, **roi_losses)
        
        else:
            return self.transformer.postprocess(results, image, ori_image_shape)


def Resnet50_Maskrcnn()



