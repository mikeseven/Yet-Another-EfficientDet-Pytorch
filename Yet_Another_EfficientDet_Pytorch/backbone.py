# Author: Zylo117

import torch
from torch import nn

from efficientdet.model import BiFPN, Regressor, Classifier, EfficientNet
from efficientdet.utils import Anchors


class EfficientDetBackbone(nn.Module):
    backbone_compound_coef = [0, 1, 2, 3, 4, 5, 6, 6, 7]
    fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
    fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
    pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
    anchor_scale = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 4.0]
    default_aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    default_num_scales = [2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    def __init__(self, num_classes=80, compound_coef=0, load_weights=False, onnx_export=False, **kwargs):
        super(EfficientDetBackbone, self).__init__()
        self.compound_coef = compound_coef
        self.onnx_export = onnx_export

        self.aspect_ratios = kwargs.get("ratios", self.default_aspect_ratios)
        self.num_scales = len(kwargs.get("scales", self.default_num_scales))
        conv_channel_coef = {
            # the channels of P3/P4/P5.
            0: [40, 112, 320],
            1: [40, 112, 320],
            2: [48, 120, 352],
            3: [48, 136, 384],
            4: [56, 160, 448],
            5: [64, 176, 512],
            6: [72, 200, 576],
            7: [72, 200, 576],
            8: [80, 224, 640],
        }

        num_anchors = len(self.aspect_ratios) * self.num_scales

        self.bifpn = nn.Sequential(
            *[
                BiFPN(
                    self.fpn_num_filters[self.compound_coef],
                    conv_channel_coef[compound_coef],
                    True if _ == 0 else False,
                    attention=True if compound_coef < 6 else False,
                    use_p8=compound_coef > 7,
                    onnx_export=self.onnx_export,
                )
                for _ in range(self.fpn_cell_repeats[compound_coef])
            ]
        )

        self.num_classes = num_classes
        self.regressor = Regressor(
            in_channels=self.fpn_num_filters[self.compound_coef],
            num_anchors=num_anchors,
            num_layers=self.box_class_repeats[self.compound_coef],
            pyramid_levels=self.pyramid_levels[self.compound_coef],
            onnx_export=self.onnx_export,
        )
        self.classifier = Classifier(
            in_channels=self.fpn_num_filters[self.compound_coef],
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_layers=self.box_class_repeats[self.compound_coef],
            pyramid_levels=self.pyramid_levels[self.compound_coef],
            onnx_export=self.onnx_export,
        )

        if not self.onnx_export:
            self.anchors = Anchors(
                anchor_scale=self.anchor_scale[compound_coef],
                pyramid_levels=(torch.arange(self.pyramid_levels[self.compound_coef]) + 3).tolist(),
                **kwargs,
            )
            self.cached_anchors = None

        self.backbone_net = EfficientNet(
            self.backbone_compound_coef[compound_coef], load_weights, onnx_export=self.onnx_export
        )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, inputs):
        max_size = inputs.shape[-1]

        _, p3, p4, p5 = self.backbone_net(inputs)

        features = (p3, p4, p5)
        features = self.bifpn(features)

        regression = self.regressor(features)
        classification = self.classifier(features)

        if not self.onnx_export:
            # if inputs are always the same dimensions, anchors don't change, anchors can be precaculated
            # and this avoids errors in int8 quantization, especially with position>256 on inputs>256
            if self.cached_anchors is None:
                self.cached_anchors = self.anchors(inputs, inputs.dtype)
            return features, regression, classification, self.cached_anchors

        return regression, classification

    def init_backbone(self, path):
        state_dict = torch.load(path)
        try:
            ret = self.load_state_dict(state_dict, strict=False)
            print(ret)
        except RuntimeError as e:
            print("Ignoring " + str(e) + '"')
