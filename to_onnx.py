import torch
from onnxsim import simplify
import onnx
from backbone import EfficientDetBackbone

# fmt: off
obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
        'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'] 
num_classes = len(obj_list)
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
nms_threshold = 0.5
threshold = 0.05
compound_coef = 0
model_name = f'efficientdet-d{compound_coef}'
# fmt: on

model = EfficientDetBackbone(
    compound_coef=compound_coef, num_classes=num_classes, ratios=anchor_ratios, scales=anchor_scales, onnx_export=True
)
model.load_state_dict(torch.load(f"weights/{model_name}.pth", map_location="cpu"))
model.requires_grad_(False)
model.eval()

input_name = "input"
input_shape = (1, 3, 512, 512)
dummy_input = torch.randn(*input_shape, dtype=torch.float32)
output_path = f"models/{model_name}.onnx"
print(f"*** export model to {output_path}")
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    input_names=[input_name],
    output_names=["regression", "classification"],
    opset_version=13,
    dynamic_axes={"input": {0: "N"}, "regression": {0: "N"}, "classification": {0: "N"}},
    do_constant_folding=True,
)

print("*** simplify model")
simp_model_name = f"models/{model_name}_simp.onnx"
simplified_model, check = simplify(output_path, test_input_shapes={input_name: input_shape})

print(f"*** saving simplified model {simp_model_name}")
onnx.save(simplified_model, simp_model_name)
