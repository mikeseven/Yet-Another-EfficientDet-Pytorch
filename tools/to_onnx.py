import torch
from onnxsim import simplify
import onnx
from backbone import EfficientDetBackbone
import yaml


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


device = "cuda" if torch.cuda.is_available() else "cpu"
params = Params(f"projects/coco.yml")

d_level = 0
model_name = f"efficientdet-d{d_level}"

model = EfficientDetBackbone(
    num_classes=len(params.obj_list),
    compound_coef=d_level,
    onnx_export=True,
    ratios=eval(params.anchors_ratios),
    scales=eval(params.anchors_scales),
)
model = model.to(device=device)
model.load_state_dict(torch.load(f"weights/{model_name}.pth", map_location=device))
model.requires_grad_(False)
model.eval()

input_name = "input"
input_shape = (1, 3, 512, 512)
dummy_input = torch.randn(*input_shape, dtype=torch.float32).to(device=device)
output_path = f"models/{model_name}.onnx"
print(f"*** export model to {output_path}")
with torch.inference_mode():
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
overwrite_input_shapes = {input_name: input_shape}  # or None to preserve dynamic batch size
simplified_model, check = simplify(output_path, overwrite_input_shapes=overwrite_input_shapes)

print(f"*** saving simplified model {simp_model_name}")
onnx.save(simplified_model, simp_model_name)
