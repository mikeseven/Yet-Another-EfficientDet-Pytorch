import torch
from onnxsim import simplify
import onnx
from backbone import EfficientDetBackbone
import yaml
from pathlib import Path


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def change_batch_size(model, batch_size: str | int = "N"):
    import struct

    # model = onnx.load(str(in_model_path))
    graph = model.graph

    # Change batch size in input, output and value_info
    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor.type.tensor_type.shape.dim[0].dim_param = batch_size

    # Set dynamic batch size in reshapes (-1)
    for node in graph.node:
        if node.op_type != "Reshape":
            continue
        for init in graph.initializer:
            # node.input[1] is expected to be a reshape
            if init.name != node.input[1]:
                continue
            # Shape is stored as a list of ints
            if len(init.int64_data) > 0:
                # This overwrites bias nodes' reshape shape but should be fine
                init.int64_data[0] = -1
            # Shape is stored as bytes
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into("q", shape, 0, -1)
                init.raw_data = bytes(shape)
    return model


if __name__ == "__main__":
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

    # these names can be changed to whatever needed
    input_name = "input"
    output_names = ["regression", "classification"]

    input_size = EfficientDetBackbone.input_sizes[d_level]
    input_shape = (1, 3, input_size, input_size)
    dummy_input = torch.randn(*input_shape, dtype=torch.float32).to(device=device)
    output_path = f"models/{model_name}.onnx"
    print(f"*** export model to {output_path}")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=[input_name],
            output_names=output_names,
            opset_version=13,
            dynamic_axes={"input": {0: "N"}, "regression": {0: "N"}, "classification": {0: "N"}},
            do_constant_folding=True,
        )

    print("*** simplify model")
    simp_model_name = f"models/{model_name}_simp.onnx"
    overwrite_input_shapes = {input_name: input_shape}  # we use bs=1 to simplify graph further
    simplified_model, check = simplify(output_path, overwrite_input_shapes=overwrite_input_shapes)

    print(f"*** saving simplified model to {simp_model_name}")
    # we restore dynamic batch size everywhere
    simplified_model = change_batch_size(simplified_model, batch_size="N")
    onnx.save(simplified_model, simp_model_name)
