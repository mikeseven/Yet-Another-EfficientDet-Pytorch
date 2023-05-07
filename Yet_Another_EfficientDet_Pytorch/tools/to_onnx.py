from pathlib import Path

import onnx
import torch
import yaml
from onnxsim import simplify

from Yet_Another_EfficientDet_Pytorch.backbone import EfficientDetBackbone


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def rebatch(infile, outfile, batch_size="N"):
    # from https://github.com/onnx/onnx/issues/2182#issuecomment-881752539
    import struct

    model = onnx.load(str(infile))
    graph = model.graph

    # if isinstance(batch_size, str):
    #     batch_size = -1

    # Change batch size in input, output and value_info
    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor.type.tensor_type.shape.dim[0].dim_param = batch_size

    # Set dynamic batch size in reshapes (-1)
    for node in graph.node:
        if node.op_type != "Reshape":
            continue

        # found a Reshape op
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

    onnx.save(model, str(outfile))


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

    # save torch model
    print(f"*** saving full torch model to models/{model_name}.pt")
    torch.save(model, f"models/{model_name}.pt")

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
            dynamic_axes={"input": {0: "N"}},
            do_constant_folding=True,
        )

    print("*** simplify model")
    overwrite_input_shapes = {input_name: input_shape}  # we use bs=1 to simplify graph further
    simplified_model, check = simplify(output_path, overwrite_input_shapes=overwrite_input_shapes)

    simp_model_name = Path(f"models/{model_name}_simp.onnx")
    print(f"*** saving simplified model to {simp_model_name}")
    onnx.save(simplified_model, str(simp_model_name))

    # rbatch cannot work as last reshapes already use -1
    # rebatch_name = Path(f"models/{model_name}_simp2.onnx")
    # rebatch(simp_model_name, rebatch_name, batch_size="N")
    # simp_model_name.unlink()
    # rebatch_name.rename(simp_model_name)
