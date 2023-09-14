from pathlib import Path
import tvm
from tvm import relay
import numpy as np
import torch
from Yet_Another_EfficientDet_Pytorch.backbone import EfficientDetBackbone

try:
    from rich import print
except ModuleNotFoundError:
    pass

# fmt:off
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
# fmt:on


def get_model(compound_coef=0, device="cpu"):
    model_name = f"efficientdet-d{compound_coef}"
    weights_path = f"weights/{model_name}.pth"
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list), onnx_export=True).to(device)
    model.backbone_net.model.set_swish(False)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def export(model, compound_coef=0, device="cpu"):
    print("### convert to torchscript")
    input_size = EfficientDetBackbone.input_sizes[compound_coef]
    input_shape = (1, 3, input_size, input_size)
    dummy_img = np.random.random(input_shape).astype(np.float32)
    dummy_input = torch.as_tensor(dummy_img).to(device=device)
    scripted_model = torch.jit.trace(model, dummy_input)
    scripted_model.eval()

    print("### tvm import")
    input_name = "data"
    mod, params = relay.frontend.from_pytorch(scripted_model, [(input_name, input_shape)])
    return mod, params


def tvm_build(model_name, mod, params, target_device):
    print("### Compile...")
    if target_device == "cuda":
        target = tvm.target.cuda(model="a100", arch="sm_80", options="-libs=cudnn,cublas")
        target_host = "llvm"
    else:
        target = "llvm -mcpu=native -opt-level=2"
        target_host = "llvm"
        ctx = tvm.cpu()

    with relay.build_config(opt_level=3):
        lib = relay.build(mod, params=params, target=tvm.target.Target(target, host=target_host))

    print(f"### exporting library to {model_name}.{target_device}.so")
    lib_path = Path(f"{model_name}_{target_device}.so")
    lib.export_library(lib_path)
    return lib_path


def bench(lib_path, input_name, input_shape):
    from tvm.contrib import graph_executor
    import time

    print("### TVM inference")
    ctx = tvm.cuda() if lib_path.stem.endswith("cuda") else tvm.cpu()
    loaded_lib = tvm.runtime.load_module(lib_path)
    m = graph_executor.GraphModule(loaded_lib["default"](ctx))

    num_frames = 100
    dummy_img = np.random.randn(*input_shape).astype("float32")
    avg_time = 0
    for i in range(num_frames):
        start = time.time()
        m.run(**{input_name: tvm.nd.array(dummy_img, ctx)})
        tvm_output = m.get_output(0)
        avg_time += time.time() - start
    avg_time /= num_frames
    print(f"TVM {ctx=} {avg_time=} sec")


"""
benchmark on d0: cuda 8ms, cpu 28ms
"""
# target_device = "cuda"
target_device = "x86"
device = "cuda" if torch.cuda.is_available() else "cpu"

input_name = "data"
compound_coef = 0

model_name = f"efficientdet-d{compound_coef}"
max_size = EfficientDetBackbone.input_sizes[compound_coef]
input_shape = (1, 3, max_size, max_size)

model = get_model(compound_coef, device)
mod, params = export(model, compound_coef, device)
lib_path = tvm_build(model_name, mod, params, target_device)
bench(lib_path, input_name, input_shape)
