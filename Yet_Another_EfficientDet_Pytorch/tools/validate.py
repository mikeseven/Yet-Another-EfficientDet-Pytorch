import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm.auto import tqdm

from Yet_Another_EfficientDet_Pytorch.backbone import EfficientDetBackbone
from Yet_Another_EfficientDet_Pytorch.efficientdet.utils import BBoxTransform, ClipBoxes
from Yet_Another_EfficientDet_Pytorch.utils.utils import invert_affine, postprocess, preprocess

try:
    from rich import print
except ModuleNotFoundError:
    pass


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


class CocoDS(Dataset):
    def __init__(self, root, ann, max_size=512):
        super().__init__()
        self.root = Path(root)
        self.coco = COCO(str(ann))
        self.image_ids = self.coco.getImgIds()
        self.max_size = max_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = self.root / image_info["file_name"]

        _, framed_imgs, framed_metas = preprocess(str(image_path), max_size=self.max_size)
        img = torch.from_numpy(framed_imgs[0]).permute(2, 0, 1)  # HWC to CHW
        framed_metas = torch.as_tensor(framed_metas[0])
        return img, framed_metas, image_id


def precalc_anchors(input_shape=(1, 3, 512, 512), d_level=0):
    from Yet_Another_EfficientDet_Pytorch.backbone import Anchors

    anchors_mod = Anchors(
        anchor_scale=EfficientDetBackbone.anchor_scale[d_level],
        pyramid_levels=(torch.arange(EfficientDetBackbone.pyramid_levels[d_level]) + 3).tolist(),
    )

    inputs = torch.randn(*input_shape, dtype=torch.float32)
    anchors = anchors_mod(inputs, inputs.dtype)
    return anchors


# torch modules to compute boxes in postprocessing, stateless
regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()


def postprocess_image_results(
    results,
    img,
    framed_metas,
    image_id,
    anchors,
    regression,
    classification,
    nms_threshold=0.5,
    threshold=0.05,
):
    # convert numpy tensors to torch tensors
    regression = torch.from_numpy(regression)
    classification = torch.from_numpy(classification)
    classification = torch.sigmoid(classification)  # apply sigmoid as post-process

    preds = postprocess(
        img,
        anchors,
        regression,
        classification,
        regressBoxes,
        clipBoxes,
        threshold,
        nms_threshold,
    )

    if not preds:
        return

    preds = invert_affine([framed_metas], preds)[0]

    scores = preds["scores"]
    class_ids = preds["class_ids"]
    rois = preds["rois"]

    if rois.shape[0] > 0:
        # x1,y1,x2,y2 -> x1,y1,w,h
        rois[:, 2] -= rois[:, 0]
        rois[:, 3] -= rois[:, 1]

        bbox_score = scores

        for roi_id in range(rois.shape[0]):
            score = float(bbox_score[roi_id])
            label = int(class_ids[roi_id])
            box = rois[roi_id, :]

            image_result = {
                "image_id": int(image_id),
                "category_id": label + 1,
                "score": float(score),
                "bbox": box.tolist(),
            }
            results.append(image_result)


def validate_onnx(onnx_model_path, val_path, ann_file, results_file_json, batch_size=10, gpu_id=0):
    onnx_model = onnx.load(onnx_model_path)
    input = onnx_model.graph.input[0]
    input_name = input.name
    input_shape = tuple([d.dim_value for d in input.type.tensor_type.shape.dim])
    input_shape = (1,) + input_shape[1:]

    # CUDAExecutionProvider requires pip install onnxruntime-gpu --no-deps
    ort_session = ort.InferenceSession(
        model_path,
        providers=[
            ("CUDAExecutionProvider", {"device_id": gpu_id}),
            "CPUExecutionProvider",
        ],
    )
    ort_device = ort.get_device()

    dataset = CocoDS(val_path, ann_file)

    # loader is not put on GPU to avoid round trip to cpu to pass data to ORT
    # this also avoids a synchronization issue between loader and ORT
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # slower to use multiprocessing on CPU
        drop_last=False,
    )

    # precalc anchors for 512x512 inputs, could be saved as numpy array
    anchors = precalc_anchors(input_shape=input_shape)
    results = []

    with torch.inference_mode():
        for batch in tqdm(loader):
            imgs, metas, image_ids = batch
            outputs_ort = ort_session.run(None, {input_name: imgs.cpu().numpy()})
            regression_, classification_ = outputs_ort

            batch_size = imgs.shape[0]
            for batch_id in range(batch_size):
                framed_metas = metas[batch_id]  # for bs=1
                image_id = image_ids[batch_id]
                classification = np.expand_dims(classification_[batch_id], 0)
                regression = np.expand_dims(regression_[batch_id], 0)
                img = np.expand_dims(imgs[batch_id], 0)

                postprocess_image_results(
                    results,
                    img,
                    framed_metas,
                    image_id,
                    anchors,
                    regression,
                    classification,
                )

    # write output
    print(f"*** writing results to {results_file_json}")
    with Path(results_file_json).open("wt", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def validate_tvm(onnx_model_path, val_path, ann_file, results_file_json, batch_size=10, gpu_id=0):
    from tvm import relay
    from tvm.relay import transform as rt
    import tvm
    import cloudpickle as pkl

    # all outputs are next to ONNX file
    onnx_model_path = Path(onnx_model_path)
    relay_path = onnx_model_path.with_suffix(".relay")

    use_lib = False  # whether to use TVM generated .so (TODO: slow???)
    path_lib = onnx_model_path.with_suffix(".so")

    if not relay_path.exists():
        print("**** Importing ONNX model to TVM")
        onnx_model = onnx.load(str(onnx_model_path))

        input_shapes = {
            n.name: [1 if d.dim_value == 0 else d.dim_value for d in n.type.tensor_type.shape.dim]
            for n in onnx_model.graph.input
        }
        input_name, input_shape = next(iter(input_shapes.items()))

        mod, params = relay.frontend.from_onnx(onnx_model, shape=input_shapes, freeze_params=True)
        if params:
            func = relay.build_module.bind_params_by_name(func, params)
            mod = tvm.IRModule.from_expr(func)

        print("**** Model simplification for inference")
        with tvm.transform.PassContext(opt_level=3):
            input_dict = {p: 0 for p in mod["main"].params}
            transforms = [
                rt.InferType(),
                rt.ChangeBatch(input_dict, batch_size=batch_size),
                rt.CanonicalizeOps(),  # bias_add to add
                rt.SimplifyInference(),
                # fuse conv & bn etc.
                rt.FoldConstant(),
                rt.FoldScaleAxis(),
                rt.SimplifyExpr(),
                rt.FoldConstant(),
                rt.FoldScaleAxis(),
            ]
            for xf in (pbar := tqdm(transforms)):
                pbar.set_description(f"{xf}")  # pass description
                mod = xf(mod)

        with relay_path.open("wb") as f:
            pkl.dump(mod, f, protocol=-1)

    with relay_path.open("rb") as f:
        mod = pkl.load(f)
    print(f'{mod["main"]=}')
    input_name = mod["main"].params[0].name_hint
    input_shape = mod["main"].params[0].checked_type.concrete_shape
    print(f"TVM module: {input_name=} {input_shape}")
    freqs = relay.analysis.list_op_freqs(mod)
    freqs = {str(name): int(v) for name, v in freqs.items()}
    print(f"relay ops {freqs=}")

    if use_lib and not path_lib.exists():
        print("**** Compile model")
        target = "llvm"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target)
        lib.export_library(path_lib)

    if use_lib:
        from tvm.contrib import graph_executor

        target = "llvm"
        loaded_lib = tvm.runtime.load_module(path_lib)
        executor = graph_executor.GraphModule(loaded_lib["default"](tvm.device(target)))
    else:
        print("**** Compile model")
        target = "llvm -mcpu=native --opt-level=2"
        with tvm.transform.PassContext(opt_level=3):
            executor = relay.create_executor(
                "graph",
                mod=mod,
                device=tvm.device(target),
                target=tvm.target.Target(target, host="llvm"),
            ).evaluate()

    dataset = CocoDS(val_path, ann_file)
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,  # TODO: seems the last batch comes with 49 images???
    )

    # precalc anchors, could be saved as numpy array
    anchors = precalc_anchors(input_shape)

    results = []
    max_batches = len(loader)
    print(f"{max_batches=}")
    for batch in tqdm(loader):
        imgs, metas, image_ids = batch
        if use_lib:
            executor.run(**{input_name: imgs.cpu().numpy()})
            regression_ = executor.get_output(0).numpy()
            classification_ = executor.get_output(0).numpy()
        else:
            regression_, classification_ = executor(**{input_name: imgs.cpu().numpy()})
            regression_ = regression_.numpy()
            classification_ = classification_.numpy()

        for batch_id in range(batch_size):
            framed_metas = metas[batch_id]
            image_id = image_ids[batch_id]
            classification = np.expand_dims(classification_[batch_id], 0)
            regression = np.expand_dims(regression_[batch_id], 0)
            img = np.expand_dims(imgs[batch_id], 0)

            postprocess_image_results(
                results,
                img,
                framed_metas,
                image_id,
                anchors,
                regression,
                classification,
            )

    # write output
    print(f"*** writing results to {results_file_json}")
    with Path(results_file_json).open("wt", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def eval_results_file(ann_file, results_file, max_batches=-1):
    def _eval(coco_gt: COCO, image_ids: list[int], pred_json_path: str, iou_type="bbox"):
        from pycocotools.cocoeval import COCOeval

        # load results in COCO evaluation tool
        coco_pred = coco_gt.loadRes(str(pred_json_path))

        # run COCO evaluation
        coco_eval = COCOeval(coco_gt, coco_pred, iou_type)
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    # evaluate on all val images
    print("*** Evaluating results")
    coco_gt = COCO(str(ann_file))
    image_ids = coco_gt.getImgIds()[:max_batches]
    _eval(coco_gt, image_ids, results_file)


def test_loader(batch_size):
    dataset = CocoDS(val_path, ann_file)
    loader = DataLoader(
        dataset,
        sampler=SequentialSampler(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,  # TODO: seems the last batch comes with 49 images???
    )

    max_samples = len(dataset)
    max_batches = len(loader)
    num_image_ids = len(dataset.coco.getImgIds())
    print(f"{max_samples=} {max_batches=} {num_image_ids=}")
    assert num_image_ids == max_samples
    assert max_batches == num_image_ids // batch_size
    for batch_idx, batch in enumerate(tqdm(loader)):
        imgs, metas, image_ids = batch
        assert imgs.shape[0] == batch_size, f"batch {batch_idx} has only {imgs.shape[0]} items out of {batch_size}"


if __name__ == "__main__":
    params = Params(f"projects/coco.yml")

    coco_root = f"datasets/{params.project_name}"
    val_path = f"{coco_root}/{params.val_set}"
    ann_file = f"{coco_root}/annotations/instances_{params.val_set}.json"

    d_level = 0
    model_name = f"efficientdet-d{d_level}"
    model_path = f"models/{model_name}_simp.onnx"

    # test_loader(batch_size=50)

    # efficientdet-d0 mAP=0.332
    ### validate ONNX
    # results_file_json = f"{model_name}_bbox_onnx_results.json"
    # validate_onnx(model_path, val_path, ann_file, results_file_json=results_file_json, batch_size=50)
    # eval_results_file(ann_file, results_file_json)

    ### validate TVM
    results_file_json = f"{model_name}_bbox_tvm_results.json"
    validate_tvm(model_path, val_path, ann_file, results_file_json=results_file_json, batch_size=50)
    eval_results_file(ann_file, results_file_json)
