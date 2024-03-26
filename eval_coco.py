from pytorch_quantization import quant_modules
from yolov7.models.yolo import Model
import torch

def load_yolov7_model(weight, device="cpu"):
    ckpt = torch.load(weight, map_location=device)
    model = Model("yolov7/cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model

import collections   
from yolov7.utils.datasets import create_dataloader
def prepare_dataset(cocodir, batch_size=4):
    dataloder = create_dataloader(
        path=f"{cocodir}",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("opt", "single_cls")(False),
        augment=False, hyp=None, rect=False, cache=False,
        stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloder

import yolov7.test as test
from pathlib import Path
import os
def evaluate_coco(path, model, loader, save_dir='.', conf_thres=0.001, iou_thres=0.005):

    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    return test.test(
        data=path,
        save_dir=Path(save_dir),
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        model=model,
        dataloader=loader,
        is_coco=True,
        plots=False,
        half_precision=True,
        save_json=False
    )[0][3]

if __name__ == '__main__':
    weight = "yolov7.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_yolov7_model(weight, device)

    coco128_dir = "datasets/coco128"
    coco2017_dir = "datasets/coco2017"
    coco128_yaml = "yolov7/data/coco128.yaml"
    coco2017_yaml = "yolov7/data/coco2017.yaml"
    dataloder = prepare_dataset(coco128_dir)
    

    ap = evaluate_coco(coco128_yaml, model, dataloder)