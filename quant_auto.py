from pytorch_quantization import quant_modules
from yolov7.models.yolo import Model
import torch

def load_yolov7_model(weight, device="cpu"):
    ckpt = torch.load(weight, map_location=device)
    model = Model("yolov7/cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model


def prepare_model(weight, device):
    '''
    description: 自动插入量化节点
    '''
    quant_modules.initialize()
    model = load_yolov7_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad(): # 使用上下文管理器, 禁止梯度计算
        model.fuse() # conv和bn合并, 提速
    return model



import collections   
from yolov7.utils.datasets import create_dataloader
def prepare_dataset(cocodir, batch_size=4):
    dataloder = create_dataloader(
        f"{cocodir}",
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
def evaluate_coco(model, loader, save_dir='.', conf_thres=0.001, iou_thres=0.005):

    if save_dir and os.path.dirname(save_dir) != "":
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    return test.test(
        "yolov7/data/coco128.yaml",
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

    pth_model = load_yolov7_model(weight, device) # torch模型
    print("pth_model: ", pth_model)
    qnt_model = prepare_model(weight, device) # 加入量化节点后的模型
    print("qnt_model: ", qnt_model)

    # cocodir = "datasets/coco128"
    # dataloder = prepare_dataset(cocodir)
    # ap = evaluate_coco(qnt_model, dataloder)