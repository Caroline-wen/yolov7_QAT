from pytorch_quantization import quant_modules
from yolov7.models.yolo import Model
import torch

from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn

def load_yolov7_model(weight, device="cpu"):
    ckpt = torch.load(weight, map_location=device)
    model = Model("yolov7/cfg/training/yolov7.yaml", ch=3, nc=80).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model

from pytorch_quantization.tensor_quant import QuantDescriptor
from absl import logging as quant_logging
def initialize():
    '''
    function: input:Max ===》Histogram
    '''
    quant_desc_intput = QuantDescriptor(calib_method="histogram")
    quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_intput)
    quant_nn.QuantMaxPool2d.set_default_quant_desc_input(quant_desc_intput)
    quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_intput)

    quant_logging.set_verbosity(quant_logging.ERROR)

def prepare_model(weight, device):
    '''
    function: 自动插入量化节点
    '''
    # quant_modules.initialize()
    initialize()
    model = load_yolov7_model(weight, device)
    model.float()
    model.eval()
    with torch.no_grad(): # 使用上下文管理器, 禁止梯度计算
        model.fuse() # conv和bn合并, 提速
    return model

def tranfer_torch_to_quantization(nn_instance, quant_module):
    quant_instance = quant_module.__new__(quant_module)
    for k, val in vars(nn_instance).items():
        setattr(quant_instance, k, val)
    
    def __init__(self):
        # 返回两个量化描述符实例
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            self.init_quantizer(quant_desc_input)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator): # 判断校准方式是否为直方图
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                    self._input_quantizer._calibrator._torch_hist = True
                    self._weight_quantizer._calibrator._torch_hist = True
    
    __init__(quant_instance)
    return quant_instance

import re
def quantization_ignore_match(ignore_layer, path):
    if ignore_layer is None:
        return False
    if isinstance(ignore_layer, str) or isinstance(ignore_layer, list):
        if isinstance(ignore_layer, str):
            ignore_layer = [ignore_layer]
        if path in ignore_layer:
            return True
        for item in ignore_layer:
            if re.match(item, path):
                return True
    return False



def torch_model_find_quant_module(module, module_dict, ignore_layer, prefix=''):
    for name in module._modules:
        submodule = module._modules[name]
        path = name if prefix == '' else prefix + '.' + name
        torch_model_find_quant_module(submodule, module_dict, ignore_layer, prefix=path)

        submodule_id = id(type(submodule))
        if submodule_id in module_dict:
            ignored = quantization_ignore_match(ignore_layer, path)
            if ignored:
                print(f"Quantization: {path} has ignored.")
                continue
            # 转换
            module._modules[name] = tranfer_torch_to_quantization(submodule, module_dict[submodule_id])

def replace_to_quantization_model(model, ignore_layer=None):
    '''
    function: 手动插入量化节点
    '''
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    torch_model_find_quant_module(model, module_dict, ignore_layer)
    

import collections   
from yolov7.utils.datasets import create_dataloader
def prepare_val_dataset(cocodir, batch_size=4):
    dataloder = create_dataloader(
        f"{cocodir}/val2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("opt", "single_cls")(False),
        augment=False, hyp=None, rect=False, cache=False,
        stride=32, pad=0.5, image_weights=False
    )[0]
    return dataloder

import yaml
def prepare_train_dataset(cocodir, batch_size=4):

    with open("yolov7/data/hyp.scratch.p5.yaml") as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)


    dataloder = create_dataloader(
        f"{cocodir}/train2017.txt",
        imgsz=640,
        batch_size=batch_size,
        opt=collections.namedtuple("opt", "single_cls")(False),
        augment=False, hyp=hyp, rect=False, cache=False,
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

def collect_stats(model, dataloader, device, num_batch=200):
    model.eval()

    # 开启校准器
    for name, module in model.named_modules():
        if isinstance(model, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()
    
    # test
    with torch.no_grad():
        for i in enumerate(data_loader):
            imgs = datas[0].to(device, non_bloking=True).float()/255.0
            model(imgs)

            if i >= num_batch:
                break
    
    # 关闭校准器
    for name, module in model.named_modules():
        if isinstance(model, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.diable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
                module._amax = module._amax.to(device)


def calibration_model(model, dataloader, device):
    # 收集信息
    collect_stats(model, dataloader, device)

    # 获取动态范围, 计算amax和scale值
    compute_amax(model, method='mes')


def export_ptq(model, save_file, device, dynamic_batch=False):
    input_dummy = torch.randn(1, 3, 640, 640, device=device)
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, input_dummy, save_file, opset_version=13,
            input_names=['input'], output_names=['ouptput'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if dynamic_batch else None,
        )
    quant_nn.TensorQuantizer.use_fb_fake_quant = False

def have_quantizer(layer):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            return True

class disable_quantization():
    def __init__(self, model):
        self.model = model
    
    def apply(self, disabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled
    
    def __enter__(self):
        self.apply(disabled=True)
    
    def __exit__(self, *args, **kwargs):
        self.apply(disabled=False)


class enable_quantization():
    def __init__(self, model):
        self.model = model
    
    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled
    
    def __enter__(self):
        self.apply(enabled=True)
        return self
    
    def __exit__(self, *args, **kwargs):
        self.apply(enabled=False)

import json
class SummaryTool:
    def __init__(self, file):
        self.file = file
        self.data = []

    def append(self, item):
        self.data.append(item)
        json.dump(self.data, open(self.file, "w"), indent=4)



def sensitive_analysis(model, loader):
    '''
        1.for循环model的每一个quantizer层(model:自动插入QDQ后的模型)
        2.关闭量化层,其余层的量化保留
        3.验证模型的精度,并保存精度值(用evaluate_coco()计算AP和mAP)
        4.验证结束,重启该层的量化操作(还原层)
        5.for循环结束,得到所有层的精度值
        6.排序,得到前10个对精度影响比较大的层,将这些层进行打印
    '''
    save_file = "sensitive_analysis.json"
    summary = SummaryTool(save_file)


    # for循环每一个层
    print("sensitive analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        # 判断layer是否为量化层
        if have_quantizer(layer):  # 如果为量化层
            # 使该层量化失效, 不做Int8, 使用FP16运算
            disable_quantization(layer).apply()
            # 计算map值
            ap = evaluate_coco(model, loader)

            # 保存精度值(JSON文件)
            summary.append([ap, f"model.{i}"])

            # 重启该层的量化
            enable_quantization(layer).apply()
            print(f"layer {i} ap: {ap}")
        else:
            print(f"ignore model.{i} because it is {type(layer)}")

    # 循环结束, 打印前10个影响比较大的层
    summary = sorted(summary.data, key=lambda x: x[0], reverse=True)
    print("Sensitive Summary: ")
    for n, (ap, name) in range(0, 10):
        print(f"Top{n}: Using FP16 {name}, ap = {ap:.5f}")


if __name__ == '__main__':
    weight = "yolov7.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载数据
    print("Evaluate Dataset...")
    cocodir = "datasets/coco128"
    val_dataloder = prepare_val_dataset(cocodir)
    train_dataloder = prepare_train_dataset(cocodir)

    # # 加载pth(pytorch)模型
    # print("Load Origin pytorch model...")
    # pth_model = load_yolov7_model(weight, device)
    # # pth模型验证
    # print("Evaluate Origin pytorch model...")
    # pth_ap = evaluate_coco(pth_model, dataloder)

    # 获取伪量化模型(手动插入QDQ, 手动Initial)
    qnt_auto_model = prepare_model(weight, device)
    # replace_to_quantization_model(qnt_auto_model)
    
    # 模型标定
    # calibration_model(qnt_auto_model, train_dataloder, device)

    # 敏感层分析
    # sensitive_analysis(qnt_auto_model, val_dataloder)

    # 如何处理敏感层分析出的结果:将影响较大的层关闭量化,使用fp16进行计算
    # 所以在进行PTQ量化之前就要进行敏感层的分析,得到影响较大的层,然后在手动插入量化节点
    # 的时候将这些影响层进行量化的关闭

    ignore_layer = [
        "model\.104\.(.*)", "model\.37\.(.*)","mnodel\'2\'.(*)","model\.1\.(.*)","model\.77\.(.*)",
        "model\.99\.(.*)","model\.70\.(.*)","model\.95\.(.*)","odel\.92\.(.*)","model\.81\.(.*)"
    ]
    replace_to_quantization_model(qnt_auto_model, ignore_layer)
    print(qnt_auto_model)


    # # 导出PTQ模型的ONNX文件
    # print("Export PTQ model...")
    # export_ptq(qnt_auto_model, "ptq_yolov7.onnx", device)

    # # PTQ模型验证
    # print("Evaluate PTQ model...")
    # ptq_ap = evaluate_coco(qnt_auto_model, dataloder)