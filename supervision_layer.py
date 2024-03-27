import quantize
from copy import deepcopy

ptq_model = quantize.prepare_model("yolov7.pt", "cuda:0")
quantize.replace_to_quantization_model(ptq_model, "model\.105\.m\.(.*)")

# 省略标定步骤

origin_model = deepcopy(ptq_model).eval()
quantize.disable_quantization(origin_model).apply() # 关闭量化操作获取原始模型

print(ptq_model.model)
supervision_list = []
for item in ptq_model.model:
    print(item)
    supervision_list.append(id(item))

keep_idx = list(range(0, len(ptq_model.model) - 1, 1)) # -1: 模型最后一层不需要Loss计算和反向传播
keep_idx.append(len(ptq_model.model) - 2) # -2: 保证keep_idx不为空

def match_model(name, module):
    if id(module) not in supervision_list:
        return False

    idx = supervision_list.index(id(module))

    if idx in keep_idx:
        print(f"Supervision: {name} will compute loss ...")
    else:
        print(f"Supervision: {name} not compute loss ...")
    
    return idx in keep_idx # Truse/False


from pytorch_quantization import nn as quant_nn
# 定义一个list装PTQ模型层和Origin模型层匹配对
ptq_origin_layer_pairs = []
# 遍历PTQ模型层和Origin模型层的每一个模块
for ((ptq_name, ptq_module), (origin_name, origin_module)) in zip(ptq_model.named_modules(), origin_model.named_modules()):
    
    print("ptq_name: ", ptq_name)
    print("type(ptq_module): ", type(ptq_module))

    # 由于TensorQuantizer里面都是scale和标定算法超参, 不需要进行loss计算和反向传播, 所以遇到该层直接跳过
    if isinstance(ptq_model, quant_nn.TensorQuantizer): 
        continue
    if not match_model(ptq_name, ptq_module): # 没有匹配直接跳过
        continue
    ptq_origin_layer_pairs.append([ptq_module, origin_module])

# print(ptq_origin_layer_pairs)

