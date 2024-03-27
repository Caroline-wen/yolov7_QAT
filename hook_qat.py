import torch
import torch.nn as nn

# 定义一个示例模块
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = MyModule()

# 定义前向传播的hook函数
def forward_hook(module, input, output):
    print(f"Inside forward_hook for {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")

# 注册前向传播的hook
hook_handle = model.fc1.register_forward_hook(forward_hook)

# 准备输入数据并进行模型的前向传播
input_data = torch.randn(2, 10)
output = model(input_data)

# 注销前向传播
hook_handle.remove()

# QAT: loss(PTQ_model and Origin_model) ==》 update: PTQ_model
# PTQ_model and Origin_model 网络层/模块 配对(在这先假设一个匹配结果)
ptq_origin_layer_pairs = [[ptq_layer0, origin_layer0], [ptq_layer1, origin_layer1], [ptq_layer2, origin_layer2], ...]

# 分别定义一个list接收PTQ模型输出和Origin模型输出
ptq_outputs = []
origin_outputs = []

# 定义hook函数返回forward_hook(闭包)
def make_layer_forward_hook(module_outputs):
    def forward_hook(module, input, output):
        module_outputs.append(output)
    return forward_hook

remove_handle = []
# 为网络每一层注册hook
for ptq_m, ori_m in ptq_origin_layer_pairs:
    remove_handle.append(ptq_m.register_forward_hook(make_layer_forward_hook(ptq_outputs)))
    remove_handle.append(ori_m.register_forward_hook(make_layer_forward_hook(origin_outputs)))

# ptq模型前向
ptq_model(imgs)
# oring模型前向
origin_model(imgs)   

# 计算ptq和origin的loss
loss = 0.
for index, (ptq_out, ori_out) in enumerate(zip(ptq_outputs, origin_outputs)):
    loss += loss_function(ptq_out, ori_out)

# remove hook handle
for rm in remove_handle:
    rm.remove()