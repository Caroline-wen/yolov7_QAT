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