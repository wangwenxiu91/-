import torch.nn as nn

# 定义网络
class Net(nn.Module):
  def __init__(self):
    # nn.Module的子类必须在构造函数中执行父类的构造函数
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, 5) # 依次为输入通道、输出通道和卷积核
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(400, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

net = Net() # 创建网络
# print(net) # 打印网络

# 仅查看网络中的可学习参数（遍历）
for params in net.parameters():
    # print(params)
    # print(params.data) # 获取纯数据
    print(params.shape)

