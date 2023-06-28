import torch
import torchvision
from PIL import Image

from Network.lenet5 import Lenet5

image_path = "../image/plane.png"
image = Image.open(image_path)  # 加载3通道的图片数据，3阶张量
print(image)  # <PIL.Image.Image train mode=RGB size=352x261 at 0x1E61DE1FE50>

# 将图片 Resize(缩放) 到32x32尺寸，适合模型输入，最后在转化为Tensor实例
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                             std=[0.229, 0.224, 0.225])])
image = transform(image)
# 转化为4阶张量（模型网络的输入要求张量的阶数为4）
image = torch.reshape(image, (1, 3, 32, 32))
print(image.shape)  # torch.Size([1, 3, 32, 32])
classes = ('plane', 'car', 'bird', 'cat',  # 这是该数据的类别
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

net = Lenet5()
PATH = '../data/cifar_net.pth'  # 权重路径
net.load_state_dict(torch.load(PATH))  # 加载训练好的模型
net.eval()
with torch.no_grad():  # 节约内存、性能
    output = net(image)
print(output)

# 预测输出：tensor([[ 2.2722, -1.4881, -0.8504,  1.8632,  0.8308,  1.0495,  0.6538, -1.5160,
#          -0.6486, -1.6084]])
# 第0类最大 tensor([0]) 不准确，dog是类5
print(classes[output.argmax(1)])
