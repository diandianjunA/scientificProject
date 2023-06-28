import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_dw(in_channel, out_channel, stride):
    return nn.Sequential(
        nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, groups=in_channel, bias=True),
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )


def conv(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
                         nn.BatchNorm2d(out_channel),
                         nn.ReLU())


class MobileNetV1(nn.Module):
    def __init__(self, classes=1000):
        super(MobileNetV1, self).__init__()
        self.conv1 = conv(in_channel=3, out_channel=32, kernel_size=3, stride=2, padding=1)
        self.conv_dw1 = conv_dw(in_channel=32, out_channel=64, stride=1)
        self.conv_dw2 = conv_dw(in_channel=64, out_channel=128, stride=2)
        self.conv_dw3 = conv_dw(in_channel=128, out_channel=128, stride=1)
        self.conv_dw4 = conv_dw(in_channel=128, out_channel=256, stride=2)
        self.conv_dw5 = conv_dw(in_channel=256, out_channel=256, stride=1)
        self.conv_dw6 = conv_dw(in_channel=256, out_channel=512, stride=2)
        self.conv_dw7 = conv_dw(in_channel=512, out_channel=512, stride=1)
        self.conv_dw8 = conv_dw(in_channel=512, out_channel=512, stride=1)
        self.conv_dw9 = conv_dw(in_channel=512, out_channel=512, stride=1)
        self.conv_dw10 = conv_dw(in_channel=512, out_channel=512, stride=1)
        self.conv_dw11 = conv_dw(in_channel=512, out_channel=512, stride=1)
        self.conv_dw12 = conv_dw(in_channel=512, out_channel=1024, stride=2)
        self.conv_dw13 = conv_dw(in_channel=1024, out_channel=1024, stride=1)
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.fc = nn.Linear(1024, classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_dw1(out)
        out = self.conv_dw2(out)
        out = self.conv_dw3(out)
        out = self.conv_dw4(out)
        out = self.conv_dw5(out)
        out = self.conv_dw6(out)
        out = self.conv_dw7(out)
        out = self.conv_dw8(out)
        out = self.conv_dw9(out)
        out = self.conv_dw10(out)
        out = self.conv_dw11(out)
        out = self.conv_dw12(out)
        out = self.conv_dw13(out)
        out = self.pool(out)
        out = self.fc(out.view(out.size(0), -1))
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    a = torch.randn(2, 3, 224, 224)
    net = MobileNetV1()
    print(net(a))
    print(net)
