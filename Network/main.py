import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim

from lenet5 import Lenet5
from resnet import ResNet18


def main():
    batchsz = 128

    cifar_train = datasets.CIFAR10('cifar', True, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ]), download=True)
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    x, label = next(iter(cifar_train))
    print('x:', x.shape, 'label:', label.shape)

    device = torch.device('cuda')
    model = Lenet5().to(device)
    # model = ResNet18().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    for epoch in range(1000):
        running_loss = 0.0
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            # [b, 3, 32, 32]
            # [b]
            x, label = x.to(device), label.to(device)

            logits = model(x)
            # logits: [b, 10]
            # label:  [b]
            # loss: tensor scalar
            loss = criteon(logits, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()  # .item()这也属于迭代器，将每一步的损失取出来
            if batchidx % 50 == 0:
                print('[%d, %5d] loss: %f' %
                      (epoch + 1, batchidx + 1, running_loss / 2000))
                running_loss = 0.0

    PATH = '../data/cifar_net.pth'  # 权重路径
    torch.save(model.state_dict(), PATH)


if __name__ == '__main__':
    main()
