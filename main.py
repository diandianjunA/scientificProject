import numpy as np
import torch
import torch.nn as nn

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)
y_value = [2 * i + 1 for i in x_values]
y_train = np.array(y_value, dtype=np.float32)
y_train = y_train.reshape(-1, 1)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # 全连接层，输入数据维度和输出数据维度
        self.linear = nn.Linear(input_dim, output_dim)

    # 前向传播
    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

# 使用GPU训练模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 迭代次数
epochs = 1000
# 学习率
learning_rate = 0.01
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# 损失函数
criterion = nn.MSELoss()

for epoch in range(epochs):
    epoch += 1
    # 转成tensor格式的数据
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)
    # 每次迭代梯度清零
    optimizer.zero_grad()
    # 前向传播
    outputs = model(inputs)
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 更新权重参数
    optimizer.step()
    if epoch % 50 == 0:
        print("epoch: {},loss: {}".format(epoch, loss.item()))

# 测试模型预测结果
predicted = model(torch.from_numpy(x_train).requires_grad_().to(device)).data.cpu().numpy()
print(predicted)

# 模型的保存与读取
torch.save(model.state_dict(), "model.pkl")
model.load_state_dict(torch.load("model.pkl"))
