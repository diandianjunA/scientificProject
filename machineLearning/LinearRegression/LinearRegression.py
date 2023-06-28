# Load our data set
import copy
import math

import numpy as np
from matplotlib import pyplot as plt


def compute_cost(x, y, w, b):
    """
    计算线性回归模型的损失函数

    参数:
      x (ndarray (m,)): 数据量为m的样本
      y (ndarray (m,)): 样本数据对应的结果
      w,b (scalar)    : 模型参数

    返回值
      w,b当前值对应的损失函数的值
    """
    # 样本数据量
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


def compute_gradient(x, y, w, b):
    """
    计算线性回归两参数的梯度
    Args:
      x (ndarray (m,)): 数据量为m的样本
      y (ndarray (m,)): 样本数据对应的结果
      w,b (scalar)    : 模型参数
    Returns
      dj_dw (scalar): w的梯度（偏导数）
      dj_db (scalar): b的梯度（偏导数）
     """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        dj_dw += dj_dw_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, num_iters, compute_cost, compute_gradient):
    """
    梯度下降过程更新w,b

    Args:
      x (ndarray (m,)): 数据量为m的样本
      y (ndarray (m,)): 样本数据对应的结果
      w,b (scalar)    : 模型参数

      alpha (float):     学习率
      num_iters (int):   更新多少次
      compute_cost:      计算损失的函数
      compute_gradient:  计算梯度的函数

    Returns:
      w (scalar): 梯度下降更新后的w参数
      b (scalar): 梯度下降更新后的b参数
      J_history (List): 过去的损失值
      p_history (list): 过去的参数值w，b
      """

    w = copy.deepcopy(w_in)  # avoid modifying global w_in
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        # Update Parameters using equation (3) above
        b = b - alpha * dj_db
        w = w - alpha * dj_dw

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost(x, y, w, b))
            p_history.append([w, b])
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} ",
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  f"w: {w: 0.3e}, b:{b: 0.5e}")

    return w, b, J_history, p_history  # return w and J,w history for graphing


x_train = np.array([1.0, 2.0])  # features
y_train = np.array([300.0, 500.0])  # target value
# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha,
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")
# plot cost versus iteration
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)")
ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')
ax2.set_ylabel('Cost')
ax1.set_xlabel('iteration step')
ax2.set_xlabel('iteration step')
plt.show()
