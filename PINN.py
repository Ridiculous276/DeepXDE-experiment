"""
Lotka-Volterra PINN 实验完整代码 (PyTorch 专用版)
环境：DeepXDE + PyTorch (既然你已经装了torch，直接运行这个)
功能：一键运行不同配置，自动保存对比图片
"""
import os
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 【关键设置】强制告诉 DeepXDE 使用 PyTorch，不要去找 TensorFlow
os.environ["DDE_BACKEND"] = "pytorch"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import torch  # 导入你已经安装好的 PyTorch

# ==============================================================================
# ▼▼▼ 实验控制面板 (每次运行前修改这里) ▼▼▼
# ==============================================================================

# 1. 网络架构选择 (三选一: 去掉 # 号即为选中)
EXP_ARCH = "BASELINE"   # 基准: [8] + [64]*6 + [2]
#EXP_ARCH = "WIDE"     # 宽网络: [8] + [128]*3 + [2]
#EXP_ARCH = "DEEP"     # 深网络: [8] + [32]*10 + [2]

# 2. 采样策略选择 (三选一: 去掉 # 号即为选中)
# EXP_SAMPLING = "BASELINE" # 基准: 3000个点，随机采样
#EXP_SAMPLING = "SPARSE"   # 稀疏: 200个点，随机采样
EXP_SAMPLING = "UNIFORM"  # 均匀: 2000个点，均匀网格

# ==============================================================================
# ▲▲▲ 配置结束，以下代码无需修改 ▲▲▲
# ==============================================================================

print(f"--- 开始运行实验 ---")
print(f"当前配置: 架构=[{EXP_ARCH}] | 采样=[{EXP_SAMPLING}]")

# 1. 问题定义 (Lotka-Volterra)
ub = 200
rb = 20

def func(t, r):
    x, y = r
    dx_t = 1 / ub * rb * (2.0 * ub * x - 0.04 * ub * x * ub * y)
    dy_t = 1 / ub * rb * (0.02 * ub * x * ub * y - 1.06 * ub * y)
    return dx_t, dy_t

def gen_truedata():
    """生成真实解用于画图对比"""
    t = np.linspace(0, 1, 100)
    sol = integrate.solve_ivp(func, (0, 10), (100 / ub, 15 / ub), t_eval=t)
    x_true, y_true = sol.y
    x_true = x_true.reshape(100, 1)
    y_true = y_true.reshape(100, 1)
    return x_true, y_true

def ode_system(x, y):
    """PINN 残差定义"""
    r = y[:, 0:1]
    p = y[:, 1:2]
    dr_t = dde.grad.jacobian(y, x, i=0)
    dp_t = dde.grad.jacobian(y, x, i=1)
    return [
        dr_t - 1 / ub * rb * (2.0 * ub * r - 0.04 * ub * r * ub * p),
        dp_t - 1 / ub * rb * (0.02 * r * ub * p * ub - 1.06 * p * ub),
    ]

# 2. 数据集配置
geom = dde.geometry.TimeDomain(0, 1.0)

# 根据控制面板设置采样点
num_domain = 3000   # 默认
anchors = None      # 默认

if EXP_SAMPLING == "SPARSE":
    num_domain = 200
    print("-> 应用: 稀疏采样 (200点)")
elif EXP_SAMPLING == "UNIFORM":
    num_domain = 0  # 使用 anchors 时需设为0
    anchors = np.linspace(0, 1.0, 2000).reshape(-1, 1)
    print("-> 应用: 均匀采样 (2000点)")

data = dde.data.PDE(
    geom,
    ode_system,
    [],
    num_domain=num_domain,
    num_boundary=2,
    num_test=3000,
    anchors=anchors
)

# 3. 网络架构配置
if EXP_ARCH == "BASELINE":
    layer_size = [8] + [64] * 6 + [2]
elif EXP_ARCH == "WIDE":
    layer_size = [8] + [128] * 3 + [2]
elif EXP_ARCH == "DEEP":
    layer_size = [8] + [32] * 10 + [2]

activation = "tanh"
initializer = "Glorot normal"
net = dde.nn.FNN(layer_size, activation, initializer)

# ▼▼▼ 特征嵌入 (PyTorch 语法修改版) ▼▼▼
def input_transform(t):
    # 使用 torch.cat 和 torch.sin (对应 PyTorch)
    return torch.cat(
        (t, torch.sin(t), torch.sin(2 * t), torch.sin(3 * t), torch.sin(4 * t),
         torch.sin(5 * t), torch.sin(6 * t), torch.sin(7 * t)), dim=1,
    )

# ▼▼▼ 硬约束 (PyTorch 语法修改版) ▼▼▼
def output_transform(t, y):
    y1 = y[:, 0:1]
    y2 = y[:, 1:2]
    # 使用 torch.cat 和 torch.tanh
    return torch.cat([y1 * torch.tanh(t) + 100 / ub, y2 * torch.tanh(t) + 15 / ub], dim=1)

net.apply_feature_transform(input_transform)
net.apply_output_transform(output_transform)

# 4. 训练模型
model = dde.Model(data, net)
model.compile("adam", lr=0.001)

print("-> 开始 Adam 训练 (50000 iterations)...")
losshistory, train_state = model.train(iterations=50000)

print("-> 开始 L-BFGS 微调...")
model.compile("L-BFGS")
losshistory, train_state = model.train()

# 5. 结果可视化与自动保存
plt.figure(figsize=(10, 6))
plt.title(f"Conf: Arch={EXP_ARCH}, Sampling={EXP_SAMPLING} | Final Loss={train_state.loss_train[-1]:.2e}")
plt.xlabel("t")
plt.ylabel("Population")

# 画真实解
t_eval = np.linspace(0, 1, 100)
x_true, y_true = gen_truedata()
plt.plot(t_eval, x_true, color="black", label="True Prey (x)", linewidth=2, alpha=0.6)
plt.plot(t_eval, y_true, color="blue", label="True Predator (y)", linewidth=2, alpha=0.6)

# 画预测解
t_pred = t_eval.reshape(100, 1)
# PyTorch 输出通常需要 detach 转 numpy，DeepXDE 内部会自动处理，但以防万一
sol_pred = model.predict(t_pred)
x_pred = sol_pred[:, 0:1]
y_pred = sol_pred[:, 1:2]

plt.plot(t_eval, x_pred, color="red", linestyle="--", label="Pred Prey (x)")
plt.plot(t_eval, y_pred, color="orange", linestyle="--", label="Pred Predator (y)")

plt.legend()
plt.grid(True, alpha=0.3)

# 自动保存图片
filename = f"Result_{EXP_ARCH}_{EXP_SAMPLING}.png"
plt.savefig(filename, dpi=300)
print(f"-> 图片已保存为: {filename}")
print(f"-> 实验结束。")
plt.show()