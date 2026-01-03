"""
DeepONet Experiment (Part 2) - Real Dataset Version
环境：DeepXDE + PyTorch
需文件：antiderivative_aligned_train.npz, antiderivative_aligned_test.npz
"""
import os
import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np
import torch

# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# 【关键设置】强制使用 PyTorch 后端
os.environ["DDE_BACKEND"] = "pytorch"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# ==============================================================================
# ▼▼▼ 实验控制面板 (每次运行前修改这里) ▼▼▼
# ==============================================================================

# 1. 网络架构选择 (三选一)
EXP_ARCH = "BASELINE"  # 基准: Branch/Trunk 均为 [40, 40]
#EXP_ARCH = "WIDE"      # 宽网络: [128, 128]
#EXP_ARCH = "DEEP"      # 深网络: [40]*6 层

# 2. 采样策略选择 (控制使用多少训练数据)
#EXP_DATA = "BASELINE"  # 基准: 使用 1000 个样本
#EXP_DATA = "SPARSE"      # 稀疏: 只截取前 150 个样本 (模拟数据匮乏)
EXP_DATA = "DENSE"       # 稠密: 使用全部数据 (通常是 10000 或更多)

# ==============================================================================
# ▲▲▲ 配置结束，以下代码无需修改 ▲▲▲
# ==============================================================================

print(f"--- DeepONet 实验开始 (Real Dataset) ---")
print(f"当前配置: 架构=[{EXP_ARCH}] | 数据量=[{EXP_DATA}]")


# 1. 数据加载与切片函数
def load_data(filename, n_samples=None):
    """
    加载 .npz 文件，并根据 n_samples 进行切片
    """
    print(f"-> 正在加载 {filename}...")
    d = np.load(filename, allow_pickle=True)

    # 原始数据
    X_branch_full = d["X"][0].astype(np.float32)  # Shape: (N, m)
    X_trunk_full = d["X"][1].astype(np.float32)  # Shape: (P, dim)
    y_full = d["y"].astype(np.float32)  # Shape: (N, P)

    # 获取总样本数
    total_samples = X_branch_full.shape[0]

    # 确定我们要用多少
    if n_samples is None or n_samples > total_samples:
        n_use = total_samples
    else:
        n_use = n_samples

    print(f"   原数据集大小: {total_samples}, 实际使用: {n_use}")

    # 切片操作 (只切 Branch 和 y，Trunk 保持不变)
    X_branch_sliced = X_branch_full[:n_use]
    y_sliced = y_full[:n_use]

    return (X_branch_sliced, X_trunk_full), y_sliced


# 2. 准备数据
# 根据控制面板设定 n_train
if EXP_DATA == "BASELINE":
    n_train_target = 1000
elif EXP_DATA == "SPARSE":
    n_train_target = 150
    print("-> 警告: 使用稀疏数据 (N=150)")
elif EXP_DATA == "DENSE":
    n_train_target = None  # None 表示读取所有
    print("-> 提示: 使用全部可用数据")

# 加载训练集
X_train, y_train = load_data("antiderivative_aligned_train.npz", n_train_target)
# 加载测试集 (全部加载用于评估)
X_test, y_test = load_data("antiderivative_aligned_test.npz", None)

data = dde.data.TripleCartesianProd(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)

# 3. 网络架构定义
m = X_train[0].shape[1]  # 自动获取传感器数量 (例如 100)
dim_x = X_train[1].shape[1]  # 自动获取维度 (例如 1)

if EXP_ARCH == "BASELINE":
    # 原始教程配置
    net = dde.nn.DeepONetCartesianProd(
        [m, 40, 40],
        [dim_x, 40, 40],
        "relu", "Glorot normal"
    )
elif EXP_ARCH == "WIDE":
    # 宽网络
    net = dde.nn.DeepONetCartesianProd(
        [m, 128, 128],
        [dim_x, 128, 128],
        "relu", "Glorot normal"
    )
elif EXP_ARCH == "DEEP":
    # 深网络 (6层)
    deep_vec = [40] * 6
    net = dde.nn.DeepONetCartesianProd(
        [m] + deep_vec,
        [dim_x] + deep_vec,
        "relu", "Glorot normal"
    )

# 4. 模型训练
model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["mean l2 relative error"])

print("-> 开始训练 (10000 iterations)...")
# 为了节省时间，如果你是用全部数据跑(DENSE)，可能稍微慢一点
losshistory, train_state = model.train(iterations=10000)

# 5. 结果可视化与保存
plt.figure(figsize=(10, 6))
# 标题包含配置信息和最终 Loss
plt.title(f"DeepONet: Arch={EXP_ARCH}, Data={EXP_DATA} | Final Loss={train_state.loss_train[-1]:.2e}")
plt.xlabel("x")
plt.ylabel("v(x)")

# 随机选一个测试样本画图
idx = np.random.randint(0, y_test.shape[0])
x_grid = X_test[1].flatten()
y_true_sample = y_test[idx]

# 预测
u_sample = X_test[0][idx].reshape(1, -1)
# 这里的 predict 稍微复杂一点，DeepONetCartesianProd 需要特定格式
y_pred_sample = model.predict((u_sample, X_test[1])).flatten()

plt.plot(x_grid, y_true_sample, 'k-', label="True Solution", linewidth=2)
plt.plot(x_grid, y_pred_sample, 'r--', label="Prediction", linewidth=2)
plt.legend()
plt.grid(True, alpha=0.3)

filename = f"DeepONet_{EXP_ARCH}_{EXP_DATA}.png"
plt.savefig(filename, dpi=300)
print(f"-> 图片已保存为: {filename}")
print("-> 实验结束。")
plt.show()