import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import stft

# ================== 参数区 ==================
# 数据根目录：这里指向某一类故障（Ball）下的一个工况文件夹
data_root = "data/Ball/0007"     # 存放 .mat 文件的文件夹

# 选择 .mat 文件中要读取的通道名：驱动端加速度信号（DE: Drive End）
channel_key = "X118_DE_time"     # 驱动端通道

# 采样率（Hz）：该数据集常见为 12 kHz
fs = 12000                        # 采样率 (Hz)

# 分段参数：把长时间序列切成多个样本（滑窗切片）
segment_len = 2048                # 每个样本的长度（点数）
step = 1024                       # 滑动步长（这里等价于 50% 重叠）

# STFT 参数：短时傅里叶变换（时频图）
# - window: 窗函数类型
# - nperseg: 每段长度（点数），影响频率分辨率
# - noverlap: 重叠长度（点数）
# - nfft: FFT 点数（可进行零填充，影响频率采样密度）
window = "hann"
nperseg = 64
noverlap = 32
nfft = 128

# 频带裁剪：只保留关注的频率范围（去掉低频漂移/高频噪声等）
f_low = 200
f_high = 3200
# =====================================================================


def process_one_mat(mat_path):
    """读取单个 .mat 文件，滑窗切片得到多个样本，并计算每个样本的 STFT 幅值谱。

    返回:
        specs: list[np.ndarray]
            列表中每个元素对应一个样本的谱图（频率 bins × 时间 frames）
    """
    mat = loadmat(mat_path)

    # 安全检查：确保该 .mat 文件里存在我们指定的通道
    # 注意：不同文件的 key 可能不完全一致（例如 X118_DE_time / X119_DE_time 等）
    chosen_key = channel_key
    if chosen_key not in mat:
        # 兜底策略：自动寻找“驱动端 + time”这一类字段
        candidates = [k for k in mat.keys() if isinstance(k, str) and ("DE" in k) and ("time" in k)]
        if len(candidates) == 0:
            raise KeyError(
                f"{channel_key} not found in {mat_path}. "
                f"Available keys: {sorted([k for k in mat.keys() if isinstance(k, str)])}"
            )
        chosen_key = candidates[0]
        print(f"[WARN] {os.path.basename(mat_path)} 使用通道 {chosen_key}（原本期望 {channel_key}）")

    # squeeze() 把 (N,1) 或 (1,N) 等形状压成一维 (N,)
    signal = mat[chosen_key].squeeze()
    total_len = len(signal)

    specs = []

    # 滑动窗口切片：每次取 segment_len 点，步长为 step
    # 例如：0:2048, 1024:3072, 2048:4096 ...
    for start in range(0, total_len - segment_len + 1, step):
        segment = signal[start:start + segment_len]

        # 对当前 segment 做 STFT（输出为复数时频矩阵 Zxx）
        f, t, Zxx = stft(
            segment,
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            padded=False,
            boundary=None
        )

        # 频带裁剪：仅保留 [f_low, f_high] 的频率行
        idx = (f >= f_low) & (f <= f_high)

        # 取幅值谱（丢弃相位），作为 CNN/分类模型常用输入
        spec = np.abs(Zxx[idx, :])

        specs.append(spec)

    return specs


# ================== 主流程：批量处理文件夹内所有 .mat ==================
all_specs = []

# 记录每个样本来自哪个文件：
# - 后续划分训练/测试集时，建议按“文件级/工况级”划分，避免同一文件切出来的片段同时出现在训练和测试里（数据泄漏）
file_index = []

mat_files = sorted([
    os.path.join(data_root, f)
    for f in os.listdir(data_root)
    if f.endswith(".mat")
])

print(f"Found {len(mat_files)} .mat files")

for mat_path in mat_files:
    # 逐个文件处理：一个 .mat 会产生多个 STFT 样本
    specs = process_one_mat(mat_path)
    all_specs.extend(specs)
    file_index.extend([os.path.basename(mat_path)] * len(specs))
    print(f"{os.path.basename(mat_path)} -> {len(specs)} samples")

# 转成 numpy 数组，方便后续直接喂给模型
all_specs = np.array(all_specs)
file_index = np.array(file_index)

print("Total samples:", all_specs.shape)
