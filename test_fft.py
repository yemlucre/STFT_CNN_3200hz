import numpy as np
import matplotlib.pyplot as plt


def fft_analysis(signal, fs):
    """
    signal : 1D numpy array，振动信号
    fs     : 采样频率 (Hz)
    """
    N = len(signal)

    # 去直流分量（很重要，不然低频会炸）
    signal = signal - np.mean(signal)

    # FFT
    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals) / N   # 归一化幅值

    # 只取正频率
    freqs = np.fft.fftfreq(N, d=1/fs)
    idx = freqs >= 0

    return freqs[idx], fft_vals[idx]


# ================== 使用示例 ==================
from scipy.io import loadmat

# Load a single .mat file containing the vibration time-series
mat_path = "data/Outer Race/Orthogonal/0007/144.mat"
mat = loadmat(mat_path)

print(mat.keys())

# Extract the drive-end accelerometer channel

preferred_channel_key: str = "X118_DE_time"

def _choose_channel_key(mat_dict: dict, preferred: str) -> str:
    """选择 mat 中的振动通道 key

    - 优先使用 preferred（例如 "X118_DE_time"）
    - 不存在时，兜底选择包含 'DE' 和 'time' 的字段
    """
    if preferred in mat_dict:
        return preferred

    candidates = [
        k for k in mat_dict.keys()
        if isinstance(k, str) and ("DE" in k) and ("time" in k)
    ]
    if not candidates:
        available = sorted([k for k in mat_dict.keys() if isinstance(k, str)])
        raise KeyError(f"Channel key not found. preferred={preferred}, available={available}")

    return candidates[0]


# 选择可用的通道并取出信号
channel_key = _choose_channel_key(mat, preferred_channel_key)
signal = mat[channel_key]

# Drop singleton dimensions to get a 1D array
signal = signal.squeeze()
print(signal.shape)

fs = 12000  # ← 改成你真实的采样频率

freqs, amps = fft_analysis(signal, fs)

plt.figure()
plt.plot(freqs, amps)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("FFT Spectrum")
plt.xlim(0, fs/2)
plt.grid(True)
plt.show()
