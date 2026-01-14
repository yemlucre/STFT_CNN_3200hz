"""datasets.py —— 只放 Dataset（STFT 在这里完成）

功能：
- 递归扫描 data_dir 下各类别文件夹的所有 .mat
- 滑窗切片 → 在 __getitem__ 计算 STFT 幅值谱 → 频带裁剪
- 输出特征形状为 (1, F, T) 的 numpy.ndarray（float32）

说明：
- 当前版本不依赖 torch（仅 numpy + scipy）
- 为避免数据泄漏，建议按"文件级"划分 train/test
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.io import loadmat
from scipy.signal import stft


# ===================== 数据结构 =====================

@dataclass(frozen=True)
class SampleMeta:
    """样本元信息（定位样本来自哪个文件/哪个切片）"""

    mat_path: str
    class_name: str
    channel_key: str
    start: int


# ===================== 辅助函数 =====================

def _is_mat_file(name: str) -> bool:
    """判断文件是否为 .mat"""
    return name.lower().endswith(".mat")


def _iter_mat_files(root_dir: str) -> Iterable[str]:
    """递归遍历 root_dir，返回所有 .mat 文件路径"""
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if _is_mat_file(fn):
                yield os.path.join(dirpath, fn)


def _safe_squeeze_1d(x: np.ndarray) -> np.ndarray:
    """把 mat 读出来的数组尽量转成 1D"""
    x = np.asarray(x).squeeze()
    if x.ndim != 1:
        x = x.reshape(-1)
    return x


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


@lru_cache(maxsize=64)
def _loadmat_cached(mat_path: str) -> dict:
    """缓存 loadmat 结果，减少频繁磁盘 IO"""
    return loadmat(mat_path)


# ===================== STFT 特征计算 =====================

def compute_stft_feature(
    segment: np.ndarray,
    *,
    fs: int,
    window: str,
    nperseg: int,
    noverlap: int,
    nfft: int,
    f_low: float,
    f_high: float,
    normalize: bool,
) -> np.ndarray:
    """把 1D segment 转成 STFT 幅值谱，并裁剪频带

    返回：
        spec: np.ndarray, shape=(F, T), dtype=float32
    """
    f, _t, Zxx = stft(
        segment,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        padded=False,
        boundary=None,
    )

    # 频带裁剪
    idx = (f >= f_low) & (f <= f_high)
    spec = np.abs(Zxx[idx, :]).astype(np.float32)

    # 可选归一化
    if normalize:
        spec_min = float(spec.min())
        spec_max = float(spec.max())
        spec = (spec - spec_min) / (spec_max - spec_min + 1e-8)

    return spec


# ===================== Dataset 类 =====================

class STFTFolderDataset:
    """从 data_dir 扫描所有 .mat，切片并在 __getitem__ 计算 STFT 特征

    目录结构示例：
        data/
          Ball/...
          Inner Race/...
          Normal Baseline/...
          Outer Race/...

    参数说明见 config.py 的 DataConfig
    """

    def __init__(
        self,
        *,
        data_dir: str,
        class_to_idx: dict[str, int],
        preferred_channel_key: str = "X118_DE_time",
        fs: int = 12000,
        segment_len: int = 2048,
        step: int = 1024,
        window: str = "hann",
        nperseg: int = 64,
        noverlap: int = 32,
        nfft: int = 128,
        f_low: float = 200.0,
        f_high: float = 3200.0,
        normalize: bool = True,
        max_files_per_class: int | None = None,
    ):
        self.data_dir = data_dir
        self.class_to_idx = dict(class_to_idx)
        self.preferred_channel_key = preferred_channel_key

        self.fs = fs
        self.segment_len = segment_len
        self.step = step

        self.window = window
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft

        self.f_low = f_low
        self.f_high = f_high
        self.normalize = normalize

        self._validate_params()

        # 构建索引
        self._index: list[SampleMeta] = []

        for class_name in self.class_to_idx.keys():
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            mat_paths = sorted(list(_iter_mat_files(class_dir)))
            if max_files_per_class is not None:
                mat_paths = mat_paths[:max_files_per_class]

            for mat_path in mat_paths:
                mat = _loadmat_cached(mat_path)
                channel_key = _choose_channel_key(mat, self.preferred_channel_key)
                signal = _safe_squeeze_1d(mat[channel_key])
                total_len = int(signal.size)

                for start in range(0, total_len - self.segment_len + 1, self.step):
                    self._index.append(
                        SampleMeta(
                            mat_path=mat_path,
                            class_name=class_name,
                            channel_key=channel_key,
                            start=start,
                        )
                    )

    def _validate_params(self) -> None:
        """参数校验"""
        if not isinstance(self.data_dir, str) or not self.data_dir:
            raise ValueError("data_dir must be a non-empty string")
        if self.segment_len <= 0:
            raise ValueError("segment_len must be > 0")
        if self.step <= 0:
            raise ValueError("step must be > 0")
        if self.fs <= 0:
            raise ValueError("fs must be > 0")
        if self.nperseg <= 0:
            raise ValueError("nperseg must be > 0")
        if self.noverlap < 0:
            raise ValueError("noverlap must be >= 0")
        if self.noverlap >= self.nperseg:
            raise ValueError("noverlap must be < nperseg")
        if self.nfft <= 0:
            raise ValueError("nfft must be > 0")
        if self.f_low < 0 or self.f_high <= 0 or self.f_low >= self.f_high:
            raise ValueError("frequency band must satisfy 0 <= f_low < f_high")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, SampleMeta]:
        """取出一个样本

        返回:
        - feature: np.ndarray, shape=(1, F, T)
        - label: int
        - meta: SampleMeta
        """
        meta = self._index[idx]

        mat = _loadmat_cached(meta.mat_path)
        signal = _safe_squeeze_1d(mat[meta.channel_key])

        segment = signal[meta.start : meta.start + self.segment_len]
        spec = compute_stft_feature(
            segment,
            fs=self.fs,
            window=self.window,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            f_low=self.f_low,
            f_high=self.f_high,
            normalize=self.normalize,
        )

        # 增加通道维： (F, T) -> (1, F, T)
        feature = spec[None, :, :]
        label = int(self.class_to_idx[meta.class_name])
        return feature, label, meta

    def file_groups(self) -> dict[str, list[int]]:
        """按文件分组：{mat_path: [sample_indices...]}，便于按文件划分 train/test"""
        groups: dict[str, list[int]] = {}
        for i, meta in enumerate(self._index):
            groups.setdefault(meta.mat_path, []).append(i)
        return groups

    def clear_cache(self) -> None:
        """清空 loadmat 缓存（内存紧张时可用）"""
        _loadmat_cached.cache_clear()
