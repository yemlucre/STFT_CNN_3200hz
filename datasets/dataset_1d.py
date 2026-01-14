"""dataset_1d.py —— 1D原始信号数据集（不做STFT）

功能：
- 递归扫描 data_dir 下各类别文件夹的所有 .mat
- 滑窗切片 → 直接返回原始1D信号
- 输出特征形状为 (1, segment_len) 的 numpy.ndarray（float32）

说明：
- 用于1D-CNN对照实验
- 不做STFT变换，直接使用原始振动信号
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from scipy.io import loadmat


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


# ===================== Dataset 类 =====================

class Raw1DDataset:
    """从 data_dir 扫描所有 .mat，切片并直接返回原始1D信号
    
    目录结构示例：
        data/
          Ball/...
          Inner Race/...
          Normal Baseline/...
          Outer Race/...
    
    参数说明：
        data_dir: 数据根目录
        class_to_idx: 类别名称到标签的映射
        preferred_channel_key: 优先读取的通道key
        segment_len: 每个样本长度（点数）
        step: 滑动步长
        normalize: 是否归一化到[0, 1]
        max_files_per_class: 每个类别最多读取的文件数（None表示全部）
    """

    def __init__(
        self,
        *,
        data_dir: str,
        class_to_idx: dict[str, int],
        preferred_channel_key: str = "X118_DE_time",
        segment_len: int = 2048,
        step: int = 1024,
        normalize: bool = True,
        max_files_per_class: int | None = None,
    ):
        self.data_dir = data_dir
        self.class_to_idx = dict(class_to_idx)
        self.preferred_channel_key = preferred_channel_key

        self.segment_len = segment_len
        self.step = step
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

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int, SampleMeta]:
        """取出一个样本

        返回:
        - feature: np.ndarray, shape=(1, segment_len) - 1D信号加通道维
        - label: int
        - meta: SampleMeta
        """
        meta = self._index[idx]

        mat = _loadmat_cached(meta.mat_path)
        signal = _safe_squeeze_1d(mat[meta.channel_key])

        segment = signal[meta.start : meta.start + self.segment_len].astype(np.float32)
        
        # 可选归一化
        if self.normalize:
            seg_min = float(segment.min())
            seg_max = float(segment.max())
            segment = (segment - seg_min) / (seg_max - seg_min + 1e-8)

        # 增加通道维： (segment_len,) -> (1, segment_len)
        feature = segment[None, :]
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
