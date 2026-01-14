"""config.py —— 所有参数集中管理

说明：
- 数据路径、类别映射
- 采样率、滑窗切片参数
- STFT 参数、频带裁剪
- 训练/测试划分

建议：你论文/报告里写清楚这些参数，可以直接引用本文件。
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """数据与特征参数"""

    # ===== 数据路径 =====
    data_dir: str = "data"

    # 类别映射：文件夹名 -> 标签 id
    # 你的目录中有："Normal Baseline"、"Ball"、"Inner Race"、"Outer Race"
    class_to_idx: dict[str, int] = None  # type: ignore[assignment]

    # 通道优先 key（不同 .mat 可能会自动兜底匹配）
    preferred_channel_key: str = "X118_DE_time"

    # ===== 采样率 =====
    fs: int = 12000  # Hz

    # ===== 滑窗切片参数 =====
    segment_len: int = 2048  # 每个样本长度（点数）
    step: int = 1024  # 滑动步长（50% 重叠）

    # ===== STFT 参数 =====
    window: str = "hann"
    nperseg: int = 64
    noverlap: int = 32
    nfft: int = 128

    # ===== 频带裁剪 =====
    f_low: float = 200.0
    f_high: float = 3200.0

    # ===== 特征归一化 =====
    normalize: bool = True

    # ===== 划分参数（按文件划分更安全） =====
    test_size: float = 0.2
    random_seed: int = 42

    # ===== 调试 =====
    max_files_per_class: int | None = None

    def __post_init__(self):
        if self.class_to_idx is None:
            object.__setattr__(
                self,
                "class_to_idx",
                {
                    "Normal Baseline": 0,
                    "Ball": 1,
                    "Inner Race": 2,
                    "Outer Race": 3,
                },
            )


@dataclass(frozen=True)
class TrainConfig:
    """训练参数（后续真正训练 CNN 时用）"""

    batch_size: int = 16
    epochs: int = 30
    lr: float = 1e-3
    device: str = "cuda:0"  # 或 "cpu"    
    # 正则化参数（缓解过拟合）
    weight_decay: float = 1e-4  # L2 正则化强度
    dropout_rate: float = 0.3  # Dropout 比例（0.0 = 不使用）
    
    # 早停参数
    early_stopping: bool = True  # 是否启用早停
    patience: int = 5  # 测试准确率连续多少epoch不提升则停止
    
    # 学习率调度
    use_scheduler: bool = True  # 是否使用学习率衰减
    scheduler_patience: int = 3  # ReduceLROnPlateau 的 patience
@dataclass(frozen=True)
class ModelConfig:
    """模型相关开关（选择器）"""

    # 可选："baseline_cnn" / "resnet18" / "mobilenet_v2"
    name: str = "mobilenet_v2"
    # 是否记录推理时间（平均 batch 前向时间）
    log_inference_time: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()


CFG = ProjectConfig()
