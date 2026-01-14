"""models.py —— 模型仓库：BaselineCNN / ResNet18 / MobileNetV2

接口约定：
- 输入：(N, 1, F, T) 的 torch.Tensor
- 输出：(N, num_classes) 的 logits（未做 softmax）

说明：
- ResNet18/MobileNetV2 使用 torchvision，首层改为单通道；分类头改为 `num_classes`。
- BaselineCNN 为极简 CNN，适配任意 (F, T) 尺寸。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models as tv_models
except Exception:  # torchvision 可选依赖（无则仅能用 BaselineCNN）
    tv_models = None


class BaselineCNN(nn.Module):
    """极简 CNN(baseline)

    结构：
    - Conv(1→32, k3) + BN + ReLU + MaxPool2d(2)
    - Conv(32→64, k3) + BN + ReLU + MaxPool2d(2)
    - AdaptiveAvgPool2d(1×1) → Dropout → Linear(64→num_classes)
    """

    def __init__(self, num_classes: int, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


def build_resnet18(num_classes: int, in_channels: int = 1, dropout: float = 0.3) -> nn.Module:
    """构建 ResNet18（输入通道=1，添加Dropout）"""
    if tv_models is None:
        raise RuntimeError("torchvision 未安装，无法使用 ResNet18")

    model = tv_models.resnet18(weights=None)
    # 改首层为单通道
    model.conv1 = nn.Conv2d(in_channels, model.conv1.out_channels, kernel_size=model.conv1.kernel_size,
                            stride=model.conv1.stride, padding=model.conv1.padding, bias=False)
    # 改最后分类头：添加 Dropout
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def build_mobilenet_v2(num_classes: int, in_channels: int = 1, dropout: float = 0.3) -> nn.Module:
    """构建 MobileNetV2（输入通道=1，添加Dropout）"""
    if tv_models is None:
        raise RuntimeError("torchvision 未安装，无法使用 MobileNetV2")

    model = tv_models.mobilenet_v2(weights=None)
    # 修改首层 conv 的输入通道
    first_conv: nn.Conv2d = model.features[0][0]
    model.features[0][0] = nn.Conv2d(in_channels, first_conv.out_channels,
                                     kernel_size=first_conv.kernel_size,
                                     stride=first_conv.stride,
                                     padding=first_conv.padding,
                                     bias=False)
    # 修改分类头：添加 Dropout
    last_features = model.classifier[1].in_features
    if dropout > 0:
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_features, num_classes)
        )
    else:
        model.classifier[1] = nn.Linear(last_features, num_classes)
    return model


def get_model(name: str, num_classes: int, in_channels: int = 1, dropout: float = 0.3) -> nn.Module:
    """模型选择器

    支持："baseline" / "resnet18" / "mobilenet_v2"
    """
    name = (name or "baseline").lower()
    if name in ("baseline", "baseline_cnn", "cnn"):
        return BaselineCNN(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    if name in ("resnet", "resnet18"):
        return build_resnet18(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    if name in ("mobilenet", "mobilenet_v2"):
        return build_mobilenet_v2(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
    raise ValueError(f"Unknown model name: {name}")


def count_parameters(model: nn.Module) -> int:
    """统计模型参数量（包含所有可学习参数）"""
    return sum(p.numel() for p in model.parameters())
