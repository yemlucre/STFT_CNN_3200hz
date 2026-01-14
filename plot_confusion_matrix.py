"""plot_confusion_matrix.py —— 绘制混淆矩阵（Confusion Matrix）

功能：
- 加载最佳模型（从 checkpoints/ 目录）
- 在测试集上生成预测
- 计算混淆矩阵
- 绘制并保存为图像

使用方法：
    python plot_confusion_matrix.py [--model_name mobilenet_v2]

"""

from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from config import CFG
from datasets import STFTFolderDataset
from models import get_model
from train import split_by_file, batch_iter


def get_all_predictions(
    ds: STFTFolderDataset,
    indices: list[int],
    model: nn.Module,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """获取所有预测结果和真实标签
    
    返回：
        (predictions, true_labels) 都是 numpy 数组
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in batch_iter(indices, batch_size):
            xs = []
            ys = []
            for i in batch:
                x, y, _meta = ds[i]
                xs.append(x)
                ys.append(y)

            x_arr = np.stack(xs, axis=0).astype(np.float32)
            y_arr = np.array(ys, dtype=np.int64)

            x_t = torch.from_numpy(x_arr).to(device)
            logits = model(x_t)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(pred)
            all_labels.extend(y_arr)

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: str = None,
) -> None:
    """绘制混淆矩阵（使用 matplotlib）
    
    参数：
        cm: 混淆矩阵（从 sklearn.metrics.confusion_matrix 得到）
        class_names: 类别名称列表
        title: 标题
        save_path: 保存路径（如果为 None，只显示不保存）
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制热力图
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # 添加色条
    plt.colorbar(im, ax=ax, label='Count')
    
    # 设置坐标轴标签
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel='Predicted Label',
           ylabel='True Label')
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在单元格中添加数字
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center",
                          color="white" if cm[i, j] > cm.max() / 2 else "black",
                          fontsize=12, fontweight='bold')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存至: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="绘制测试集的混淆矩阵")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mobilenet_v2",
        choices=["baseline_cnn", "resnet18", "mobilenet_v2"],
        help="模型名称（默认: mobilenet_v2）"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="保存结果的目录（默认: results）"
    )
    args = parser.parse_args()

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # 设备（自动检测 CUDA 可用性）
    if "cuda" in CFG.train.device and not torch.cuda.is_available():
        device = "cpu"
        print(f"警告: CUDA 不可用，改用 CPU")
    else:
        device = CFG.train.device

    # 1) 加载数据集
    print("加载数据集...")
    ds = STFTFolderDataset(
        data_dir=CFG.data.data_dir,
        class_to_idx=CFG.data.class_to_idx,
        preferred_channel_key=CFG.data.preferred_channel_key,
        segment_len=CFG.data.segment_len,
        step=CFG.data.step,
        window=CFG.data.window,
        nperseg=CFG.data.nperseg,
        noverlap=CFG.data.noverlap,
        nfft=CFG.data.nfft,
        f_low=CFG.data.f_low,
        f_high=CFG.data.f_high,
        normalize=CFG.data.normalize,
        max_files_per_class=CFG.data.max_files_per_class,
    )

    if len(ds) == 0:
        raise RuntimeError(f"数据集为空，请检查 data_dir={CFG.data.data_dir}")

    print(f"总样本数: {len(ds)}")

    # 2) 划分 train/test（必须使用相同的划分）
    groups = ds.file_groups()
    train_idx, test_idx = split_by_file(groups, CFG.data.test_size, CFG.data.random_seed)

    if not test_idx:
        raise RuntimeError("测试集为空，无法绘制混淆矩阵")

    print(f"训练样本数: {len(train_idx)}")
    print(f"测试样本数: {len(test_idx)}")

    # 3) 加载最佳模型
    checkpoint_dir = Path("checkpoints")
    best_model_path = checkpoint_dir / f"{args.model_name}_best.pth"

    if not best_model_path.exists():
        raise FileNotFoundError(
            f"未找到最佳模型: {best_model_path}\n"
            f"请先运行 train.py 进行训练，或检查 --model_name 参数"
        )

    print(f"\n加载模型: {best_model_path}")
    num_classes = len(CFG.data.class_to_idx)
    model = get_model(args.model_name, num_classes=num_classes, in_channels=1, dropout=CFG.train.dropout_rate)
    model = model.to(device)

    # 加载权重
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"✓ 模型已加载 (epoch {checkpoint['epoch']}, 测试准确率: {checkpoint['test_acc']:.4f})")

    # 4) 生成预测（在测试集上）
    print("\n在测试集上生成预测...")
    predictions, true_labels = get_all_predictions(
        ds, test_idx, model, CFG.train.batch_size, device
    )

    # 5) 计算混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    
    # 6) 获取类别名称
    idx_to_class = {v: k for k, v in CFG.data.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    # 7) 绘制混淆矩阵
    print("\n" + "=" * 50)
    print("混淆矩阵")
    print("=" * 50)
    print(cm)
    
    confusion_matrix_save_path = save_dir / f"{args.model_name}_confusion_matrix.png"
    plot_confusion_matrix(
        cm,
        class_names,
        title=f"Confusion Matrix - {args.model_name.upper()}",
        save_path=str(confusion_matrix_save_path)
    )

    # 8) 计算并打印分类报告
    print("\n" + "=" * 50)
    print("分类报告（测试集）")
    print("=" * 50)
    print(classification_report(true_labels, predictions, target_names=class_names))

    # 9) 计算总体准确率
    test_accuracy = np.mean(predictions == true_labels)
    print(f"\n总体测试准确率: {test_accuracy:.4f}")

    # 10) 保存详细报告
    report_save_path = save_dir / f"{args.model_name}_classification_report.txt"
    with open(report_save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write(f"模型: {args.model_name}\n")
        f.write(f"数据集: {CFG.data.data_dir}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("混淆矩阵\n")
        f.write("-" * 50 + "\n")
        f.write(str(cm) + "\n\n")
        
        f.write("分类报告\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(true_labels, predictions, target_names=class_names))
        f.write("\n\n")
        
        f.write("总体指标\n")
        f.write("-" * 50 + "\n")
        f.write(f"总体测试准确率: {test_accuracy:.4f}\n")
        f.write(f"测试样本总数: {len(test_idx)}\n")

    print(f"✓ 详细报告已保存至: {report_save_path}")

    print("\n" + "=" * 50)
    print("完成！")
    print("=" * 50)
    print(f"结果保存目录: {save_dir}/")


if __name__ == "__main__":
    main()
