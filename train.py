"""train.py —— 训练、测试、主入口（模型选择器 + 完整训练循环）

功能：
- 构建 STFTFolderDataset（numpy 特征）
- 按文件级划分 train/test（避免数据泄漏）
- 从 models.py 选择模型（BaselineCNN / ResNet18 / MobileNetV2）
- 完整训练循环：优化器、损失函数、多epoch训练
- 自动记录每个epoch的train/test loss和accuracy
- 保存最佳模型到 checkpoints/ 目录
- 支持 Ctrl+C 中断，已保存的最佳模型不丢失
"""

from __future__ import annotations

import os
import random
from typing import Iterable
from pathlib import Path

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import CFG
from datasets import STFTFolderDataset
from models import get_model, count_parameters


def split_by_file(
    groups: dict[str, list[int]],
    test_size: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """按文件划分索引，返回 train_indices, test_indices"""

    files = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(files)

    if len(files) <= 1:
        return list(range(sum(len(v) for v in groups.values()))), []

    n_test = max(1, int(round(len(files) * test_size)))
    test_files = set(files[:n_test])

    train_idx: list[int] = []
    test_idx: list[int] = []
    for f, idxs in groups.items():
        if f in test_files:
            test_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    return train_idx, test_idx


def batch_iter(indices: list[int], batch_size: int) -> Iterable[list[int]]:
    """简单的 batch 迭代器"""
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def train_one_epoch(
    ds: STFTFolderDataset,
    indices: list[int],
    model: nn.Module,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    device: str,
) -> tuple[float, float]:
    """训练一个epoch，返回 (平均loss, 准确率)"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    # 随机打乱训练数据
    shuffled_indices = indices.copy()
    random.shuffle(shuffled_indices)

    for batch in batch_iter(shuffled_indices, batch_size):
        xs = []
        ys = []
        for i in batch:
            x, y, _meta = ds[i]
            xs.append(x)
            ys.append(y)

        x_arr = np.stack(xs, axis=0).astype(np.float32)
        y_arr = np.array(ys, dtype=np.int64)

        x_t = torch.from_numpy(x_arr).to(device)
        y_t = torch.from_numpy(y_arr).to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(x_t)
        loss = criterion(logits, y_t)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y_t).sum().item())
        total += len(y_arr)
        n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def eval_accuracy(
    ds: STFTFolderDataset,
    indices: list[int],
    model: nn.Module,
    batch_size: int,
    device: str,
) -> float:
    """计算准确率（torch 模型）"""

    if not indices:
        return float("nan")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in batch_iter(indices, batch_size):
            xs = []
            ys = []
            for i in batch:
                x, y, _meta = ds[i]
                xs.append(x)
                ys.append(y)

            x_arr = np.stack(xs, axis=0).astype(np.float32)  # (N, 1, F, T)
            y_arr = np.array(ys, dtype=np.int64)

            x_t = torch.from_numpy(x_arr).to(device)
            logits = model(x_t)  # (N, num_classes)
            pred = torch.argmax(logits, dim=1).cpu().numpy()

            correct += int((pred == y_arr).sum())
            total += int(y_arr.size)

    return correct / max(1, total)


def eval_loss_and_accuracy(
    ds: STFTFolderDataset,
    indices: list[int],
    model: nn.Module,
    criterion: nn.Module,
    batch_size: int,
    device: str,
) -> tuple[float, float]:
    """计算验证集的loss和准确率"""
    if not indices:
        return float("nan"), float("nan")

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

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
            y_t = torch.from_numpy(y_arr).to(device)

            logits = model(x_t)
            loss = criterion(logits, y_t)

            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == y_t).sum().item())
            total += len(y_arr)
            n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    accuracy = correct / max(1, total)
    return avg_loss, accuracy


def measure_inference_time(
    ds: STFTFolderDataset,
    indices: list[int],
    model: nn.Module,
    batch_size: int,
    device: str,
) -> float:
    """测量平均前向推理时间（秒/批次）"""
    if not indices:
        return float("nan")

    model.eval()
    times: list[float] = []
    with torch.no_grad():
        for batch in batch_iter(indices, batch_size):
            xs = []
            for i in batch:
                x, _y, _meta = ds[i]
                xs.append(x)
            x_arr = np.stack(xs, axis=0).astype(np.float32)
            x_t = torch.from_numpy(x_arr).to(device)

            start = time.perf_counter()
            _ = model(x_t)
            torch.cuda.synchronize() if device.startswith("cuda") else None
            end = time.perf_counter()
            times.append(end - start)

    if not times:
        return float("nan")
    return float(sum(times) / len(times))


def main() -> None:
    print("=" * 50)
    print("轴承故障诊断 - 训练/测试入口")
    print("=" * 50)

    # 1) 构建数据集
    ds = STFTFolderDataset(
        data_dir=CFG.data.data_dir,
        class_to_idx=CFG.data.class_to_idx,
        preferred_channel_key=CFG.data.preferred_channel_key,
        fs=CFG.data.fs,
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

    # 查看单个样本形状
    x0, y0, meta0 = ds[0]
    print(f"样本形状: {x0.shape}, 数据类型: {x0.dtype}")
    print(f"样本标签: {y0}, 类别: {meta0.class_name}")

    # 2) 按文件划分 train/test
    groups = ds.file_groups()
    train_idx, test_idx = split_by_file(groups, CFG.data.test_size, CFG.data.random_seed)

    print(f"训练样本数: {len(train_idx)}")
    print(f"测试样本数: {len(test_idx)}")

    # 3) 模型选择 + 指标
    num_classes = len(CFG.data.class_to_idx)
    device = CFG.train.device
    model = get_model(CFG.model.name, num_classes=num_classes, in_channels=1, dropout=CFG.train.dropout_rate)
    model = model.to(device)

    # 参数量
    params = count_parameters(model)
    print(f"\n模型: {CFG.model.name}")
    print(f"参数量: {params:,}")
    print(f"Dropout: {CFG.train.dropout_rate}")
    print(f"权重衰减: {CFG.train.weight_decay}")

    # 4) 训练设置
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CFG.train.lr, weight_decay=CFG.train.weight_decay)
    
    # 学习率调度器（当测试loss不下降时降低学习率）
    scheduler = None
    if CFG.train.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=CFG.train.scheduler_patience, verbose=True
        )

    # 创建checkpoint目录
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    best_model_path = checkpoint_dir / f"{CFG.model.name}_best.pth"

    # 5) 训练循环
    print("\n" + "=" * 50)
    print(f"开始训练 ({CFG.train.epochs} epochs)")
    print("=" * 50)

    best_test_acc = 0.0
    best_test_loss = float('inf')
    epochs_no_improve = 0
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    try:
        for epoch in range(1, CFG.train.epochs + 1):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = train_one_epoch(
                ds, train_idx, model, optimizer, criterion,
                CFG.train.batch_size, device
            )

            # 验证
            test_loss, test_acc = eval_loss_and_accuracy(
                ds, test_idx, model, criterion,
                CFG.train.batch_size, device
            ) if test_idx else (float("nan"), float("nan"))

            epoch_time = time.time() - epoch_start

            # 记录历史
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc)

            # 打印进度
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch}/{CFG.train.epochs}] - {epoch_time:.2f}s - LR: {current_lr:.6f}")
            print(f"  训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            if test_idx:
                print(f"  测试 - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

            # 学习率调度器（基于测试loss）
            if scheduler is not None and test_idx:
                scheduler.step(test_loss)

            # 保存最佳模型
            if test_idx and test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_loss = test_loss
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "history": history,
                }, best_model_path)
                print(f"  ✓ 保存最佳模型 (测试准确率: {test_acc:.4f})")
            else:
                epochs_no_improve += 1

            # 早停检查
            if CFG.train.early_stopping and test_idx and epochs_no_improve >= CFG.train.patience:
                print(f"\n早停触发：测试准确率连续 {CFG.train.patience} 个epoch未提升")
                print(f"最佳测试准确率: {best_test_acc:.4f} (Epoch {epoch - epochs_no_improve})")
                break

    except KeyboardInterrupt:
        print("\n\n训练被中断 (Ctrl+C)")
        print(f"最佳模型已保存至: {best_model_path}")

    # 6) 训练完成
    print("\n" + "=" * 50)
    print("训练完成")
    print("=" * 50)
    print(f"最佳测试准确率: {best_test_acc:.4f}")
    print(f"最佳模型路径: {best_model_path}")

    # 7) 推理时间测试（可选）
    if CFG.model.log_inference_time:
        print("\n测量推理时间...")
        infer_time_train = measure_inference_time(ds, train_idx, model, CFG.train.batch_size, device)
        infer_time_test = measure_inference_time(ds, test_idx, model, CFG.train.batch_size, device) if test_idx else float("nan")
        print(f"平均推理时间/训练集 (秒/批次): {infer_time_train:.6f}")
        if test_idx:
            print(f"平均推理时间/测试集 (秒/批次): {infer_time_test:.6f}")


if __name__ == "__main__":
    main()
