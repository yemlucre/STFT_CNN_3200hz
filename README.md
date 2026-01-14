# 轴承故障诊断CNN项目

## 项目概述

本项目是一个基于深度学习的**轴承故障诊断系统**，采用短时傅里叶变换（STFT）提取时频特征，结合多种CNN架构进行分类诊断。项目包括数据加载、特征提取、模型训练、评估等完整的机器学习流程。

### 核心特性
- 🔧 **多模型支持**：BaselineCNN、ResNet18、MobileNetV2三种架构可选
- 📊 **实时特征提取**：STFT按需计算，支持频带裁剪（200-3200Hz）
- 🔐 **防数据泄漏**：按文件级划分训练/测试集，确保数据独立性
- 💾 **自动保存**：训练过程中自动保存最佳模型和性能指标
- 🎯 **完整评估**：包含混淆矩阵、分类报告等多维度评估工具

---

## 项目结构与文件说明

### 核心程序文件

#### 1. **config.py** - 配置管理中心
**功能**：集中管理项目所有超参数和配置参数  
**主要配置项**：
- **数据路径**：数据集根目录位置
- **类别映射**：故障类型文件夹 → 标签ID的对应关系
  - Normal Baseline（正常）
  - Ball（球缺陷）
  - Inner Race（内圈缺陷）
  - Outer Race（外圈缺陷）
- **采样率**：12000 Hz
- **滑窗切片参数**：
  - 段长度：2048点（约170ms）
  - 步长：1024点（50%重叠）
- **STFT参数**：
  - 窗函数：汉宁窗
  - 每段点数：64
  - 重叠点数：32
  - FFT点数：128
- **频带范围**：200-3200Hz（带通滤波范围）
- **特征归一化**：自动Z-score归一化

**建议**：论文/报告中应明确说明这些参数，便于复现和对比实验。

#### 2. **datasets.py** - 数据集加载与特征提取
**功能**：
- 递归扫描data目录下各类别文件夹中的所有 `.mat` 文件
- 实现样本元信息管理（SampleMeta类）
- 在数据加载时动态计算STFT幅值谱
- 自动频带裁剪和特征归一化
- 支持灵活的train/test划分策略

**技术细节**：
- 使用scipy的loadmat加载MATLAB数据文件
- 采用scipy.signal.stft计算时频特征
- 输出格式：(1, F, T)的numpy数组（float32）
  - 1：单通道（驱动端加速度计DE_time）
  - F：频率轴维度
  - T：时间轴维度
- 关键方法：
  - `SampleMeta`：记录样本来源文件和切片位置
  - `_choose_channel_key()`：自动选择正确的数据通道
  - `STFTFolderDataset`：PyTorch Dataset接口实现

**优势**：不依赖torchvision，易于定制和扩展。

#### 3. **models.py** - 神经网络模型库
**功能**：定义并提供三种CNN架构供选择  

**包含模型**：

1. **BaselineCNN**（极简基准模型）
   - 结构简洁，参数少（适合快速实验）
   - 两层卷积：1ch→32ch→64ch
   - 自适应平均池化 + Dropout + 全连接层
   - 适配任意输入尺寸(F, T)

2. **ResNet18**（经典深度网络）
   - 基于torchvision官方实现改进
   - 首层改为单通道输入（原始为3通道）
   - 残差连接，深度18层
   - 性能与参数量的良好平衡
   - 适合中等规模数据集

3. **MobileNetV2**（轻量级网络）
   - 基于torchvision官方实现改进
   - 倒残差块（Inverted Residual）结构
   - 参数量少，推理速度快
   - 适合嵌入式部署和实时诊断

**接口约定**：
- 输入：(N, 1, F, T)的torch.Tensor
  - N：批量大小
  - 1：单通道
  - F×T：STFT时频图
- 输出：(N, num_classes)的logits（未经softmax）

**辅助函数**：
- `get_model(name, num_classes)`：模型工厂函数
- `count_parameters(model)`：计算总参数量

#### 4. **train.py** - 主训练程序与入口
**功能**：完整的训练、验证、测试流程  

**核心功能**：
- **数据加载**：构建STFTFolderDataset，准备train/test数据加载器
- **按文件级划分**：避免同一振动文件的多个切片同时出现在train和test中（防止数据泄漏）
- **模型选择**：从models.py动态选择BaselineCNN/ResNet18/MobileNetV2
- **完整训练循环**：
  - 前向传播 + 损失计算
  - 反向传播 + 优化器更新
  - 多epoch训练，记录每epoch的train/test loss和accuracy
- **最佳模型保存**：自动保存准确率最高的模型到checkpoints/目录
- **优雅中断**：支持Ctrl+C中断，已保存的最佳模型不丢失
- **性能指标记录**：
  - 模型参数总数
  - 平均推理时间
  - 训练和测试精度

#### 5. **plot_confusion_matrix.py** - 模型评估与可视化
**功能**：加载训练好的模型，在测试集上生成混淆矩阵并绘制可视化  

**工作流程**：
- 加载指定的最佳模型（从checkpoints/目录）
- 在整个测试集上进行推理
- 计算混淆矩阵和分类性能指标
- 绘制混淆矩阵热力图
- 保存可视化结果和分类报告

**使用方式**：
```bash
python plot_confusion_matrix.py --model_name mobilenet_v2
```

**输出**：
- 混淆矩阵图像
- 详细分类报告（precision, recall, F1-score等）
- 保存到results/目录

#### 6. **test_load_mat.py** - 单个.mat文件加载测试
**功能**：测试和验证.mat文件的加载流程  

**用途**：
- 检查数据文件的可读性
- 验证通道选择逻辑
- 可视化原始振动信号
- 快速原型实验

**典型操作**：
```python
# 加载.mat文件
mat = loadmat("data/Outer Race/Orthogonal/0007/144.mat")
# 提取驱动端时间波形
signal = mat['X144_DE_time'].squeeze()
# 绘制前2000个采样点
```

#### 7. **test_fft.py** - FFT频谱分析工具
**功能**：实现FFT分析函数，用于频域特征检查  

**核心函数**：
- `fft_analysis(signal, fs)`：计算FFT幅值谱
  - 去直流分量（防止低频分量过大）
  - 计算单边FFT幅值
  - 返回频率轴和幅值

**应用场景**：
- 验证特征提取的正确性
- 频域故障特征可视化
- 频谱峭度等诊断特征计算

**示例**：
```python
freqs, mag = fft_analysis(signal, fs=12000)
plt.plot(freqs, mag)
```

#### 8. **test1.py** - 通用测试脚本
**功能**：快速测试和调试各个模块  

**用途**：
- 测试数据加载流程
- 验证模型前向传播
- 调试特征提取参数
- 性能基准测试

---

### 数据集结构

```
data/
├── Normal Baseline/        # 正常工作状态（参考基线）
│   ├── normal_0.mat
│   ├── normal_1.mat
│   ├── normal_2.mat
│   └── normal_3.mat
├── Ball/                   # 球故障（不同尺寸）
│   ├── 0007/               # 0.007" 缺陷
│   │   ├── B007_0.mat ~ B007_3.mat
│   ├── 0014/               # 0.014" 缺陷
│   ├── 0021/               # 0.021" 缺陷
│   └── 0028/               # 0.028" 缺陷
├── Inner Race/             # 内圈故障（不同尺寸）
│   ├── 0007/ ~ 0028/
│   └── [同上结构]
└── Outer Race/             # 外圈故障（不同位置）
    ├── Centered/           # 故障位置：圆心处（6点钟位置）
    │   ├── 0007/ ~ 0021/
    │   └── OR00X@6_0.mat ~ OR00X@6_3.mat
    ├── Opposite/           # 故障位置：相对位置（12点钟位置）
    │   ├── 0007/ ~ 0021/
    │   └── OR00X@12_0.mat ~ OR00X@12_3.mat
    └── Orthogonal/         # 故障位置：正交位置
        ├── 0007/
        │   ├── 144.mat ~ 147.mat
        └── 0021/
            ├── 246.mat ~ 249.mat
```

**数据说明**：
- **采样率**：12000 Hz
- **振动通道**：驱动端加速度计（DE_time）
- **样本数量**：每个条件4个文件，每个文件可通过滑窗切片生成多个样本
- **故障类型数**：5类（Normal, Ball, InnerRace, OuterRace-Centered, OuterRace-Opposite, OuterRace-Orthogonal）

---

### 模型检查点

```
checkpoints/
├── baseline_cnn_best.pth       # BaselineCNN最佳模型权重
├── resnet18_best.pth           # ResNet18最佳模型权重
└── mobilenet_v2_best.pth       # MobileNetV2最佳模型权重
```

### 结果输出

```
results/
├── baseline_cnn_classification_report.txt        # BaselineCNN分类报告
├── resnet18_classification_report.txt            # ResNet18分类报告
└── mobilenet_v2_classification_report.txt        # MobileNetV2分类报告
```

---

## 安装与运行

### 1. 环境要求
- Python 3.8+
- Windows/Linux/macOS
- CPU或NVIDIA GPU（可选，推荐用于大规模训练）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

**依赖包**：
- **numpy** (≥1.24)：数值计算
- **scipy** (≥1.10)：科学计算（loadmat、STFT等）
- **torch** (≥2.1)：深度学习框架
- **torchvision** (≥0.16)：预训练模型库

### 3. 配置模型和设备

编辑 `config.py`，修改以下参数：

```python
# 选择模型（3选1）
CFG.model.name = "baseline_cnn"   # 或 "resnet18" 或 "mobilenet_v2"

# 选择设备
CFG.train.device = "cpu"           # 或 "cuda:0"（GPU）
```

### 4. 运行训练

```bash
python train.py
```

**程序输出示例**：
```
总样本数: 2048
样本形状: (1, 32, 64)
划分大小 - Train: 1638, Test: 410
模型: ResNet18
参数总数: 11,189,760
平均推理时间: 12.34 ms
...
Epoch 1/50
  Train Loss: 1.523, Accuracy: 0.456
  Test  Loss: 1.381, Accuracy: 0.512
...
最佳模型已保存: checkpoints/resnet18_best.pth
```

### 5. 生成评估报告

```bash
python plot_confusion_matrix.py --model_name mobilenet_v2
```

**输出内容**：
- 混淆矩阵热力图（PNG格式）
- 详细分类报告（TXT格式）
  - 各类别的 Precision、Recall、F1-score
  - 宏平均和加权平均指标

---

## 快速调试指南

### 测试数据加载
```bash
python test_load_mat.py
```
验证 .mat 文件的可读性和数据形状。

### 测试FFT特征
```bash
python test_fft.py
```
绘制原始信号和FFT频谱，检查特征提取的合理性。

### 运行完整流程
```bash
python test1.py
```
快速验证数据加载→特征提取→模型推理的完整流程。

---

## 论文参考建议

在论文中应清晰说明以下内容（可直接引用config.py）：

1. **数据采集参数**
   - 采样率：12000 Hz
   - 轴承类型和故障类型
   - 每种工况的样本数量

2. **特征提取方案**
   - 时间窗口：2048点（170ms）
   - 滑窗步长：1024点（50%重叠）
   - STFT参数：窗长64、FFT点数128
   - 频带范围：200-3200Hz

3. **模型架构对比**
   - BaselineCNN：参数量最少，作为baseline
   - ResNet18：中等复杂度，性能基准
   - MobileNetV2：轻量级，适合部署

4. **数据划分方式**
   - 按文件级划分（防止数据泄漏）
   - Train/Test比例

5. **性能指标**
   - 分类准确率
   - 每个故障类的Precision、Recall、F1-score
   - 模型参数量和推理时间

---

## 常见问题

### Q1: 如何修改STFT参数？
**A**：编辑 `config.py` 中的 DataConfig 类，修改 `nperseg`, `noverlap`, `nfft` 等参数。重新运行训练即可生效。

### Q2: 如何添加新的故障类型？
**A**：
1. 在 `data/` 目录下新建文件夹
2. 在 `config.py` 的 `class_to_idx` 中添加映射
3. 将数据文件放入相应文件夹
4. 重新运行训练

### Q3: 训练中断如何继续？
**A**：程序支持从最佳模型恢复。重新运行 `train.py`，将自动加载最佳模型并继续训练。

### Q4: 如何在GPU上训练？
**A**：在 `config.py` 中设置 `CFG.train.device = "cuda:0"`，确保已安装 CUDA 和相应的 PyTorch GPU 版本。

### Q5: 如何提高模型性能？
**A**：
- 调整STFT参数（更精细的时频分辨率）
- 增加数据增强（噪声、时间缩放等）
- 调整模型超参数（学习率、正则化系数）
- 尝试不同的模型架构或集成方法

---

## 版本信息

- **Python**：3.8+
- **PyTorch**：2.1+
- **最后更新**：2026年1月

---

## 许可证

此项目用于学习和研究使用。
