# ResiComp 项目结构说明

本文档提供 ResiComp 项目结构的全面概述，帮助开发者理解代码库的组织方式。

## 目录结构

```
ResiComp/
├── config/                 # 配置文件
│   └── resicom.yaml       # 训练和测试配置
├── data/                   # 数据集处理
│   └── datasets.py        # OpenImages 和 Kodak 数据集加载器
├── figs/                   # 文档图片
│   └── Fig_framework.png  # ResiComp 框架图
├── layer/                  # 神经网络层组件
│   ├── layer_utils.py     # 工具函数（卷积、量化等）
│   └── transformer.py     # Transformer 模块（Swin、窗口注意力）
├── loss/                   # 损失函数
│   ├── distortion.py      # MSE 和 MS-SSIM 损失封装
│   ├── gan_loss.py        # 基于 GAN 的损失函数
│   ├── vgg_loss.py        # VGG 感知损失
│   └── perceptual_similarity/  # LPIPS 感知损失
├── net/                    # 网络架构
│   ├── elic.py            # ELIC 编码器/解码器（分析/合成变换）
│   └── resicomp.py        # 主要的 ResiComp 模型（含双功能 Transformer）
├── network/                # 网络仿真工具
│   └── packet_tracer.py   # 丢包仿真（马尔可夫模型）
├── main.py                 # 训练和评估入口
├── utils.py                # 工具函数（日志、检查点、指标）
└── README.md               # 项目概述和安装说明
```

## 关键组件

### 1. 配置 (`config/`)
- **resicom.yaml**: 包含所有训练和测试的超参数
  - 模型设置（alpha、lambda 值）
  - 训练参数（学习率、轮数、批大小）
  - 数据集路径
  - 检查点路径

### 2. 数据模块 (`data/`)
- **datasets.py**: 数据加载和预处理
  - `OpenImages`: 训练数据集，带随机裁剪
  - `Datasets`: 测试数据集（Kodak），无数据增强
  - 支持单 GPU 和多 GPU 训练的数据加载器

### 3. 层组件 (`layer/`)
- **layer_utils.py**: 基础构建模块
  - `quantize_ste()`: 量化的直通估计器
  - `make_conv()`, `make_deconv()`: 卷积层构造函数
  
- **transformer.py**: Transformer 架构组件
  - `TransformerBlock`: 标准 Transformer 块
  - `SwinTransformerBlock`: 带移位窗口的 Swin Transformer
  - `WindowAttention`: 基于窗口的多头自注意力
  - `Mlp`: 前馈网络
  - 窗口划分/还原工具

### 4. 损失函数 (`loss/`)
- **distortion.py**: 图像质量指标
  - MSE（均方误差）
  - MS-SSIM（多尺度结构相似性）
  
- **gan_loss.py**: 对抗训练损失
- **vgg_loss.py**: 基于 VGG 特征的感知损失
- **perceptual_similarity/**: LPIPS 感知相似性指标

### 5. 网络架构 (`net/`)

#### ELIC 组件 (`elic.py`)
- `ELICAnalysis`: 编码器（分析变换）
  - 4 个带 GDN 的卷积层
  - 16 倍下采样（从图像到潜在表示）
  
- `ELICSynthesis`: 解码器（合成变换）
  - 4 个带逆 GDN 的转置卷积层
  - 16 倍上采样（从潜在表示到图像）
  - 包含注意力模块以提升质量

#### ResiComp 模型 (`resicomp.py`)
抗丢失图像压缩系统的核心实现：

- **`GaussianMixtureEntropyModel`**: 高斯混合模型熵编码
  - 压缩/解压缩潜在表示
  - 使用范围编码（通过 constriction 库）
  
- **`DualFunctionalTransformer`**: ResiComp 的主要创新
  - 双重功能：熵预测 + 掩码令牌预测
  - 基于 Transformer 的上下文建模
  - 支持多种编码顺序（QLDS、棋盘格等）
  
- **`ResiComp`**: 完整的压缩模型
  - 结合 ELIC 编码器/解码器与双功能 Transformer
  - 多种推理模式（有/无丢包、渐进式）

### 6. 网络仿真 (`network/`)
- **packet_tracer.py**: 丢包仿真
  - `Random_packet_tracer`: 随机丢包
  - `Two_state_Markov_wlan_packet_tracer`: 两状态马尔可夫模型
  - `Three_state_Markov_wlan_packet_tracer`: 三状态马尔可夫模型（EP1-EP6）
  - 模拟真实的无线网络条件

### 7. 训练脚本 (`main.py`)
训练和评估的入口点：

- **训练函数**:
  - `train_one_epoch()`: 单轮训练循环
  - 使用 MSE 或 MS-SSIM 进行率失真优化
  - 双重损失训练：重建（x_hat）+ 弹性预测（x_check）
  
- **测试函数**:
  - `rd_test()`: 无丢包的率失真测试
  - `progressive_test()`: 渐进式解码评估
  - `packet_loss_test()`: 丢包下的弹性测试
  
- **配置**:
  - 支持单 GPU 和多 GPU 训练
  - WandB 集成用于实验追踪
  - 可配置的失真指标（MSE/MS-SSIM）

### 8. 工具 (`utils.py`)
训练基础设施的辅助函数：

- `logger_configuration()`: 设置日志和输出目录
- `save_checkpoint()`: 保存模型检查点
- `load_weights()`: 加载模型权重
- `AverageMeter`: 计算指标的运行平均值
- `worker_init_fn_seed()`: 可重现的数据加载

## 数据流

### 训练流程
```
输入图像 (3×H×W)
    ↓
ELIC 编码器 (g_a)
    ↓
潜在表示 (192×H/16×W/16)
    ↓
量化 (STE)
    ↓
双功能 Transformer（掩码视觉令牌建模）
    ↓
├─ 熵参数（概率、均值、尺度）
├─ 似然 → BPP 损失
└─ 潜在预测 → 弹性重建
    ↓
ELIC 解码器 (g_s)
    ↓
├─ x_hat: 完整重建
└─ x_check: 弹性重建（来自预测令牌）
    ↓
损失 = λ × (MSE(x_hat) + α × MSE(x_check)) / (1 + α) + BPP
```

### 推理流程（无丢包）
```
输入图像 → g_a → 量化潜在 → 自回归编码
    → 熵编码 → 码流
    
码流 → 熵解码 → 自回归解码
    → 重建潜在 → g_s → 输出图像
```

### 推理流程（有丢包）
```
输入图像 → g_a → 量化潜在 → 分包编码
    → 码流（分包） → 网络（有丢失） → 接收包
    
接收包 → 上下文感知解码 → 令牌预测（针对丢失包）
    → 重建潜在 → g_s → 输出图像
```

## 关键依赖

- **PyTorch**: 深度学习框架
- **CompressAI**: 压缩构建模块（GDN、AttentionBlock）
- **constriction**: 用于熵编码的范围编码
- **timm**: Transformer 组件（DropPath 等）
- **torchvision**: 图像变换
- **PIL**: 图像加载
- **wandb**: 实验追踪（可选）

## 配置文件

### config/resicom.yaml
关键参数：
- `lambda_value`: 率失真权衡（0.0035 用于低比特率）
- `alpha_value`: 弹性重建损失权重（0.1）
- `distortion_metric`: 'MSE' 或 'MS-SSIM'
- `checkpoint`: 预训练模型路径
- `eval-dataset-path`: 测试数据集路径（Kodak）

## 输出结构

训练时会创建以下结构：
```
history/
└── <method>/
    └── <exp_name>/
        ├── samples/        # 可视化样本
        ├── models/         # 保存的检查点
        └── <exp_name>.log  # 训练日志
```

保存的检查点：
- `checkpoint_best_loss.pth.tar`: 验证损失最佳的模型
- `EP<epoch>.pth.tar`: 定期的轮次检查点

## 测试模式

代码支持三种测试模式：

1. **标准率失真测试** (`rd_test`):
   - 评估无丢包情况下的率失真性能
   - 输出：PSNR、MS-SSIM、BPP

2. **渐进式解码** (`progressive_test`):
   - 测试不同解码阶段的质量
   - 用于分析渐进式传输

3. **丢包测试** (`packet_loss_test`):
   - 使用马尔可夫模型（EP1-EP6）模拟丢包
   - 测试不同上下文模式下的弹性：
     - `Layered`: 分层编码
     - `IntraSlice`: 仅片内（无片间依赖）
     - `MultiDescription_N`: 多描述编码（N=2,3,4,5）
     - `SLC`: 单循环编码
     - `TwoLoop`: 双循环编码

## 开发工作流

1. **准备数据集**: 将训练图像放在指定路径，Kodak 数据集用于测试
2. **配置**: 编辑 `config/resicom.yaml` 设置所需的超参数
3. **训练**: 运行 `python main.py --config ./config/resicom.yaml`
4. **评估**: 使用 `--test-only` 标志进行评估
5. **监控**: 使用 WandB 或日志文件跟踪训练进度

## 代码风格和约定

- **设备管理**: 支持单 GPU 和多 GPU（DistributedDataParallel）
- **日志记录**: 全面记录指标（PSNR、MS-SSIM、BPP、损失）
- **检查点**: 定期自动保存和最佳模型跟踪
- **可重现性**: 设置种子以获得可重现的结果
- **梯度裁剪**: 用于稳定训练（`clip_max_norm=1.0`）
