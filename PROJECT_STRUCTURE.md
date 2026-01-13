# ResiComp Project Structure

This document provides a comprehensive overview of the ResiComp project structure to help developers understand the codebase organization.

## Directory Structure

```
ResiComp/
├── config/                 # Configuration files
│   └── resicom.yaml       # Training and testing configuration
├── data/                   # Dataset handling
│   └── datasets.py        # Dataset loaders for OpenImages and Kodak
├── figs/                   # Documentation figures
│   └── Fig_framework.png  # ResiComp framework diagram
├── layer/                  # Neural network layer components
│   ├── layer_utils.py     # Utility functions (convolution, quantization)
│   └── transformer.py     # Transformer blocks (Swin, WindowAttention)
├── loss/                   # Loss functions
│   ├── distortion.py      # MSE and MS-SSIM loss wrappers
│   ├── gan_loss.py        # GAN-based loss functions
│   ├── vgg_loss.py        # Perceptual VGG loss
│   └── perceptual_similarity/  # LPIPS perceptual loss
├── net/                    # Network architectures
│   ├── elic.py            # ELIC encoder/decoder (analysis/synthesis transforms)
│   └── resicomp.py        # Main ResiComp model with Dual-Functional Transformer
├── network/                # Network simulation utilities
│   └── packet_tracer.py   # Packet loss simulation (Markov models)
├── main.py                 # Training and evaluation entry point
├── utils.py                # Utilities (logging, checkpointing, metrics)
└── README.md               # Project overview and setup instructions
```

## Key Components

### 1. Configuration (`config/`)
- **resicom.yaml**: Contains all hyperparameters for training and testing
  - Model settings (alpha, lambda values)
  - Training parameters (learning rate, epochs, batch size)
  - Dataset paths
  - Checkpoint paths

### 2. Data Module (`data/`)
- **datasets.py**: Data loading and preprocessing
  - `OpenImages`: Training dataset with random cropping
  - `Datasets`: Test dataset (Kodak) without augmentation
  - Data loaders for both single-GPU and multi-GPU training

### 3. Layer Components (`layer/`)
- **layer_utils.py**: Fundamental building blocks
  - `quantize_ste()`: Straight-through estimator for quantization
  - `make_conv()`, `make_deconv()`: Convolution layer constructors
  
- **transformer.py**: Transformer architecture components
  - `TransformerBlock`: Standard transformer block
  - `SwinTransformerBlock`: Swin Transformer with shifted windows
  - `WindowAttention`: Window-based multi-head self-attention
  - `Mlp`: Feed-forward network
  - Window partition/reverse utilities

### 4. Loss Functions (`loss/`)
- **distortion.py**: Image quality metrics
  - MSE (Mean Squared Error)
  - MS-SSIM (Multi-Scale Structural Similarity)
  
- **gan_loss.py**: Adversarial training losses
- **vgg_loss.py**: Perceptual loss based on VGG features
- **perceptual_similarity/**: LPIPS perceptual similarity metric

### 5. Network Architectures (`net/`)

#### ELIC Components (`elic.py`)
- `ELICAnalysis`: Encoder (analysis transform)
  - 4 convolutional layers with GDN
  - Downsamples 16x (from image to latent)
  
- `ELICSynthesis`: Decoder (synthesis transform)
  - 4 transposed convolutional layers with inverse GDN
  - Upsamples 16x (from latent to image)
  - Includes attention blocks for quality enhancement

#### ResiComp Model (`resicomp.py`)
Core implementation of the loss-resilient image compression system:

- **`GaussianMixtureEntropyModel`**: Entropy coding with Gaussian mixture models
  - Compresses/decompresses latent representations
  - Uses range coding (via constriction library)
  
- **`DualFunctionalTransformer`**: Main innovation of ResiComp
  - Dual functionality: entropy prediction + masked token prediction
  - Transformer-based context modeling
  - Supports various coding orders (QLDS, checkerboard, etc.)
  
- **`ResiComp`**: Complete compression model
  - Combines ELIC encoder/decoder with Dual-Functional Transformer
  - Multiple inference modes (with/without packet loss, progressive)

### 6. Network Simulation (`network/`)
- **packet_tracer.py**: Packet loss simulation
  - `Random_packet_tracer`: Random packet loss
  - `Two_state_Markov_wlan_packet_tracer`: Two-state Markov model
  - `Three_state_Markov_wlan_packet_tracer`: Three-state Markov model (EP1-EP6)
  - Simulates realistic wireless network conditions

### 7. Training Script (`main.py`)
Entry point for training and evaluation:

- **Training Functions**:
  - `train_one_epoch()`: Single epoch training loop
  - Rate-distortion optimization with MSE or MS-SSIM
  - Dual-loss training: reconstruction (x_hat) + resilient prediction (x_check)
  
- **Testing Functions**:
  - `rd_test()`: Rate-distortion testing without packet loss
  - `progressive_test()`: Progressive decoding evaluation
  - `packet_loss_test()`: Resilience testing under packet loss
  
- **Configuration**:
  - Supports both single-GPU and multi-GPU training
  - WandB integration for experiment tracking
  - Configurable distortion metrics (MSE/MS-SSIM)

### 8. Utilities (`utils.py`)
Helper functions for training infrastructure:

- `logger_configuration()`: Setup logging and output directories
- `save_checkpoint()`: Model checkpoint saving
- `load_weights()`: Model weight loading
- `AverageMeter`: Running average computation for metrics
- `worker_init_fn_seed()`: Reproducible data loading

## Data Flow

### Training Pipeline
```
Input Image (3×H×W)
    ↓
ELIC Encoder (g_a)
    ↓
Latent Representation (192×H/16×W/16)
    ↓
Quantization (STE)
    ↓
Dual-Functional Transformer (masked visual token modeling)
    ↓
├─ Entropy Parameters (probs, means, scales)
├─ Likelihoods → BPP Loss
└─ Latent Prediction → Resilient Reconstruction
    ↓
ELIC Decoder (g_s)
    ↓
├─ x_hat: Full reconstruction
└─ x_check: Resilient reconstruction (from predicted tokens)
    ↓
Loss = λ × (MSE(x_hat) + α × MSE(x_check)) / (1 + α) + BPP
```

### Inference Pipeline (Without Packet Loss)
```
Input Image → g_a → Quantized Latent → Autoregressive Encoding
    → Entropy Coding → Bitstream
    
Bitstream → Entropy Decoding → Autoregressive Decoding
    → Reconstructed Latent → g_s → Output Image
```

### Inference Pipeline (With Packet Loss)
```
Input Image → g_a → Quantized Latent → Packet-wise Encoding
    → Bitstream (packets) → Network (with loss) → Received Packets
    
Received Packets → Context-aware Decoding → Token Prediction (for lost packets)
    → Reconstructed Latent → g_s → Output Image
```

## Key Dependencies

- **PyTorch**: Deep learning framework
- **CompressAI**: Compression building blocks (GDN, AttentionBlock)
- **constriction**: Range coding for entropy coding
- **timm**: Transformer components (DropPath, etc.)
- **torchvision**: Image transformations
- **PIL**: Image loading
- **wandb**: Experiment tracking (optional)

## Configuration Files

### config/resicom.yaml
Key parameters:
- `lambda_value`: Rate-distortion trade-off (0.0035 for low bitrate)
- `alpha_value`: Weight for resilient reconstruction loss (0.1)
- `distortion_metric`: 'MSE' or 'MS-SSIM'
- `checkpoint`: Path to pre-trained model
- `eval-dataset-path`: Path to test dataset (Kodak)

## Output Structure

When training, the following structure is created:
```
history/
└── <method>/
    └── <exp_name>/
        ├── samples/        # Visualization samples
        ├── models/         # Saved checkpoints
        └── <exp_name>.log  # Training logs
```

Checkpoints saved:
- `checkpoint_best_loss.pth.tar`: Best model by validation loss
- `EP<epoch>.pth.tar`: Periodic epoch checkpoints

## Testing Modes

The code supports three testing modes:

1. **Standard R-D Testing** (`rd_test`):
   - Evaluates rate-distortion performance without packet loss
   - Outputs: PSNR, MS-SSIM, BPP

2. **Progressive Decoding** (`progressive_test`):
   - Tests quality at different decoding stages
   - Useful for analyzing progressive transmission

3. **Packet Loss Testing** (`packet_loss_test`):
   - Simulates packet loss with Markov models (EP1-EP6)
   - Tests resilience with different context modes:
     - `Layered`: Layered coding
     - `IntraSlice`: Intra-slice only (no inter-slice dependency)
     - `MultiDescription_N`: Multiple description coding (N=2,3,4,5)
     - `SLC`: Single-loop coding
     - `TwoLoop`: Two-loop coding

## Development Workflow

1. **Prepare Dataset**: Place training images in specified path, Kodak dataset for testing
2. **Configure**: Edit `config/resicom.yaml` with desired hyperparameters
3. **Train**: Run `python main.py --config ./config/resicom.yaml`
4. **Evaluate**: Run with `--test-only` flag for evaluation
5. **Monitor**: Use WandB or log files to track training progress

## Code Style and Conventions

- **Device Management**: Supports both single-GPU and multi-GPU (DistributedDataParallel)
- **Logging**: Comprehensive logging of metrics (PSNR, MS-SSIM, BPP, loss)
- **Checkpointing**: Automatic saving at intervals and best model tracking
- **Reproducibility**: Seed setting for reproducible results
- **Gradient Clipping**: Applied for stable training (`clip_max_norm=1.0`)
