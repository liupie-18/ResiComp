# ResiComp Model Design

This document provides an in-depth explanation of the ResiComp model architecture and its components for loss-resilient image compression.

## Overview

**ResiComp** (Loss-Resilient Image Compression) is a neural image codec designed to maintain good reconstruction quality even when compressed bitstreams experience packet loss during transmission over unreliable networks (e.g., wireless networks).

### Key Innovation: Dual-Functional Masked Visual Token Modeling

The core innovation is the **Dual-Functional Transformer** that serves two purposes simultaneously:
1. **Entropy Prediction**: Estimates probability distributions for arithmetic coding
2. **Token Prediction**: Predicts missing/masked tokens for error resilience

This dual functionality enables the model to both compress efficiently and recover gracefully from packet loss.

## Architecture Components

### 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        ResiComp Model                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Input Image (3×H×W)                                         │
│       ↓                                                       │
│  ┌──────────────────────┐                                    │
│  │  ELIC Analysis (g_a) │  ← Encoder                         │
│  │  - Conv + GDN (4×)   │                                    │
│  │  - 16× downsampling  │                                    │
│  └──────────────────────┘                                    │
│       ↓                                                       │
│  Latent y (192×H/16×W/16)                                    │
│       ↓                                                       │
│  Quantization (STE)                                          │
│       ↓                                                       │
│  ┌────────────────────────────────────────┐                  │
│  │  Dual-Functional Transformer (DFT)     │  ← Innovation   │
│  │  ┌──────────────────────────────────┐  │                  │
│  │  │ 1. Embedding Layer              │  │                  │
│  │  │ 2. Transformer Blocks (12×)     │  │                  │
│  │  │    - Swin Transformer           │  │                  │
│  │  │    - Window Attention           │  │                  │
│  │  │    - MLP                         │  │                  │
│  │  │ 3. Dual Prediction Heads:       │  │                  │
│  │  │    a) Entropy Parameters        │  │                  │
│  │  │       → GMM (probs, μ, σ)       │  │                  │
│  │  │    b) Latent Prediction         │  │                  │
│  │  │       → Missing tokens          │  │                  │
│  │  └──────────────────────────────────┘  │                  │
│  └────────────────────────────────────────┘                  │
│       ↓                        ↓                             │
│  Likelihoods          Predicted Latents                      │
│       ↓                        ↓                             │
│  ┌──────────────────────┐  ┌──────────────────────┐         │
│  │ ELIC Synthesis (g_s) │  │ ELIC Synthesis (g_s) │         │
│  │ - Deconv + iGDN (4×) │  │ - Deconv + iGDN (4×) │         │
│  │ - Attention Blocks   │  │ - Attention Blocks   │         │
│  └──────────────────────┘  └──────────────────────┘         │
│       ↓                        ↓                             │
│  x_hat (full recon)       x_check (resilient recon)         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2. ELIC Encoder-Decoder

Based on the state-of-the-art ELIC (Efficient Learned Image Compression) architecture:

#### Analysis Transform (g_a)
- **Input**: RGB image (3 channels)
- **Architecture**: 4 convolutional layers with GDN (Generalized Divisive Normalization)
- **Downsampling**: 16× (each layer has stride 2)
- **Output**: Latent representation with 192 channels
- **Purpose**: Transforms image into a compact latent space

```python
Conv(3→128, k=5, s=2) → GDN 
  → Conv(128→128, k=5, s=2) → GDN
  → Conv(128→128, k=5, s=2) → GDN
  → Conv(128→192, k=5, s=2)
```

#### Synthesis Transform (g_s)
- **Input**: Latent representation (192 channels)
- **Architecture**: 4 transposed convolutional layers with inverse GDN
- **Upsampling**: 16× (reconstructs original image resolution)
- **Output**: RGB image (3 channels)
- **Enhancement**: Includes residual attention blocks for improved quality

```python
Deconv(192→128, k=5, s=2) → iGDN
  → Deconv(128→128, k=5, s=2) → iGDN + Attention
  → Deconv(128→128, k=5, s=2) → iGDN + Attention
  → Deconv(128→3, k=5, s=2)
```

### 3. Dual-Functional Transformer

The heart of ResiComp, implementing masked visual token modeling with dual prediction heads.

#### 3.1 Architecture Details

**Input Processing**:
- Latent representation: `y_hat` (B×192×H×W)
- Flattened to tokens: (B×L×192) where L = H×W
- Masked tokens replaced with learnable `mask_token`

**Embedding**:
```python
x = latent / delta  # Normalization (delta=5.0)
x = Linear(192 → 768)  # Embedding to transformer dimension
```

**Transformer Blocks** (12 layers):
- **Type**: Swin Transformer (shifted window attention)
- **Dimension**: 768
- **Heads**: 8
- **Window Size**: 4×4
- **Shift Strategy**: Alternating (0 and window_size//2)
- **MLP Ratio**: 4× (768 → 3072 → 768)

**Output Heads**:

1. **Entropy Parameters Head**:
   ```python
   Conv(768 → 3072, k=1) → GELU
     → Conv(3072 → 3072, k=1) → GELU
     → Conv(3072 → 192×9, k=1)
   
   Output: Gaussian Mixture Model parameters
     - probs: (3×B×192×H×W) - mixture weights
     - means: (3×B×192×H×W) - component means
     - scales: (3×B×192×H×W) - component scales
   ```

2. **Prediction Head**:
   ```python
   Conv(768 → 3072, k=1) → GELU
     → Conv(3072 → 3072, k=1) → GELU
     → Conv(3072 → 192, k=1)
   
   Output: Predicted latent values (B×192×H×W)
   ```

#### 3.2 Masked Visual Token Modeling

**Training Phase**:
1. Random masking: 5%-99% of tokens masked randomly
2. Forward pass with masked input
3. Predict entropy parameters for ALL tokens
4. Predict values for MASKED tokens
5. Compute dual loss:
   - Rate loss from entropy estimates
   - Distortion loss from predictions

**Inference Phase** (Autoregressive):
1. Use specific coding order (e.g., QLDS - Quasi-Low Discrepancy Sequence)
2. Encode/decode tokens sequentially
3. Each token uses previously decoded tokens as context
4. Entropy parameters adapt based on available context

### 4. Gaussian Mixture Model Entropy Model

Uses a 3-component Gaussian mixture for flexible probability modeling:

**Probability Calculation**:
```python
p(y) = Σ(i=0 to 2) probs[i] × N(y | means[i], scales[i])
```

Where:
- `probs`: Mixture weights (sum to 1 via softmax)
- `N(y | μ, σ)`: Gaussian cumulative distribution for quantized values
- Clamped to [y-0.5, y+0.5] for discrete symbols

**Laplace Smoothing**:
```python
p_final = 0.999 × p_GMM + 0.001 × p_Laplace
```
Prevents zero probabilities that would break arithmetic coding.

**Compression/Decompression**:
- Uses range coding (via `constriction` library)
- Entropy-optimal compression
- Symbol-by-symbol encoding based on predicted PMF

### 5. Coding Order Strategies

Different strategies for autoregressive encoding/decoding:

#### 5.1 QLDS (Quasi-Low Discrepancy Sequence)
- **Purpose**: Spatially balanced token ordering
- **Algorithm**: Based on golden ratio φ = 1.324...
- **Advantage**: Good context from all directions
- **Usage**: Default for standard compression

#### 5.2 Progressive Coding Order
Enables quality scalability:
```python
ratio = (step_i / total_steps) ^ beta
tokens_to_encode = floor(L × ratio)
```
- `beta=2.2`: Controls progression curve
- Early steps: Few tokens (low quality)
- Later steps: More tokens (higher quality)

#### 5.3 Checkerboard Patterns
- `checkerboard2`: 2-stage encoding (like chess board)
- `checkerboard4`: 4-stage encoding
- Fast but less flexible context

### 6. Packet Loss Resilience

#### 6.1 Packet Partitioning

The latent space is divided into packets using QLDS-based ordering:

```python
# Adjust packet size based on dependencies
packet_size[i] ∝ (num_dependencies[i] / total_packets + 1) ^ beta

# Partition latents
for i in range(num_packets):
    packet[i] = latents[partition_i]
```

#### 6.2 Context Modes

Different dependency structures between packets:

**IntraSlice** (No inter-packet dependency):
```
[P0] [P1] [P2] [P3] ...
  ↓    ↓    ↓    ↓
Each packet encoded independently
```
- Maximum resilience
- Lower compression efficiency

**Layered** (Sequential dependency):
```
[P0] → [P1] → [P2] → [P3] ...
```
- P0: Base layer (critical)
- P1: Uses P0 as context
- P2: Uses P0, P1 as context, etc.

**MultiDescription_N** (N-description coding):
```
N=2 example:
[P0] [P1] [P2] [P3] [P4] ...
  ↓     ↓    ↓     ↓
      ↘   ↙      ↘   ↙
  P2 uses P0   P4 uses P2
  P3 uses P1   P5 uses P3
```
- Balanced resilience and efficiency
- N parallel streams

#### 6.3 Packet Loss Recovery

When packets are lost:

1. **Identify Received Packets**: Check which packets arrived
2. **Validate Dependencies**: 
   - If packet P[i] depends on P[j] and P[j] is lost, mark P[i] as unusable
3. **Context Construction**:
   - Build mask of available tokens
4. **Token Prediction**:
   - Use Dual-Functional Transformer
   - Predict missing tokens based on received ones
5. **Reconstruction**:
   - Combine received and predicted tokens
   - Decode with synthesis transform

**Example**:
```
Sent: [P0] [P1] [P2] [P3] [P4]
Lost:       X           X
Received: [P0] [P2] [P3]

IntraSlice mode:
  - All received packets are usable
  - Predict tokens from P1 and P4 using P0, P2, P3

Layered mode:
  - P0: usable (no dependency)
  - P2: NOT usable (depends on P1 which is lost)
  - P3: NOT usable (depends on P2)
  - Only use P0, predict rest
```

### 7. Network Simulation

#### Three-State Markov Model

Simulates realistic wireless packet loss:

**States**:
- State 1: Good (no loss)
- State 2: Loss (packet dropped)
- State 3: Burst loss recovery

**Transition Probabilities**:
```
P(1→2) = P       # Enter loss state
P(2→1) = p       # Exit loss state
P(2→2) = q       # Continue loss (burst)
P(3→3) = Q'      # Stay in recovery
```

**Network Conditions** (EP1-EP6):
- EP1: 0.2% loss, burst length 6.5 (light)
- EP2: 3.1% loss, burst length 1.6 (moderate)
- EP3: 6.5% loss, burst length 5.0 (moderate-heavy)
- EP4: 13.8% loss, burst length 1.7 (heavy)
- EP5: 21.4% loss, burst length 10.0 (very heavy)
- EP6: 32.4% loss, burst length 2.7 (extreme)

## Training Strategy

### Dual Loss Function

```python
# Main reconstruction loss
L_main = λ × D(x, x_hat)

# Resilient reconstruction loss
L_resilient = λ × α × D(x, x_check)

# Rate loss
L_rate = -log2(likelihoods).mean()

# Combined
L_total = (L_main + L_resilient) / (1 + α) + L_rate
```

Where:
- `D()`: Distortion metric (MSE or MS-SSIM)
- `λ`: Rate-distortion trade-off (e.g., 0.0035)
- `α`: Resilience weight (e.g., 0.1)
- `x_hat`: Full reconstruction from all tokens
- `x_check`: Reconstruction from predicted tokens (simulating loss)

### Training Process

1. **Random Masking**: Randomly mask 5-99% of latent tokens
2. **Dual Forward Pass**: 
   - Predict entropy parameters for ALL tokens
   - Predict values for MASKED tokens
3. **Loss Computation**:
   - Rate: From entropy estimates
   - Distortion: From both full and predicted reconstructions
4. **Optimization**: Adam with warmup and step decay

### Hyperparameters

- Learning rate: 1e-4
- Weight decay: 0.03
- Gradient clipping: 1.0
- Batch size: 8-16
- Training epochs: 8-10
- Lambda warmup: 10× for first 10% of training

## Inference Modes

### 1. Standard Inference (No Packet Loss)

```python
inference_without_packet_loss(image, step=12, beta=2.2)
```

- Autoregressive encoding with QLDS order
- Full quality reconstruction
- Outputs: x_hat, likelihoods (for BPP calculation)

### 2. Progressive Inference

```python
inference_for_progressive_decoding(image, step=64)
```

- Generates 64 intermediate reconstructions
- Each step adds more tokens
- Enables quality scalability
- Outputs: List of (x_check, likelihoods) for each step

### 3. Packet Loss Inference

```python
inference_with_packet_loss(image, packet_tracer, context_mode)
```

- Simulates packet transmission with loss
- Tests resilience across 100 trials
- Varies context modes (IntraSlice, Layered, MultiDescription)
- Outputs: Average PSNR, BPP

### 4. Real Compression/Decompression

```python
# Compression
strings = mim.compress(latent, context_mode='qlds')

# Decompression
latent_hat = mim.decompress(strings, latent.shape, device)
```

- Actual bitstream generation
- Range coding for entropy coding
- Returns compressed byte strings

## Performance Characteristics

### Rate-Distortion Trade-off

Controlled by `lambda`:
- λ = 0.0017: ~0.15 bpp, ~32 dB PSNR
- λ = 0.0035: ~0.25 bpp, ~34 dB PSNR
- λ = 0.0067: ~0.40 bpp, ~36 dB PSNR

### Resilience vs. Efficiency

Controlled by context mode and `alpha`:

**Alpha Parameter**:
- α = 0.0: No resilience training, best R-D but poor packet loss recovery
- α = 0.1: Balanced (recommended)
- α = 1.0: Strong resilience, slight R-D penalty

**Context Modes**:
- IntraSlice: Best resilience, ~5% bitrate increase
- MultiDescription_2: Good balance
- Layered: Best compression, vulnerable to early packet loss

## Key Design Decisions

### 1. Why Dual Functionality?

**Problem**: Traditional codecs need separate error correction codes
- Adds overhead
- Fixed resilience (can't adapt)

**Solution**: Unified model that both compresses and recovers
- No separate FEC needed
- Adaptive resilience based on actual loss pattern

### 2. Why Transformer?

**Advantages**:
- Long-range dependencies for better context
- Attention mechanism naturally handles missing tokens
- Window-based design (Swin) reduces complexity

### 3. Why Gaussian Mixture?

**Single Gaussian**: Too restrictive for complex latent distributions
**Mixture of 3**: 
- Captures multi-modal distributions
- Flexible enough for various contexts
- Not too many parameters (3 is empirically optimal)

### 4. Why QLDS Ordering?

**Raster Scan**: Biased context (only from top-left)
**Random**: Poor spatial correlation
**QLDS**: 
- Quasi-random but deterministic
- Spatially balanced
- Good for both compression and resilience

## Comparison with Prior Art

### vs. Traditional Codecs (JPEG, H.265)
- **ResiComp**: Learned, adaptive, resilient
- **Traditional**: Hand-crafted, requires separate FEC

### vs. Other Learned Codecs
- **Hyperprior/Autoregressive**: Good R-D but no resilience
- **ResiComp**: Adds packet loss resilience with minimal overhead

### vs. Multiple Description Coding
- **Traditional MDC**: Fixed descriptions, high redundancy
- **ResiComp**: Flexible, adaptive prediction, lower overhead

## Extension Possibilities

1. **Video Compression**: Extend to temporal domain
2. **Joint Source-Channel Coding**: End-to-end with channel model
3. **Unequal Error Protection**: Adaptive alpha per packet
4. **Real-time Adaptation**: Adjust context mode based on network feedback

## Summary

ResiComp achieves loss-resilient image compression through:

1. **Dual-Functional Transformer**: Unified entropy prediction and token prediction
2. **Flexible Context Modeling**: Multiple dependency structures for different scenarios
3. **Gaussian Mixture Entropy Model**: Accurate probability estimation
4. **Smart Ordering**: QLDS and progressive coding orders
5. **Packet-level Design**: Explicit modeling of packet-based transmission

The result is a codec that maintains good quality even under significant packet loss, with minimal overhead compared to traditional compression-only methods.
