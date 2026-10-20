# Convolutional Vision Transformer (CvT) for Image Classification

## Overview
This module implements a **Convolutional Vision Transformer (CvT)** architecture in PyTorch for image classification on the CIFAR-10 benchmark. Following the architectural paradigms introduced by Wu et al. (*"CvT: Introducing Convolutions to Vision Transformers"*, ICCV 2021) and Hassani et al. (*"Escaping the Big Data Trap with Compact Vision Transformers"*, 2021), this model synthesizes the best characteristics of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs).

```
+-----------------------------------------------------------------------------------------+
|                      CONVOLUTIONAL VISION TRANSFORMER (CvT) ARCHITECTURE                |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|   Input Image (32x32x3)                                                                 |
|            |                                                                            |
|            v                                                                            |
|   +----------------------------------------------------+                                |
|   |         Multi-Stage Convolutional Tokenizer        |                                |
|   |   Conv2D (3x3, s=2, 64ch) -> BN -> GELU (16x16)    |                                |
|   |   Conv2D (3x3, s=2, 128ch)-> BN -> GELU (8x8)      |                                |
|   +----------------------------------------------------+                                |
|            |                                                                            |
|            v                                                                            |
|   Flatten Spatial Grid to Tokens (64 Tokens x 128 Dims)                                 |
|            |                                                                            |
|            v                                                                            |
|   +----------------------------------------------------+                                |
|   |      Stacked Conv-Transformer Blocks (x4 Layers)   |                                |
|   |   +--------------------------------------------+   |                                |
|   |   | LayerNorm -> Multi-Head Self-Attention     |   |                                |
|   |   |           + Residual Connection            |   |                                |
|   |   +--------------------------------------------+   |                                |
|   |   | LayerNorm -> Linear Projection             |   |                                |
|   |   | -> Depthwise Conv2D (3x3) -> BN -> GELU    |   |  <-- ConvFFN                   |
|   |   | -> Linear Projection + Residual Connection |   |                                |
|   |   +--------------------------------------------+   |                                |
|   +----------------------------------------------------+                                |
|            |                                                                            |
|            v                                                                            |
|   Global Average Pooling (Tokens -> Feature Vector)                                     |
|            |                                                                            |
|            v                                                                            |
|   LayerNorm -> Linear Classification Head (128 -> 10 Classes)                           |
+-----------------------------------------------------------------------------------------+
```

---

## Architectural Formulation & Bridging the Inductive Bias Gap

Standard Vision Transformers (ViTs) partition images into rigid, non-overlapping patches and rely entirely on global self-attention. While effective at massive scale (e.g. 300M images), pure ViTs struggle on smaller datasets because they lack **inductive bias** (locality and translation equivariance).

CvT directly resolves this limitation through two structural mechanisms:

### 1. Multi-Stage Overlapping Convolutional Tokenizer
Instead of a single large linear projection, CvT tokenizes images using overlapping convolutions with stride 2:
$$\mathbf{X}_1 = \text{GELU}\left(\text{BN}\left(\text{Conv2D}_{3 \times 3, s=2}(\mathbf{X})\right)\right) \in \mathbb{R}^{B \times 64 \times 16 \times 16}$$
$$\mathbf{X}_{\text{tokens}} = \text{GELU}\left(\text{BN}\left(\text{Conv2D}_{3 \times 3, s=2}(\mathbf{X}_1)\right)\right) \in \mathbb{R}^{B \times 128 \times 8 \times 8}$$

- **Overlapping Receptive Field:** Each token has an effective receptive field of $7 \times 7$ pixels, preserving continuous edge contours and local textures.
- **Implicit Position Modeling:** Convolutions naturally encode spatial coordinate relationships, eliminating the need for hardcoded or brittle 1D position embeddings.

### 2. Depthwise Convolutional Feed-Forward Network (ConvFFN)
Within each Transformer block, the conventional MLP is upgraded to a Convolutional FFN:
$$\mathbf{H} = \text{Linear}_{D \to 2D}\left(\text{LN}(\mathbf{z})\right)$$
$$\mathbf{H}_{\text{conv}} = \text{GELU}\left(\text{BN}\left(\text{DWConv}_{3 \times 3}\left(\text{Reshape}_{2\text{D}}(\mathbf{H})\right)\right)\right)$$
$$\text{ConvFFN}(\mathbf{z}) = \text{Linear}_{2D \to D}\left(\text{Flatten}(\mathbf{H}_{\text{conv}})\right)$$

```mermaid
flowchart LR
    In["Input Image (32x32x3)"] --> C1["Conv 3x3 s=2 (16x16x64)"]
    C1 --> C2["Conv 3x3 s=2 (8x8x128)"]
    C2 --> Flat["Flatten (64 Tokens x 128)"]
    Flat --> MHA["Multi-Head Attention (4 Heads)"]
    MHA --> Res1["Add Residual"]
    Res1 --> CFFN["Depthwise ConvFFN (3x3 DWConv)"]
    CFFN --> Res2["Add Residual"]
    Res2 --> GAP["Global Average Pooling"]
    GAP --> LN["LayerNorm"]
    LN --> Head["Linear Classifier (10 Classes)"]
```

---

## Evaluation Metrics

1. **Overall Categorical Accuracy:** Multi-class classification accuracy across all 10 CIFAR-10 categories.
2. **Macro Precision, Recall, and F1-Score:** Unweighted macro-averaged balance metrics.
3. **10x10 Confusion Matrix:** Inter-class prediction patterns and semantic error distributions.

---

## Repository Structure

```
convolutional-vision-transformer/
├── README.md                              # Technical formulation and mathematical report
├── convolutional_vision_transformer.py    # PyTorch CvT architecture & pipeline script
├── cvt_classification_results.json        # Machine-readable evaluation metrics and history
└── cvt_classification_results.png         # Empirical learning curves and confusion matrix heatmap
```

---

## Google Colab / Local Execution Guide

### Hardware Requirements
- **Recommended:** Google Colab (Standard CPU or free T4 GPU).
- **Dependencies:** `torch`, `torchvision`, `matplotlib`, `numpy`.

### Execution Command
```bash
python3 convolutional-vision-transformer/convolutional_vision_transformer.py
```

---

## Empirical Evaluation & Results

### Benchmark Summary Metrics (Unseen CIFAR-10 Test Set, N = 3,000)
| Metric | Pure ViT Baseline | Convolutional ViT (CvT) | Delta / Gain |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | `51.90%` | **`65.10%`** | **`+13.20%`** |
| **Macro-Averaged Precision** | `52.29%` | **`65.03%`** | **`+12.74%`** |
| **Macro-Averaged Recall** | `51.77%` | **`65.10%`** | **`+13.33%`** |
| **Macro-Averaged F1-Score** | `52.03%` | **`65.06%`** | **`+13.03%`** |
| **Final Training Loss** | `1.4522` | **`1.0939`** | `-0.3583` |
| **Final Validation Loss** | `1.4611` | **`1.1420`** | `-0.3191` |

### Epoch Training Progression (15,000 Train / 3,000 Test)
| Epoch | Training Loss | Validation Loss | Validation Accuracy | Epoch Delta |
| :---: | :---: | :---: | :---: | :---: |
| 01 | 1.8432 | 1.6168 | 44.37% | - |
| 02 | 1.5730 | 1.5107 | 50.27% | +5.90% |
| 03 | 1.3921 | 1.3755 | 54.97% | +4.70% |
| 04 | 1.2629 | 1.3335 | 56.93% | +1.96% |
| 05 | 1.1683 | 1.2518 | 62.07% | +5.14% |
| 06 | 1.0939 | 1.1420 | 65.10% | +3.03% |

### Confusion Matrix Breakdown (Class-by-Class Analysis)
- **Top Performing Classes:**
  - `ship`: **257 / 300** correct ($85.7\%$, up from $76.0\%$ in pure ViT)
  - `automobile`: **237 / 300** correct ($79.0\%$, up from $48.3\%$ in pure ViT — a massive **$+30.7\%$** surge!)
  - `frog`: **233 / 300** correct ($77.7\%$, up from $66.3\%$ in pure ViT)
  - `airplane`: **215 / 300** correct ($71.7\%$, up from $54.0\%$ in pure ViT)
  - `truck`: **209 / 300** correct ($69.7\%$, up from $52.7\%$ in pure ViT)
- **Persistent Hard Boundaries:**
  - `cat` $\leftrightarrow$ `dog`: 63 cats misclassified as dogs; 68 dogs misclassified as cats (expected morphological overlap in $32 \times 32$ low resolution).
  - Cross-domain false positives (e.g. animal $\leftrightarrow$ vehicle) remain virtually non-existent ($< 1.0\%$).
