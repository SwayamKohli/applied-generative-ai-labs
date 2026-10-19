# Generative Adversarial Network (AC-GAN) for Handwritten Digit Recognition

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> Auxiliary Classifier Generative Adversarial Network (AC-GAN) trained on the full standard MNIST handwritten digit benchmark (60,000 training samples, 10,000 testing samples). Features a class-conditioned generator synthesizing 28x28 grayscale digits from latent space and a dual-head discriminator that simultaneously authenticates real versus synthetic images and classifies digit categories 0 through 9.

---

## 1. Architectural Design

```mermaid
flowchart TD
    subgraph GEN["1. Conditional Generator G(z, c)"]
        Z["Latent Noise Vector z ~ N(0, I) [100-D]"] --> CAT["Vector Concatenation [110-D]"]
        LBL["Class Label c in {0, ..., 9}"] --> EMB["Class Embedding Layer [10-D]"]
        EMB --> CAT
        CAT --> G_L1["Linear(110 -> 256) + LeakyReLU(0.2) + BatchNorm1d"]
        G_L1 --> G_L2["Linear(256 -> 512) + LeakyReLU(0.2) + BatchNorm1d"]
        G_L2 --> G_L3["Linear(512 -> 1024) + LeakyReLU(0.2) + BatchNorm1d"]
        G_L3 --> G_OUT["Linear(1024 -> 784) + Tanh() -> Shape: 1 x 28 x 28"]
    end

    subgraph DISC["2. Dual-Head Discriminator-Recognizer D(x)"]
        INP["Input Image x (Real MNIST or Generated Fake)"] --> FLAT["Flatten (784-D)"]
        FLAT --> D_L1["Linear(784 -> 512) + LeakyReLU(0.2) + Dropout(0.3)"]
        D_L1 --> D_L2["Linear(512 -> 256) + LeakyReLU(0.2) + Dropout(0.3)"]
        D_L2 --> ADV_HEAD["Adversarial Head: Linear(256 -> 1) + Sigmoid -> P(Real)"]
        D_L2 --> CLS_HEAD["Recognition Head: Linear(256 -> 10) -> Digit Logits (0-9)"]
    end
```

---

## 2. Mathematical Foundations

### 2.1 The Two-Player Minimax Game
Standard GANs formulate generative modeling as a game between a generator $G$ and a discriminator $D$:

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log(1 - D(G(z)))]$$

### 2.2 Auxiliary Classifier GAN (AC-GAN) Formulation
To equip the network with multi-class digit recognition capability, the discriminator is extended with an auxiliary classification head $P(C = c \mid X)$. The total loss combines adversarial authenticity and class probability:

$$\mathcal{L}_{\text{adv}} = \mathbb{E}[\log P(S = \text{real} \mid x_{\text{real}})] + \mathbb{E}[\log P(S = \text{fake} \mid x_{\text{fake}})]$$
$$\mathcal{L}_{\text{cls}} = \mathbb{E}[\log P(C = c \mid x_{\text{real}})] + \mathbb{E}[\log P(C = c \mid x_{\text{fake}})]$$

The objective functions optimized by the two networks are:
- **Discriminator Objective**:
  $$\max_D \left( \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{cls}} \right)$$
- **Generator Objective**:
  $$\max_G \left( \mathbb{E}[\log P(S = \text{real} \mid x_{\text{fake}})] + \mathcal{L}_{\text{cls}} \right)$$

This forces the generator to produce samples that are visually indistinguishable from genuine MNIST digits while exhibiting clear semantic features of the requested digit class $c \in \{0, \dots, 9\}$.

---

## 3. Empirical Training & Recognition Results

### 3.1 Minimax Dynamics & Synthesized Digit Dashboard

![GAN Digit Recognition Results](gan_digit_recognition_results.png)

### 3.2 Evaluation Metrics on Unseen Test Partition ($N = 10,000$)

Evaluated on the complete standard MNIST test partition ($N = 10,000$ unseen handwritten digits):

| Metric | Empirical Score | Definition |
| :--- | :---: | :--- |
| **Digit Recognition Accuracy** | **92.97%** | Proportion of correctly classified real handwritten digits on unseen test set |
| **Final Generator Loss** | **2.3348** | Adversarial fooling loss + conditional generation classification loss |
| **Final Discriminator Loss** | **0.6117** | Binary discrimination loss + multi-class recognition loss |
| **Training Dataset** | **60,000 images** | Complete standard MNIST training split ($28 \times 28$ grayscale) |
| **Testing Dataset** | **10,000 images** | Standard independent MNIST test split |

### 3.3 Training Dynamics Across Epochs (Tesla T4 GPU)

```text
Epoch 01/05 | G Loss: 2.4967 | D Loss: 0.7939 | Real Digit Recognition Acc: 78.92%
Epoch 02/05 | G Loss: 2.8203 | D Loss: 0.5576 | Real Digit Recognition Acc: 87.48%
Epoch 03/05 | G Loss: 2.6968 | D Loss: 0.6085 | Real Digit Recognition Acc: 88.90%
Epoch 04/05 | G Loss: 2.9514 | D Loss: 0.6135 | Real Digit Recognition Acc: 89.05%
Epoch 05/05 | G Loss: 2.3348 | D Loss: 0.6117 | Real Digit Recognition Acc: 90.08%
```

---

## 4. How to Run

### Google Colab Execution
1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Copy and paste the contents of [`gan_mnist_recognition.py`](gan_mnist_recognition.py) into a code cell.
4. Run the cell. The model will train on the full 60,000-image dataset in ~30 seconds, display real-time recognition metrics, and render the synthesized digit grid inline.

### Local Execution
```bash
python3 gan-mnist-digit-recognition/gan_mnist_recognition.py
```
