"""Convolutional Vision Transformer (CvT) for Image Classification on CIFAR-10
Architecture: Overlapping Convolutional Tokenizer -> Stacked Transformer Blocks with ConvFFN -> Global Pooling
Benchmark: CIFAR-10 (10 Classes, 32x32 RGB Images, 8x8 Spatial Feature Map)
"""
import os, sys, json, time, numpy as np

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, Subset
    import torchvision, torchvision.transforms as transforms; HAS_TORCH = True
except ImportError:
    if "google.colab" in sys.modules:
        import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "matplotlib"])
        import torch, torch.nn as nn, torch.optim as optim; from torch.utils.data import DataLoader, Subset
        import torchvision, torchvision.transforms as transforms; HAS_TORCH = True
    else: HAS_TORCH = False

try:
    import matplotlib.pyplot as plt; HAS_MATPLOTLIB = True
except ImportError: plt, HAS_MATPLOTLIB = None, False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
RESULTS_JSON = os.path.join(BASE_DIR, "cvt_classification_results.json")
PLOT_PNG = os.path.join(BASE_DIR, "cvt_classification_results.png")
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

if HAS_TORCH:
    class ConvTokenizer(nn.Module):
        """Overlapping multi-stage convolutional tokenizer providing spatial inductive bias and local hierarchy."""
        def __init__(self, in_ch=3, embed_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(embed_dim), nn.GELU()
            )
        def forward(self, x): return self.net(x)

    class ConvTransformerBlock(nn.Module):
        """Transformer block combining Multi-Head Self-Attention with a Depthwise Convolutional FFN (ConvFFN)."""
        def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=2, drop=0.1):
            super().__init__()
            self.ln1 = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=drop, batch_first=True)
            self.ln2 = nn.LayerNorm(embed_dim)
            hidden_dim = embed_dim * mlp_ratio
            self.fc1 = nn.Linear(embed_dim, hidden_dim)
            self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False)
            self.bn = nn.BatchNorm2d(hidden_dim); self.act = nn.GELU(); self.drop = nn.Dropout(drop)
            self.fc2 = nn.Linear(hidden_dim, embed_dim)

        def forward(self, x, H=8, W=8):
            norm_x = self.ln1(x)
            attn_out, _ = self.attn(norm_x, norm_x, norm_x); x = x + attn_out
            norm_x = self.ln2(x); B, N, C = norm_x.shape
            h = self.fc1(norm_x)
            h_2d = h.transpose(1, 2).view(B, -1, H, W)
            h_conv = self.drop(self.act(self.bn(self.dwconv(h_2d))))
            h_flat = h_conv.flatten(2).transpose(1, 2)
            x = x + self.drop(self.fc2(h_flat))
            return x

    class ConvolutionalVisionTransformer(nn.Module):
        """Convolutional Vision Transformer (CvT) combining conv tokenization, self-attention, and ConvFFN."""
        def __init__(self, in_ch=3, n_classes=10, embed_dim=128, depth=4, n_heads=4, mlp_r=2, drop=0.1):
            super().__init__()
            self.tokenizer = ConvTokenizer(in_ch=in_ch, embed_dim=embed_dim)
            self.blocks = nn.ModuleList([ConvTransformerBlock(embed_dim, n_heads, mlp_r, drop) for _ in range(depth)])
            self.ln_head = nn.LayerNorm(embed_dim); self.head = nn.Linear(embed_dim, n_classes)

        def forward(self, x):
            feat = self.tokenizer(x)  # (B, 128, 8, 8)
            B, C, H, W = feat.shape
            tokens = feat.flatten(2).transpose(1, 2)  # (B, 64, 128)
            for blk in self.blocks: tokens = blk(tokens, H, W)
            pooled = self.ln_head(tokens.mean(dim=1))  # Global average pooling over token dimension
            return self.head(pooled)

def get_dataloaders(batch_size=128, n_train=15000, n_test=3000):
    """Loads standardized CIFAR-10 splits with fast mirror and spatial augmentations."""
    tar_p = os.path.join("./data", "cifar-10-python.tar.gz")
    if os.path.exists(tar_p) and os.path.getsize(tar_p) < 170000000 and not os.path.exists("./data/cifar-10-batches-py"):
        try: os.remove(tar_p)
        except OSError: pass
    torchvision.datasets.cifar.CIFAR10.url = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
    tr_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    te_tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    raw_tr = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=tr_tf)
    raw_te = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=te_tf)
    tr_ds = Subset(raw_tr, range(min(n_train, len(raw_tr))))
    te_ds = Subset(raw_te, range(min(n_test, len(raw_te))))
    return (DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
            DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True))

def compute_metrics(y_true, y_pred, n_classes=10):
    """Computes categorical classification metrics and confusion matrix."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred): cm[t, p] += 1
    acc = float(np.trace(cm)) / float(np.sum(cm)) * 100.0
    precisions, recalls = [], []
    for c in range(n_classes):
        tp = cm[c, c]; fp = np.sum(cm[:, c]) - tp; fn = np.sum(cm[c, :]) - tp
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    p_macro = float(np.mean(precisions)) * 100.0; r_macro = float(np.mean(recalls)) * 100.0
    f1_macro = (2.0 * p_macro * r_macro) / (p_macro + r_macro) if (p_macro + r_macro) > 0 else 0.0
    return {"accuracy": round(acc, 2), "precision": round(p_macro, 2), "recall": round(r_macro, 2), "f1_score": round(f1_macro, 2), "cm": cm}

def train_cvt(epochs=6, batch_size=128, lr=1e-3, weight_decay=0.05):
    """Executes end-to-end training and test evaluation of Convolutional Vision Transformer."""
    if not HAS_TORCH:
        print("PyTorch environment required. Please run on Google Colab GPU."); return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Execution Device: {device} | Loading CIFAR-10 Benchmark...")
    tr_loader, te_loader = get_dataloaders(batch_size=batch_size)

    model = ConvolutionalVisionTransformer(in_ch=3, n_classes=10, embed_dim=128, depth=4, n_heads=4, mlp_r=2, drop=0.1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Convolutional ViT Initialized | Parameters: {n_params:,} | Conv Tokenizer (32x32 -> 8x8) | Layers: 4 | Heads: 4")
    print(f"Dataset Size: {len(tr_loader.dataset):,} Train | {len(te_loader.dataset):,} Test | Batch Size: {batch_size}\n" + "-" * 85)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        model.train(); tr_loss = 0.0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); optimizer.zero_grad()
            out = model(x); loss = criterion(out, y); loss.backward(); optimizer.step(); tr_loss += loss.item()
        scheduler.step(); tr_loss /= len(tr_loader)

        model.eval(); val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in te_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x); val_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item(); total += y.size(0)
        val_loss /= len(te_loader); val_acc = (correct / total) * 100.0
        history["train_loss"].append(round(tr_loss, 4)); history["val_loss"].append(round(val_loss, 4)); history["val_acc"].append(round(val_acc, 2))
        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    model.eval(); all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in te_loader:
            x = x.to(device, non_blocking=True)
            all_preds.extend(model(x).argmax(1).cpu().numpy()); all_targets.extend(y.numpy())
    test_metrics = compute_metrics(all_targets, all_preds)
    return history, test_metrics

def plot_cvt_results(history, test_metrics):
    """Renders dual panel dashboard: Loss/Accuracy dynamics and 10x10 Confusion Matrix."""
    if not HAS_MATPLOTLIB or plt is None or test_metrics is None: return
    try:
        fig = plt.figure(figsize=(15, 7)); gs = fig.add_gridspec(2, 2)
        ax_loss, ax_acc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        ax_cm = fig.add_subplot(gs[1, :]); eps = range(1, len(history["train_loss"]) + 1)

        ax_loss.plot(eps, history["train_loss"], "o-", label="Train Loss", color="#2563eb")
        ax_loss.plot(eps, history["val_loss"], "s--", label="Val Loss", color="#dc2626")
        ax_loss.set_title("Cross-Entropy Loss Trajectory"); ax_loss.set_xlabel("Epochs"); ax_loss.grid(True, linestyle=":", alpha=0.6); ax_loss.legend()

        ax_acc.plot(eps, history["val_acc"], "^-", label="Validation Accuracy (%)", color="#16a34a")
        ax_acc.set_title("Accuracy Trajectory (%)"); ax_acc.set_xlabel("Epochs"); ax_acc.grid(True, linestyle=":", alpha=0.6); ax_acc.legend()

        cm = test_metrics["cm"]
        im = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
        ax_cm.set_title("Convolutional ViT CIFAR-10 Confusion Matrix (N = 3,000)")
        ax_cm.set_xticks(range(10)); ax_cm.set_xticklabels(CLASSES, rotation=35, ha="right", fontsize=9)
        ax_cm.set_yticks(range(10)); ax_cm.set_yticklabels(CLASSES, fontsize=9)
        fig.colorbar(im, ax=ax_cm, fraction=0.02, pad=0.02)
        for i in range(10):
            for j in range(10):
                val = cm[i, j]
                ax_cm.text(j, i, str(val), ha="center", va="center", color="white" if val > np.max(cm)/2 else "black", fontsize=8)

        plt.tight_layout(); plt.savefig(PLOT_PNG, dpi=180)
        if "google.colab" in sys.modules or "ipykernel" in sys.modules: plt.show()
        plt.close()
    except Exception as e: print(f"Plot notification: {e}")

def main():
    print("-" * 85); print("Convolutional Vision Transformer (CvT) for Image Classification on CIFAR-10"); print("-" * 85)
    print("Tokenization: Overlapping Conv Tokenizer (32x32 -> 8x8) | Embedding Dim: 128")
    print("Transformer:  4 Layers | 4 Attention Heads | Depthwise ConvFFN | Global Average Pooling\n" + "-" * 85)
    history, test_metrics = train_cvt(epochs=6, batch_size=128)

    if history is not None:
        print("\n" + "=" * 85); print("CONVOLUTIONAL VIT EVALUATION ON UNSEEN CIFAR-10 TEST SET (N = 3,000)")
        print("=" * 85)
        print(f"Overall Test Accuracy:     {test_metrics['accuracy']:.2f} %")
        print(f"Macro-Averaged Precision:  {test_metrics['precision']:.2f} %")
        print(f"Macro-Averaged Recall:     {test_metrics['recall']:.2f} %")
        print(f"Macro-Averaged F1-Score:   {test_metrics['f1_score']:.2f} %")
        print(f"Final Training Loss:       {history['train_loss'][-1]:.4f}")
        print(f"Final Validation Loss:     {history['val_loss'][-1]:.4f}")
        print("=" * 85)

        plot_cvt_results(history, test_metrics)
        eval_dict = {"task": "Convolutional Vision Transformer Image Classification on CIFAR-10",
                     "architecture": "CvT (Conv Tokenizer 32->8, Dim 128, Depth 4, Heads 4, ConvFFN)",
                     "test_metrics": {"accuracy": test_metrics["accuracy"], "precision": test_metrics["precision"],
                                      "recall": test_metrics["recall"], "f1_score": test_metrics["f1_score"]},
                     "history": history, "confusion_matrix": test_metrics["cm"].tolist()}
        with open(RESULTS_JSON, "w", encoding="utf-8") as f: json.dump(eval_dict, f, indent=2)

if __name__ == "__main__": main()
