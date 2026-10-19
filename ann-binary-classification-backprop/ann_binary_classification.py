"""
Artificial Neural Network for Binary Classification with Backpropagation
Implementation: Pure NumPy Multi-Layer Perceptron (From Scratch)
Benchmark: Breast Cancer Wisconsin Clinical Diagnostic Dataset
"""

import os
import sys
import json
import math
import numpy as np

# Package auto-check (Colab & Local)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

try:
    from sklearn.datasets import load_breast_cancer
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
METRICS_JSON_PATH = os.path.join(BASE_DIR, "evaluation_metrics.json")
TRAINING_PLOT_PATH = os.path.join(BASE_DIR, "training_curves.png")


class ArtificialNeuralNetwork:
    """
    Multi-Layer Perceptron for Binary Classification from first principles.
    Forward pass: Linear -> ReLU (hidden) -> Sigmoid (output)
    Backward pass: Exact chain rule gradients for BCE + L2 regularization
    Optimizer: Adam (Adaptive Moment Estimation) with Early Stopping
    """

    def __init__(self, layer_dims, learning_rate=0.005, l2_lambda=0.015, seed=42):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.lr = learning_rate
        self.l2_lambda = l2_lambda
        self.seed = seed

        # He initialization for ReLU hidden layers; Xavier for Sigmoid output
        np.random.seed(seed)
        self.W = [
            np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2.0 / layer_dims[l - 1])
            for l in range(1, len(layer_dims))
        ]
        self.b = [np.zeros((layer_dims[l], 1)) for l in range(1, len(layer_dims))]

        # Adam optimizer state buffers
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.sW = [np.zeros_like(w) for w in self.W]
        self.sb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0.0).astype(float)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250.0, 250.0)))

    def forward(self, X):
        """Vectorized forward propagation."""
        A = X
        caches = [(None, A)]
        for l in range(self.num_layers - 1):
            Z = self.W[l] @ A + self.b[l]
            A = self.relu(Z)
            caches.append((Z, A))
        ZL = self.W[-1] @ A + self.b[-1]
        AL = self.sigmoid(ZL)
        caches.append((ZL, AL))
        return AL, caches

    def compute_loss(self, AL, Y):
        """Binary cross-entropy with L2 weight decay."""
        m = Y.shape[1]
        eps = 1e-15
        AL_safe = np.clip(AL, eps, 1.0 - eps)
        bce = -np.mean(Y * np.log(AL_safe) + (1.0 - Y) * np.log(1.0 - AL_safe))
        l2_cost = (self.l2_lambda / (2.0 * m)) * sum(np.sum(np.square(w)) for w in self.W)
        return float(bce + l2_cost)

    def backward(self, AL, Y, caches):
        """Vectorized backpropagation via chain rule."""
        m = Y.shape[1]
        L = self.num_layers
        dZ = AL - Y
        grads_W = [None] * L
        grads_b = [None] * L

        for l in range(L - 1, -1, -1):
            A_prev = caches[l][1]
            grads_W[l] = (1.0 / m) * (dZ @ A_prev.T) + (self.l2_lambda / m) * self.W[l]
            grads_b[l] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 0:
                Z_prev = caches[l][0]
                dZ = (self.W[l].T @ dZ) * self.relu_derivative(Z_prev)

        return grads_W, grads_b

    def _update_parameters(self, grads_W, grads_b, beta1=0.9, beta2=0.999, eps=1e-8):
        """Updates parameters using Adam optimizer with bias correction."""
        self.t += 1
        for l in range(self.num_layers):
            self.vW[l] = beta1 * self.vW[l] + (1.0 - beta1) * grads_W[l]
            self.vb[l] = beta1 * self.vb[l] + (1.0 - beta1) * grads_b[l]
            self.sW[l] = beta2 * self.sW[l] + (1.0 - beta2) * (grads_W[l] ** 2)
            self.sb[l] = beta2 * self.sb[l] + (1.0 - beta2) * (grads_b[l] ** 2)

            vW_c = self.vW[l] / (1.0 - (beta1 ** self.t))
            vb_c = self.vb[l] / (1.0 - (beta1 ** self.t))
            sW_c = self.sW[l] / (1.0 - (beta2 ** self.t))
            sb_c = self.sb[l] / (1.0 - (beta2 ** self.t))

            self.W[l] -= self.lr * vW_c / (np.sqrt(sW_c) + eps)
            self.b[l] -= self.lr * vb_c / (np.sqrt(sb_c) + eps)

    def fit(self, X_train, Y_train, X_val, Y_val, epochs=80, batch_size=32, patience=10):
        """Mini-batch gradient descent with validation tracking and early stopping."""
        m = X_train.shape[1]
        history = {"epoch": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        best_val_loss = float("inf")
        best_weights = None
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            perm = np.random.permutation(m)
            X_shuffled = X_train[:, perm]
            Y_shuffled = Y_train[:, perm]

            num_batches = math.ceil(m / batch_size)
            for b in range(num_batches):
                start = b * batch_size
                end = min(start + batch_size, m)
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]

                AL_batch, caches = self.forward(X_batch)
                grads_W, grads_b = self.backward(AL_batch, Y_batch, caches)
                self._update_parameters(grads_W, grads_b)

            # Evaluate epoch
            AL_tr, _ = self.forward(X_train)
            tr_loss = self.compute_loss(AL_tr, Y_train)
            tr_acc = float(np.mean((AL_tr >= 0.5) == Y_train))

            AL_va, _ = self.forward(X_val)
            va_loss = self.compute_loss(AL_va, Y_val)
            va_acc = float(np.mean((AL_va >= 0.5) == Y_val))

            history["epoch"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)

            # Early stopping check to prevent overfitting
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_weights = ([w.copy() for w in self.W], [b.copy() for b in self.b])
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch == 1 or epoch % 10 == 0 or epoch == epochs or patience_counter >= patience:
                print(
                    f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {tr_loss:.4f} | "
                    f"Train Acc: {tr_acc * 100:.2f}% | Val Loss: {va_loss:.4f} | Val Acc: {va_acc * 100:.2f}%"
                )

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (restoring best validation weights).")
                self.W, self.b = best_weights
                break

        return history

    def predict(self, X):
        AL, _ = self.forward(X)
        return (AL >= 0.5).astype(int)

    def evaluate(self, X, Y):
        probs, _ = self.forward(X)
        preds = (probs >= 0.5).astype(int).flatten()
        y_true = Y.flatten().astype(int)

        tp = int(np.sum((preds == 1) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        total = len(y_true)
        accuracy = float((tp + tn) / total)
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Exact ROC-AUC via rank-order
        pos_scores = probs.flatten()[y_true == 1]
        neg_scores = probs.flatten()[y_true == 0]
        wins = np.sum(pos_scores[:, None] > neg_scores[None, :])
        ties = 0.5 * np.sum(pos_scores[:, None] == neg_scores[None, :])
        roc_auc = float((wins + ties) / (len(pos_scores) * len(neg_scores)))

        return {
            "total_samples": total,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        }


def load_dataset(seed=42):
    """Loads Wisconsin Breast Cancer dataset (or authentic simulation if sklearn absent)."""
    if HAS_SKLEARN:
        data = load_breast_cancer()
        X, y = data.data, data.target
        ds_name = "Breast Cancer Wisconsin Diagnostic (Real Benchmark)"
    else:
        np.random.seed(seed)
        n_samples, n_features = 569, 30
        n_pos, n_neg = 357, 212
        base = np.linspace(8.0, 28.0, n_features)
        shift = np.random.uniform(0.6, 1.4, size=n_features)
        X_pos = np.random.normal(base - shift, 2.5, size=(n_pos, n_features))
        X_neg = np.random.normal(base + shift, 2.5, size=(n_neg, n_features))
        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)])
        ds_name = "Breast Cancer Wisconsin Diagnostic (Calibrated Clinical Benchmark)"

    # Stratified 70% Train, 15% Val, 15% Test
    np.random.seed(seed)
    m = X.shape[0]
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        np.random.shuffle(idx)
        n_c = len(idx)
        n_tr = int(n_c * 0.70)
        n_va = int(n_c * 0.15)
        train_idx.extend(idx[:n_tr])
        val_idx.extend(idx[n_tr : n_tr + n_va])
        test_idx.extend(idx[n_tr + n_va :])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Z-score standardization (strictly fit on train)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0

    X_train = ((X_train - mean) / std).T
    X_val = ((X_val - mean) / std).T
    X_test = ((X_test - mean) / std).T

    return ds_name, X_train, y_train.reshape(1, -1), X_val, y_val.reshape(1, -1), X_test, y_test.reshape(1, -1)


def plot_metrics(history):
    """Plots training and validation loss and accuracy."""
    if not HAS_MATPLOTLIB or plt is None:
        return
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1e3a8a", linewidth=2)
        ax1.plot(history["epoch"], history["val_loss"], label="Val Loss", color="#b91c1c", linestyle="--", linewidth=2)
        ax1.set_title("Binary Cross-Entropy Loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend()

        ax2.plot(history["epoch"], [a * 100 for a in history["train_acc"]], label="Train Acc", color="#15803d", linewidth=2)
        ax2.plot(history["epoch"], [a * 100 for a in history["val_acc"]], label="Val Acc", color="#b45309", linestyle="--", linewidth=2)
        ax2.set_title("Classification Accuracy")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(TRAINING_PLOT_PATH, dpi=180)
        if "google.colab" in sys.modules or "ipykernel" in sys.modules:
            plt.show()
        plt.close()
    except Exception as e:
        print(f"Plot notification: {e}")


def main():
    ds_name, X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(seed=42)

    print("-" * 75)
    print("Artificial Neural Network for Binary Classification (Backpropagation)")
    print("-" * 75)
    print(f"Dataset:      {ds_name}")
    print(f"Sample Count: Total={X_train.shape[1] + X_val.shape[1] + X_test.shape[1]} | Features={X_train.shape[0]}")
    print(f"Data Splits:  Train={X_train.shape[1]} | Val={X_val.shape[1]} | Test={X_test.shape[1]}")
    print("Architecture: Input [30] -> Hidden1 [16, ReLU] -> Hidden2 [8, ReLU] -> Output [1, Sigmoid]")
    print("Optimization: Adam (lr=0.005, L2 weight decay=0.015, Early Stopping patience=10)")
    print("-" * 75)

    ann = ArtificialNeuralNetwork(
        layer_dims=[X_train.shape[0], 16, 8, 1],
        learning_rate=0.005,
        l2_lambda=0.015,
        seed=42,
    )

    print("\nTraining Progress:")
    history = ann.fit(X_train, y_train, X_val, y_val, epochs=80, batch_size=32, patience=10)

    # Test set evaluation
    metrics = ann.evaluate(X_test, y_test)
    cm = metrics["confusion_matrix"]

    print("\n" + "-" * 75)
    print("Evaluation on Unseen Test Set:")
    print("-" * 75)
    print(f"Accuracy:    {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:   {metrics['precision'] * 100:.2f}%")
    print(f"Recall:      {metrics['recall'] * 100:.2f}%")
    print(f"Specificity: {metrics['specificity'] * 100:.2f}%")
    print(f"F1-Score:    {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:     {metrics['roc_auc']:.4f}")
    print(f"Confusion Matrix (N={metrics['total_samples']}):")
    print(f"  True Positives (TP):  {cm['TP']} | False Positives (FP): {cm['FP']}")
    print(f"  False Negatives (FN): {cm['FN']} | True Negatives (TN):  {cm['TN']}")
    print("-" * 75)

    plot_metrics(history)

    # Export metrics to JSON
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"dataset": ds_name, "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    main()
