"""
Artificial Neural Network for Multi-Class Classification with Backpropagation
Implementation: Pure NumPy Multi-Layer Perceptron (From Scratch)
Benchmark: Optical Recognition of Handwritten Digits (1,797 samples, 64 features, 10 classes)
"""

import os
import sys
import json
import math
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

try:
    from sklearn.datasets import load_digits
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
METRICS_JSON_PATH = os.path.join(BASE_DIR, "evaluation_metrics.json")
TRAINING_PLOT_PATH = os.path.join(BASE_DIR, "training_curves.png")


class MultiClassNeuralNetwork:
    """
    Multi-Layer Perceptron for Multi-Class Classification from first principles.
    Forward pass: Linear -> ReLU (hidden layers) -> Softmax (output layer)
    Backward pass: Vectorized chain rule for Categorical Cross-Entropy + L2 regularization
    Optimizer: Adam (Adaptive Moment Estimation) with Early Stopping
    """

    def __init__(self, layer_dims, learning_rate=0.005, l2_lambda=0.01, seed=42):
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        self.lr = learning_rate
        self.l2_lambda = l2_lambda
        self.seed = seed

        # He initialization for ReLU hidden layers; Xavier for Softmax output
        np.random.seed(seed)
        self.W = [
            np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2.0 / layer_dims[l - 1])
            for l in range(1, len(layer_dims))
        ]
        self.b = [np.zeros((layer_dims[l], 1)) for l in range(1, len(layer_dims))]

        # Adam state buffers
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
    def softmax(z):
        # Subtract max for numerical stability against float overflow
        z_shift = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z_shift)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    def forward(self, X):
        """Vectorized forward propagation."""
        A = X
        caches = [(None, A)]
        for l in range(self.num_layers - 1):
            Z = self.W[l] @ A + self.b[l]
            A = self.relu(Z)
            caches.append((Z, A))
        ZL = self.W[-1] @ A + self.b[-1]
        AL = self.softmax(ZL)
        caches.append((ZL, AL))
        return AL, caches

    def compute_loss(self, AL, Y):
        """Categorical Cross-Entropy with L2 weight decay."""
        m = Y.shape[1]
        eps = 1e-15
        AL_safe = np.clip(AL, eps, 1.0 - eps)
        # Sum over classes, mean over mini-batch
        cce = -np.mean(np.sum(Y * np.log(AL_safe), axis=0))
        l2_cost = (self.l2_lambda / (2.0 * m)) * sum(np.sum(np.square(w)) for w in self.W)
        return float(cce + l2_cost)

    def backward(self, AL, Y, caches):
        """Vectorized backpropagation via chain rule."""
        m = Y.shape[1]
        L = self.num_layers
        dZ = AL - Y  # Exact analytical derivative for Softmax + Cross-Entropy
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

    def fit(self, X_train, Y_train, y_train_labels, X_val, Y_val, y_val_labels, epochs=80, batch_size=32, patience=10):
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
            tr_acc = float(np.mean(np.argmax(AL_tr, axis=0) == y_train_labels))

            AL_va, _ = self.forward(X_val)
            va_loss = self.compute_loss(AL_va, Y_val)
            va_acc = float(np.mean(np.argmax(AL_va, axis=0) == y_val_labels))

            history["epoch"].append(epoch)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)

            # Early stopping check
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
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=0)

    def evaluate(self, X, y_true):
        probs, _ = self.forward(X)
        preds = np.argmax(probs, axis=0)
        num_classes = self.layer_dims[-1]
        total = len(y_true)

        accuracy = float(np.mean(preds == y_true))

        # Confusion matrix (K x K)
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for t, p in zip(y_true, preds):
            cm[t, p] += 1

        # Per-class metrics
        precisions = []
        recalls = []
        f1s = []
        for c in range(num_classes):
            tp = cm[c, c]
            fp = np.sum(cm[:, c]) - tp
            fn = np.sum(cm[c, :]) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1s))

        return {
            "total_samples": total,
            "num_classes": num_classes,
            "accuracy": round(accuracy, 4),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1_score": round(macro_f1, 4),
            "confusion_matrix": cm.tolist(),
        }


def one_hot_encode(labels, num_classes):
    """Converts 1D label array to (num_classes, m) one-hot binary matrix."""
    Y = np.zeros((num_classes, len(labels)))
    Y[labels, np.arange(len(labels))] = 1.0
    return Y


def load_dataset(seed=42):
    """Loads Handwritten Digits benchmark (or calibrated simulation if sklearn absent)."""
    if HAS_SKLEARN:
        data = load_digits()
        X, y = data.data, data.target
        ds_name = "Optical Recognition of Handwritten Digits (Real Benchmark)"
    else:
        np.random.seed(seed)
        n_samples, n_features, n_classes = 1797, 64, 10
        base_patterns = np.random.uniform(0.0, 8.0, size=(n_classes, n_features))
        base_patterns[8] = 0.6 * base_patterns[8] + 0.4 * base_patterns[3]
        base_patterns[7] = 0.7 * base_patterns[7] + 0.3 * base_patterns[1]
        y = np.random.randint(0, n_classes, size=n_samples)
        noise = np.random.normal(0.0, 3.5, size=(n_samples, n_features))
        X = np.clip(base_patterns[y] + noise, 0.0, 16.0)
        ds_name = "Optical Recognition of Handwritten Digits (Calibrated Simulation)"

    num_classes = len(np.unique(y))

    # Stratified 70% Train, 15% Val, 15% Test
    np.random.seed(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in range(num_classes):
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

    # Z-score standardization (fit strictly on train)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0.0] = 1.0

    X_train = ((X_train - mean) / std).T
    X_val = ((X_val - mean) / std).T
    X_test = ((X_test - mean) / std).T

    Y_train = one_hot_encode(y_train, num_classes)
    Y_val = one_hot_encode(y_val, num_classes)
    Y_test = one_hot_encode(y_test, num_classes)

    return ds_name, num_classes, X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test


def plot_metrics(history):
    """Plots training and validation loss and accuracy curves."""
    if not HAS_MATPLOTLIB or plt is None:
        return
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
        ax1.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#1e3a8a", linewidth=2)
        ax1.plot(history["epoch"], history["val_loss"], label="Val Loss", color="#b91c1c", linestyle="--", linewidth=2)
        ax1.set_title("Categorical Cross-Entropy Loss")
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
    ds_name, num_classes, X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = load_dataset(seed=42)

    print("-" * 75)
    print("Artificial Neural Network for Multi-Class Classification (Backpropagation)")
    print("-" * 75)
    print(f"Dataset:      {ds_name}")
    print(f"Sample Count: Total={X_train.shape[1] + X_val.shape[1] + X_test.shape[1]} | Features={X_train.shape[0]} | Classes={num_classes}")
    print(f"Data Splits:  Train={X_train.shape[1]} | Val={X_val.shape[1]} | Test={X_test.shape[1]}")
    print(f"Architecture: Input [{X_train.shape[0]}] -> Hidden1 [32, ReLU] -> Hidden2 [16, ReLU] -> Output [{num_classes}, Softmax]")
    print("Optimization: Adam (lr=0.005, L2 weight decay=0.01, Early Stopping patience=10)")
    print("-" * 75)

    ann = MultiClassNeuralNetwork(
        layer_dims=[X_train.shape[0], 32, 16, num_classes],
        learning_rate=0.005,
        l2_lambda=0.01,
        seed=42,
    )

    print("\nTraining Progress:")
    history = ann.fit(X_train, Y_train, y_train, X_val, Y_val, y_val, epochs=80, batch_size=32, patience=10)

    # Test set evaluation
    metrics = ann.evaluate(X_test, y_test)
    cm = np.array(metrics["confusion_matrix"])

    print("\n" + "-" * 75)
    print("Evaluation on Unseen Test Set:")
    print("-" * 75)
    print(f"Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:   {metrics['macro_precision'] * 100:.2f}%")
    print(f"Macro Recall:      {metrics['macro_recall'] * 100:.2f}%")
    print(f"Macro F1-Score:    {metrics['macro_f1_score']:.4f}")
    print(f"\nConfusion Matrix (Rows: Actual, Cols: Predicted, N={metrics['total_samples']}):")
    # Print formatted matrix rows
    header = "     " + " ".join([f"[{c}]" for c in range(num_classes)])
    print(header)
    for r in range(num_classes):
        row_str = f"[{r}]  " + "  ".join([f"{cm[r, c]:3d}" for c in range(num_classes)])
        print(row_str)
    print("-" * 75)

    plot_metrics(history)

    # Export metrics to JSON
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"dataset": ds_name, "metrics": metrics}, f, indent=2)


if __name__ == "__main__":
    main()
