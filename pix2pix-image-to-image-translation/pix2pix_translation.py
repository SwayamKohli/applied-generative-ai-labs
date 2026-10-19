"""Pix2Pix Conditional GAN for Image-to-Image Domain Translation (Sketch to Photo)
Architecture: U-Net Generator with Skip Connections & 70x70 PatchGAN Discriminator
Objective: Adversarial BCE Loss + L1 Pixel Reconstruction Loss (lambda = 100)
"""
import os, sys, json, time, math, numpy as np

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    if "google.colab" in sys.modules:
        import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "matplotlib", "pillow"])
        import torch, torch.nn as nn, torch.optim as optim; from torch.utils.data import Dataset, DataLoader
        import torchvision.transforms as transforms; HAS_TORCH = True
    else: HAS_TORCH = False

try:
    import matplotlib.pyplot as plt; HAS_MATPLOTLIB = True
except ImportError: plt, HAS_MATPLOTLIB = None, False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
RESULTS_JSON = os.path.join(BASE_DIR, "translation_evaluation_results.json")
PLOT_PNG = os.path.join(BASE_DIR, "translation_results.png")

if HAS_TORCH:
    class UNetGenerator(nn.Module):
        """U-Net Generator with encoder-decoder skip connections preserving spatial sketch boundaries."""
        def __init__(self, in_ch=3, out_ch=3):
            super().__init__()
            def down(i, o, bn=True):
                layers = [nn.Conv2d(i, o, 4, 2, 1, bias=False)]
                if bn: layers.append(nn.BatchNorm2d(o))
                layers.append(nn.LeakyReLU(0.2, inplace=True)); return nn.Sequential(*layers)

            def up(i, o, drop=0.0):
                layers = [nn.ConvTranspose2d(i, o, 4, 2, 1, bias=False), nn.BatchNorm2d(o)]
                if drop > 0: layers.append(nn.Dropout(drop))
                layers.append(nn.ReLU(inplace=True)); return nn.Sequential(*layers)

            self.d1, self.d2 = down(in_ch, 64, bn=False), down(64, 128)
            self.d3, self.d4 = down(128, 256), down(256, 512)
            self.d5 = down(512, 512, bn=False)

            self.u1 = up(512, 512, drop=0.5)
            self.u2, self.u3 = up(1024, 256), up(512, 128)
            self.u4 = up(256, 64)
            self.final = nn.Sequential(nn.ConvTranspose2d(128, out_ch, 4, 2, 1), nn.Tanh())

        def forward(self, x):
            e1 = self.d1(x); e2 = self.d2(e1); e3 = self.d3(e2); e4 = self.d4(e3); e5 = self.d5(e4)
            d1 = self.u1(e5); d2 = self.u2(torch.cat([d1, e4], 1))
            d3 = self.u3(torch.cat([d2, e3], 1)); d4 = self.u4(torch.cat([d3, e2], 1))
            return self.final(torch.cat([d4, e1], 1))

    class PatchGANDiscriminator(nn.Module):
        """Markovian 70x70 PatchGAN Discriminator penalizing high-frequency local structural errors."""
        def __init__(self, in_ch=6):
            super().__init__()
            def block(i, o, stride=2, bn=True):
                layers = [nn.Conv2d(i, o, 4, stride, 1, bias=False)]
                if bn: layers.append(nn.BatchNorm2d(o))
                layers.append(nn.LeakyReLU(0.2, inplace=True)); return layers

            self.model = nn.Sequential(
                *block(in_ch, 64, stride=2, bn=False), *block(64, 128, stride=2),
                *block(128, 256, stride=2), *block(256, 512, stride=1),
                nn.Conv2d(512, 1, 4, stride=1, padding=1), nn.Sigmoid()
            )
        def forward(self, a, b): return self.model(torch.cat([a, b], 1))

    class PairedTranslationDataset(Dataset):
        """Paired Sketch (Domain A) to Photo (Domain B) dataset loader and synthetic procedural generator."""
        def __init__(self, num_samples=600, img_size=128, seed=42):
            super().__init__(); np.random.seed(seed); self.samples = []
            yy, xx = np.mgrid[0:img_size, 0:img_size]
            for _ in range(num_samples):
                # Domain B: Textured Realistic Scene (color gradients + lighting)
                photo = np.zeros((3, img_size, img_size), dtype=np.float32)
                sketch = np.ones((3, img_size, img_size), dtype=np.float32)  # White canvas
                bg_col = np.random.uniform(0.2, 0.8, (3, 1, 1))
                photo[:] = bg_col + np.sin(xx / 16.0) * 0.1 + np.cos(yy / 16.0) * 0.1

                # Objects: Architectural facades / geometric buildings
                for _ in range(np.random.randint(2, 5)):
                    x0, y0 = np.random.randint(10, img_size - 40, size=2)
                    w, h = np.random.randint(20, 35, size=2)
                    col = np.random.uniform(0.1, 0.9, size=3)
                    # Color in Photo
                    for c in range(3): photo[c, y0:y0+h, x0:x0+w] = col[c]
                    # Draw Line Edge Contour in Sketch (Domain A)
                    sketch[:, y0:y0+2, x0:x0+w] = 0.0; sketch[:, y0+h-2:y0+h, x0:x0+w] = 0.0
                    sketch[:, y0:y0+h, x0:x0+2] = 0.0; sketch[:, y0:y0+h, x0+w-2:x0+w] = 0.0

                photo = (np.clip(photo, 0.0, 1.0) - 0.5) / 0.5  # Normalize to [-1, 1]
                sketch = (np.clip(sketch, 0.0, 1.0) - 0.5) / 0.5
                self.samples.append((torch.tensor(sketch), torch.tensor(photo)))

        def __len__(self): return len(self.samples)
        def __getitem__(self, idx): return self.samples[idx]

def compute_psnr(pred, target):
    """Computes Peak Signal-to-Noise Ratio (PSNR) in decibels."""
    mse = np.mean((pred - target) ** 2)
    return 10.0 * math.log10(4.0 / (mse + 1e-10)) if mse > 0 else 100.0

def train_pix2pix(epochs=10, batch_size=16, lr=0.0002, lambda_l1=100.0):
    """Executes full PyTorch training loop for Pix2Pix image-to-image translation."""
    if not HAS_TORCH:
        print("PyTorch environment required. Please run on Google Colab GPU."); return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Execution Device: {device} | Initializing Pix2Pix Paired Translation Dataset (128x128)...")

    full_ds = PairedTranslationDataset(num_samples=500, img_size=128)
    tr_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    te_loader = DataLoader(full_ds, batch_size=4, shuffle=False)

    G = UNetGenerator().to(device); D = PatchGANDiscriminator().to(device)
    criterion_adv, criterion_l1 = nn.BCELoss(), nn.L1Loss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    history = {"d_loss": [], "g_loss": [], "g_adv": [], "g_l1": []}
    print(f"Beginning Pix2Pix Training ({epochs} Epochs, Batch Size {batch_size}, L1 Lambda {lambda_l1})...\n")

    for ep in range(1, epochs + 1):
        G.train(); D.train(); d_running, g_running, g_adv_run, g_l1_run = 0.0, 0.0, 0.0, 0.0
        for real_a, real_b in tr_loader:
            real_a, real_b = real_a.to(device), real_b.to(device)
            fake_b = G(real_a)

            # Discriminator update
            opt_D.zero_grad()
            pred_real = D(real_a, real_b)
            loss_d_real = criterion_adv(pred_real, torch.ones_like(pred_real))
            pred_fake = D(real_a, fake_b.detach())
            loss_d_fake = criterion_adv(pred_fake, torch.zeros_like(pred_fake))
            loss_d = (loss_d_real + loss_d_fake) * 0.5; loss_d.backward(); opt_D.step()

            # Generator update
            opt_G.zero_grad()
            pred_fake = D(real_a, fake_b)
            loss_g_adv = criterion_adv(pred_fake, torch.ones_like(pred_fake))
            loss_g_l1 = criterion_l1(fake_b, real_b) * lambda_l1
            loss_g = loss_g_adv + loss_g_l1; loss_g.backward(); opt_G.step()

            d_running += loss_d.item(); g_running += loss_g.item()
            g_adv_run += loss_g_adv.item(); g_l1_run += (loss_g_l1.item() / lambda_l1)

        n_batches = len(tr_loader)
        d_mean, g_mean = d_running / n_batches, g_running / n_batches
        g_adv_m, g_l1_m = g_adv_run / n_batches, g_l1_run / n_batches
        history["d_loss"].append(round(d_mean, 4)); history["g_loss"].append(round(g_mean, 4))
        history["g_adv"].append(round(g_adv_m, 4)); history["g_l1"].append(round(g_l1_m, 4))

        print(f"Epoch {ep:02d}/{epochs:02d} | D Loss: {d_mean:.4f} | G Total: {g_mean:.4f} (Adv: {g_adv_m:.4f}, L1: {g_l1_m:.4f})")

    # Evaluate on test samples
    G.eval()
    test_a, test_b = next(iter(te_loader))
    with torch.no_grad(): gen_b = G(test_a.to(device)).cpu()
    l1_err = float(criterion_l1(gen_b, test_b).item())
    psnr_val = compute_psnr(gen_b.numpy(), test_b.numpy())
    test_metrics = {"mean_l1_pixel_error": round(l1_err, 4), "psnr_db": round(psnr_val, 2)}
    return history, test_metrics, (test_a.numpy(), test_b.numpy(), gen_b.numpy())

def plot_translation_results(history, sample_bundle):
    """Renders 4-column qualitative translation grid alongside loss convergence curves."""
    if not HAS_MATPLOTLIB or plt is None or sample_bundle is None: return
    try:
        test_a, test_b, gen_b = sample_bundle
        fig = plt.figure(figsize=(15, 6)); gs = fig.add_gridspec(2, 4)
        ax_loss = fig.add_subplot(gs[0, :2]); ax_l1 = fig.add_subplot(gs[0, 2:])
        eps = range(1, len(history["d_loss"]) + 1)

        ax_loss.plot(eps, history["d_loss"], "s--", label="D PatchGAN Loss", color="#dc2626")
        ax_loss.plot(eps, history["g_adv"], "o-", label="G Adversarial Loss", color="#2563eb")
        ax_loss.set_title("Adversarial Minimax Loss"); ax_loss.set_xlabel("Epochs"); ax_loss.grid(True, linestyle=":", alpha=0.6); ax_loss.legend()

        ax_l1.plot(eps, history["g_l1"], "^-", label="L1 Reconstruction Loss", color="#16a34a")
        ax_l1.set_title("L1 Pixel-Level Structural Error"); ax_l1.set_xlabel("Epochs"); ax_l1.grid(True, linestyle=":", alpha=0.6); ax_l1.legend()

        # Qualitative 4-sample comparison: [Input Sketch | Translated Photo | Ground Truth]
        for i in range(4):
            ax_in = fig.add_subplot(gs[1, i])
            sk = np.transpose(test_a[i] * 0.5 + 0.5, (1, 2, 0))
            gt = np.transpose(test_b[i] * 0.5 + 0.5, (1, 2, 0))
            pr = np.transpose(np.clip(gen_b[i] * 0.5 + 0.5, 0, 1), (1, 2, 0))
            composite = np.concatenate([sk, pr, gt], axis=1)
            ax_in.imshow(composite); ax_in.axis("off")
            ax_in.set_title(f"Sample #{i+1}: Sketch | Translated | Real Photo", fontsize=8)

        plt.tight_layout(); plt.savefig(PLOT_PNG, dpi=180)
        if "google.colab" in sys.modules or "ipykernel" in sys.modules: plt.show()
        plt.close()
    except Exception as e: print(f"Plot notification: {e}")

def main():
    print("-" * 85); print("Pix2Pix Conditional GAN for Image-to-Image Domain Translation (Sketch to Photo)"); print("-" * 85)
    print("Architecture: U-Net Generator with Skip Connections & 70x70 PatchGAN Discriminator")
    print("Objective:    L_cGAN(G, D) + lambda * L1(G)  [lambda = 100]")
    print("Resolution:   128x128x3 Paired Domain A (Sketch) & Domain B (Photo)\n" + "-" * 85)
    history, test_metrics, samples = train_pix2pix(epochs=10, batch_size=16)

    if history is not None:
        print("\n" + "=" * 85); print("EVALUATION OF PIX2PIX TRANSLATION ON TEST SAMPLES")
        print("=" * 85)
        print(f"Mean L1 Pixel Error (MAE): {test_metrics['mean_l1_pixel_error']:.4f}")
        print(f"Peak Signal-to-Noise Ratio: {test_metrics['psnr_db']:.2f} dB")
        print(f"Final Discriminator Loss:   {history['d_loss'][-1]:.4f}")
        print(f"Final Generator Total Loss: {history['g_loss'][-1]:.4f}")
        print("=" * 85)

        plot_translation_results(history, samples)
        with open(RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump({"task": "Pix2Pix Image-to-Image Domain Translation", "metrics": test_metrics, "history": history}, f, indent=2)

if __name__ == "__main__": main()
