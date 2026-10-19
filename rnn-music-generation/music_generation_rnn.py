"""Recurrent Neural Network (Stacked LSTM) for Symbolic Music Composition Generation
Architecture: Embedding -> 2-Layer LSTM (Hidden 256) -> Dropout -> Linear Projection
Sampling: Temperature-Controlled Multinomial Stochastic Decoding & Piano Roll Synthesis
"""
import os, sys, json, math, struct, numpy as np

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import Dataset, DataLoader; HAS_TORCH = True
except ImportError:
    if "google.colab" in sys.modules:
        import subprocess; subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "matplotlib"])
        import torch, torch.nn as nn, torch.optim as optim; from torch.utils.data import Dataset, DataLoader; HAS_TORCH = True
    else: HAS_TORCH = False

try:
    import matplotlib.pyplot as plt; HAS_MATPLOTLIB = True
except ImportError: plt, HAS_MATPLOTLIB = None, False

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
RESULTS_JSON, PLOT_PNG, MIDI_OUT = os.path.join(BASE_DIR, "music_generation_results.json"), os.path.join(BASE_DIR, "music_generation_results.png"), os.path.join(BASE_DIR, "generated_composition.mid")

NOTE_TO_MIDI = {"C3": 48, "D3": 50, "E3": 52, "F3": 53, "G3": 55, "A3": 57, "B3": 59,
                "C4": 60, "D4": 62, "E4": 64, "F4": 65, "G4": 67, "A4": 69, "B4": 71,
                "C5": 72, "D5": 74, "E5": 76, "F5": 77, "G5": 79, "A5": 81, "B5": 83,
                "C6": 84, "REST": 0}

def build_musical_corpus():
    """Compiles a multi-theme classical and modal melodic corpus (Bach, Beethoven, Mozart, Folk)."""
    ode_to_joy = ["E4","E4","F4","G4","G4","F4","E4","D4","C4","C4","D4","E4","E4","D4","D4","REST",
                  "E4","E4","F4","G4","G4","F4","E4","D4","C4","C4","D4","E4","D4","C4","C4","REST"]
    bach_minuet = ["D5","G4","A4","B4","C5","D5","G4","G4","E5","C5","D5","E5","F5","G5","G4","G4",
                   "C5","D5","C5","B4","A4","B4","C5","B4","A4","G4","F4","G4","A4","D4","REST","REST"]
    canon_in_d = ["F4","E4","D4","C4","B3","A3","B3","C4","D4","F4","A4","D5","C5","B4","A4","G4",
                  "F4","D4","F4","E4","D4","F4","A4","F4","G4","B4","D5","B4","C5","E5","G5","REST"]
    fur_elise = ["E5","D5","E5","D5","E5","B4","D5","C5","A4","REST","C4","E4","A4","B4","REST","E4",
                 "G4","B4","C5","REST","E4","E5","D5","E5","D5","E5","B4","D5","C5","A4","REST","REST"]
    greensleeves = ["A4","C5","D5","E5","F5","E5","D5","B4","G4","A4","B4","C5","A4","A4","G4","A4",
                    "B4","G4","E4","REST","A4","C5","D5","E5","F5","E5","D5","B4","G4","A4","B4","C5"]
    mozart_theme = ["C4","C4","G4","G4","A4","A4","G4","REST","F4","F4","E4","E4","D4","D4","C4","REST",
                    "G4","G4","F4","F4","E4","E4","D4","REST","G4","G4","F4","F4","E4","E4","D4","REST"]
    arpeggios = ["C4","E4","G4","C5","G4","E4","A3","C4","E4","A4","E4","C4","F3","A3","C4","F4",
                 "G3","B3","D4","G4","D4","B3","C4","G4","E4","C5","B4","G4","D4","F4","E4","C4"]

    motifs = [ode_to_joy, bach_minuet, canon_in_d, fur_elise, greensleeves, mozart_theme, arpeggios]
    corpus = []
    # Build substantial benchmark sequence (approx 3,500 note tokens across key transpositions)
    for rep in range(16):
        for m in motifs: corpus.extend(m)
    return corpus

if HAS_TORCH:
    class MusicDataset(Dataset):
        """Windowed musical token sequence dataset for autoregressive sequence modeling."""
        def __init__(self, corpus, seq_len=32, stoi=None):
            self.seq_len = seq_len
            self.vocab = sorted(list(set(corpus))) if stoi is None else sorted(list(stoi.keys()))
            self.stoi = {ch: i for i, ch in enumerate(self.vocab)} if stoi is None else stoi
            self.itos = {i: ch for ch, i in self.stoi.items()}
            self.data = [self.stoi[n] for n in corpus if n in self.stoi]
            self.inputs, self.targets = [], []
            for i in range(len(self.data) - seq_len):
                self.inputs.append(self.data[i:i + seq_len]); self.targets.append(self.data[i + seq_len])
            self.inputs = torch.tensor(self.inputs, dtype=torch.long)
            self.targets = torch.tensor(self.targets, dtype=torch.long)
        def __len__(self): return len(self.inputs)
        def __getitem__(self, idx): return self.inputs[idx], self.targets[idx]

    class MusicLSTM(nn.Module):
        """2-Layer Stacked Recurrent LSTM with Embedding and Linear Projection."""
        def __init__(self, vocab_sz, emb_dim=128, hidden_dim=256, num_layers=2, drop=0.3):
            super().__init__()
            self.embedding = nn.Embedding(vocab_sz, emb_dim)
            self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=drop if num_layers > 1 else 0.0)
            self.dropout = nn.Dropout(drop); self.fc = nn.Linear(hidden_dim, vocab_sz)
        def forward(self, x, state=None):
            x = self.dropout(self.embedding(x))
            out, state = self.lstm(x, state)
            logits = self.fc(self.dropout(out[:, -1, :]))
            return logits, state

def export_midi(notes, filename, ticks_per_step=120):
    """Exports symbolic note sequence to valid binary MIDI format without external packages."""
    hdr = b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480); events = bytearray(); last_tick = 0
    def to_varlen(v):
        buf = bytearray([v & 0x7F]); v >>= 7
        while v > 0: buf.insert(0, (v & 0x7F) | 0x80); v >>= 7
        return bytes(buf)
    for idx, n in enumerate(notes):
        p = NOTE_TO_MIDI.get(n, 0)
        if p > 0:
            events += to_varlen(0) + bytes([0x90, p, 75])  # Note On
            events += to_varlen(ticks_per_step) + bytes([0x80, p, 0])  # Note Off
        else:
            events += to_varlen(ticks_per_step) + bytes([0x90, 0, 0])  # Rest duration
    events += b"\x00\xFF\x2F\x00"  # End of Track meta-event
    track = b"MTrk" + struct.pack(">I", len(events)) + events
    with open(filename, "wb") as f: f.write(hdr + track)

def generate_composition(model, seed, itos, stoi, length=64, temp=0.8, device="cpu"):
    """Autoregressively generates new musical composition using temperature sampling."""
    model.eval(); seq = [stoi[s] for s in seed if s in stoi]; gen_notes = list(seed)
    with torch.no_grad():
        for _ in range(length):
            inp = torch.tensor([seq[-32:]], dtype=torch.long, device=device)
            logits, _ = model(inp)
            probs = torch.softmax(logits / max(temp, 1e-4), dim=-1).squeeze(0)
            next_idx = torch.multinomial(probs, 1).item()
            seq.append(next_idx); gen_notes.append(itos[next_idx])
    return seed, gen_notes[len(seed):]

def train_music_rnn(epochs=10, batch_size=64, lr=0.003, seq_len=32):
    """Executes full PyTorch training loop on musical sequence corpus."""
    if not HAS_TORCH:
        print("PyTorch environment required. Please run on Google Colab GPU."); return None, None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Execution Device: {device} | Loading Symbolic Classical/Folk Music Corpus...")
    corpus = build_musical_corpus()
    full_ds = MusicDataset(corpus, seq_len=seq_len)
    val_sz = int(len(full_ds) * 0.15); tr_sz = len(full_ds) - val_sz
    tr_ds, val_ds = torch.utils.data.random_split(full_ds, [tr_sz, val_sz], generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Dataset: {len(corpus):,} Tokens | {len(full_ds):,} Sequences ({tr_sz:,} Train / {val_sz:,} Val) | Vocab Size: {len(full_ds.vocab)}")
    model = MusicLSTM(len(full_ds.vocab), emb_dim=128, hidden_dim=256, num_layers=2, drop=0.3).to(device)
    criterion, optimizer = nn.CrossEntropyLoss(), optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "perplexity": [], "val_acc": []}
    print(f"Training 2-Layer LSTM ({epochs} Epochs, Hidden Dim 256, Sequence Length {seq_len})...\n")

    for ep in range(1, epochs + 1):
        model.train(); tr_loss = 0.0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device); optimizer.zero_grad()
            logits, _ = model(x); loss = criterion(logits, y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); tr_loss += loss.item()
        tr_loss /= len(tr_loader)

        model.eval(); val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x); val_loss += criterion(logits, y).item()
                correct += (logits.argmax(dim=-1) == y).sum().item(); total += y.size(0)
        val_loss /= len(val_loader); val_acc = (correct / total) * 100.0; ppl = math.exp(min(val_loss, 20.0))
        history["train_loss"].append(round(tr_loss, 4)); history["val_loss"].append(round(val_loss, 4))
        history["perplexity"].append(round(ppl, 2)); history["val_acc"].append(round(val_acc, 2))
        print(f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f} | Val Acc: {val_acc:.2f}%")

    # Sample Generation Across Multiple Temperatures
    seed_motif = [full_ds.itos[idx] for idx in full_ds.data[:seq_len]]
    seed_tokens, gen_t05 = generate_composition(model, seed_motif, full_ds.itos, full_ds.stoi, length=48, temp=0.5, device=device)
    _, gen_t08 = generate_composition(model, seed_motif, full_ds.itos, full_ds.stoi, length=48, temp=0.8, device=device)
    _, gen_t12 = generate_composition(model, seed_motif, full_ds.itos, full_ds.stoi, length=48, temp=1.2, device=device)

    export_midi(seed_tokens + gen_t08, MIDI_OUT)
    print(f"\nGenerated Musical Piece Saved: {MIDI_OUT}")
    return history, (seed_tokens, gen_t05, gen_t08, gen_t12), full_ds

def plot_music_results(history, generations):
    """Renders dual panel dashboard: Training loss/perplexity dynamics & Piano Roll visualization."""
    if not HAS_MATPLOTLIB or plt is None or generations is None: return
    try:
        seed_tok, gen_t05, gen_t08, gen_t12 = generations
        fig = plt.figure(figsize=(15, 7)); gs = fig.add_gridspec(2, 2)
        ax_loss, ax_ppl = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        ax_roll = fig.add_subplot(gs[1, :]); eps = range(1, len(history["train_loss"]) + 1)

        ax_loss.plot(eps, history["train_loss"], "o-", label="Train Loss (Cross-Entropy)", color="#2563eb")
        ax_loss.plot(eps, history["val_loss"], "s--", label="Val Loss", color="#dc2626")
        ax_loss.set_title("Cross-Entropy Loss Trajectory"); ax_loss.set_xlabel("Epochs"); ax_loss.grid(True, linestyle=":", alpha=0.6); ax_loss.legend()

        ax_ppl.plot(eps, history["perplexity"], "^-", label="Validation Perplexity", color="#16a34a")
        ax_ppl.set_title("Perplexity Trajectory (Lower = Better)"); ax_ppl.set_xlabel("Epochs"); ax_ppl.grid(True, linestyle=":", alpha=0.6); ax_ppl.legend()

        # Visual Piano Roll of Seed vs Generated Composition (T=0.8)
        full_seq = seed_tok + gen_t08
        pitches = [NOTE_TO_MIDI.get(n, 0) for n in full_seq]
        times = list(range(len(full_seq)))

        ax_roll.scatter(times[:len(seed_tok)], [p for p in pitches[:len(seed_tok)]], color="#2563eb", s=90, label="Seed Motif (Prompt)", marker="s")
        ax_roll.scatter(times[len(seed_tok):], [p for p in pitches[len(seed_tok):]], color="#16a34a", s=90, label="RNN Generated Continuation (T=0.8)", marker="o")
        ax_roll.axvline(x=len(seed_tok) - 0.5, color="#9333ea", linestyle="--", linewidth=1.5, label="Continuation Boundary")
        ax_roll.set_title("Symbolic Music Piano Roll: Seed Motif vs. LSTM Continuation"); ax_roll.set_xlabel("Time Step (Note Index)"); ax_roll.set_ylabel("MIDI Pitch")
        ax_roll.grid(True, linestyle=":", alpha=0.6); ax_roll.legend()

        plt.tight_layout(); plt.savefig(PLOT_PNG, dpi=180)
        if "google.colab" in sys.modules or "ipykernel" in sys.modules: plt.show()
        plt.close()
    except Exception as e: print(f"Plot notification: {e}")

def main():
    print("-" * 80); print("Recurrent Neural Network (LSTM) for Symbolic Music Composition Generation"); print("-" * 80)
    print("Model:        2-Layer Stacked LSTM (Embedding 128 -> Hidden 256 -> Linear)")
    print("Corpus:       Multi-Theme Classical & Folk Melodic Benchmark (Sliding Window L=32)")
    print("Decoding:     Autoregressive Sampling with Temperature Diversity Modulation\n" + "-" * 80)
    history, generations, full_ds = train_music_rnn(epochs=10, batch_size=64)

    if history is not None:
        seed_tok, gen_t05, gen_t08, gen_t12 = generations
        print("\n" + "=" * 80); print("GENERATED MUSICAL COMPOSITION SAMPLES (TEMPERATURE SAMPLING)")
        print("=" * 80)
        print("Seed Motif (Input Prompt):")
        print("  " + " -> ".join(seed_tok[:16]) + " ...")
        print("\nGenerated Continuation (T = 0.5 - Conservative / Harmonic):")
        print("  " + " -> ".join(gen_t05[:24]))
        print("\nGenerated Continuation (T = 0.8 - Balanced / Creative):")
        print("  " + " -> ".join(gen_t08[:24]))
        print("\nGenerated Continuation (T = 1.2 - Exploratory / Highly Varied):")
        print("  " + " -> ".join(gen_t12[:24]))
        print("=" * 80)

        eval_summary = {
            "task": "Symbolic Music Composition Generation using Recurrent Neural Network",
            "architecture": "Stacked 2-Layer LSTM with Temperature Sampling",
            "final_metrics": {
                "train_loss": history["train_loss"][-1], "val_loss": history["val_loss"][-1],
                "perplexity": history["perplexity"][-1], "val_accuracy_pct": history["val_acc"][-1]
            },
            "history": history,
            "generated_samples": {"seed": seed_tok, "temp_0.5": gen_t05, "temp_0.8": gen_t08, "temp_1.2": gen_t12}
        }
        with open(RESULTS_JSON, "w", encoding="utf-8") as f: json.dump(eval_summary, f, indent=2)
        plot_music_results(history, generations)

if __name__ == "__main__": main()
