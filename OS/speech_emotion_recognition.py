# ============================================================
# Speech Emotion Recognition (SER) — OS-Aware Implementation
# ITCS225 Principles of Operating Systems — Final Project
# Faculty of ICT, Mahidol University
# ============================================================
# Team: GoodJob Sec3
#   Mr. Traiwit Channgam       6788015
#   Ms. Nutchada Doungjit      6788020
#   Ms. Tinna Mongkolsamoke    6788063
#   Mr. Witthayut Phicharanan  6788086
#   Mr. Woraneti Phicharanan   6788087
#   Mr. Jarupat Sakpichaiskul  6788105
#   Ms. Nattanita Engkagul     6788239
# ============================================================
#
# INSTALL DEPENDENCIES (run once in terminal):
#   pip install kagglehub librosa scikit-learn scikit-image
#               torch torchvision matplotlib seaborn tqdm psutil
#
# NOTE: TensorFlow does NOT support Python 3.13+.
#       This project uses PyTorch, which supports Python 3.14.
#
# OS CONCEPTS DEMONSTRATED
# ──────────────────────────────────────────────────────────────
#  1. MULTIPROCESSING   — parallel .wav→spectrogram via Pool
#  2. SYNCHRONISATION   — threading.Lock() mutex on shared counter
#  3. CPU SCHEDULING    — os.nice(); pool sized to cpu_count()
#  4. MEMORY MANAGEMENT — mmap zero-copy read + memory tracking
#  5. SYSTEM CALLS      — os.open/read/fstat/stat/close (POSIX)
#  6. I/O MANAGEMENT    — buffered vs unbuffered read comparison
#  7. DESIGN TRADE-OFFS — shallow vs deep CNN + single vs multi-process
# ──────────────────────────────────────────────────────────────

# ── Standard-library imports ──────────────────────────────────
import os, sys, time, mmap, threading, multiprocessing
from multiprocessing import Pool
from pathlib import Path

# resource module is Linux/macOS only — not available on Windows
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False

# ── Third-party imports ───────────────────────────────────────
import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage.transform import resize as sk_resize

# PyTorch — supports Python 3.14 on Windows
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Constants ─────────────────────────────────────────────────
EMOTION_MAP = {
    "01": "Neutral", "02": "Calm",    "03": "Happy",   "04": "Sad",
    "05": "Angry",   "06": "Fearful", "07": "Disgust", "08": "Surprised",
}
EMOTION_LABELS = list(EMOTION_MAP.values())
NUM_CLASSES    = len(EMOTION_LABELS)

SR           = 22050
DURATION     = 3
N_MELS       = 128
HOP_LENGTH   = 512
IMG_H, IMG_W = 128, 128

BATCH_SIZE   = 32
EPOCHS       = 30
SEED         = 42
PATIENCE     = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("outputs", exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────
# ██  OS CONCEPT 1 — FILE MANAGEMENT & SYSTEM CALLS
# ─────────────────────────────────────────────────────────────
def parse_emotion(filename: str):
    """Extract emotion label from RAVDESS 7-part filename."""
    parts = Path(filename).stem.split("-")
    if len(parts) != 7:
        return None
    return EMOTION_MAP.get(parts[2], None)


def file_metadata_via_syscall(wav_path: Path) -> dict:
    """Use low-level POSIX os.stat() system call for file metadata
    instead of Python's high-level open() — direct OS interaction."""
    stat = os.stat(wav_path)            # ← stat(2) system call
    return {
        "size_bytes": stat.st_size,
        "inode":      stat.st_ino,
        "modified":   stat.st_mtime,
    }


# ─────────────────────────────────────────────────────────────
# ██  OS CONCEPT 2 — MEMORY MANAGEMENT  (mmap zero-copy read)
# ─────────────────────────────────────────────────────────────
def read_wav_mmap(wav_path: Path) -> bytes:
    """Read .wav via memory-mapped I/O.
    mmap maps the file directly into the process virtual address space.
    The OS copies pages on demand (demand paging) — avoids an extra
    kernel-buffer copy compared to a standard buffered read.
    O_BINARY flag added for Windows compatibility."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)  # O_BINARY for Windows
    fd = os.open(str(wav_path), flags)      # ← open(2) syscall
    try:
        size = os.fstat(fd).st_size         # ← fstat(2) syscall
        if size == 0:
            return b""
        with mmap.mmap(fd, size, access=mmap.ACCESS_READ) as mm:
            return mm.read()               # demand-paged by OS
    finally:
        os.close(fd)                        # ← close(2) syscall


def get_mem_mb() -> float:
    """Return current process RSS memory in MB."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / 1024 / 1024
    if HAS_RESOURCE:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return 0.0   # fallback on Windows without psutil


# ─────────────────────────────────────────────────────────────
# ██  Preprocessing helper  (used by single & multi-process)
# ─────────────────────────────────────────────────────────────
def wav_to_melspec(wav_path: Path) -> np.ndarray:
    """Load .wav → fixed-size normalised Mel Spectrogram (H x W float32)."""
    y, _ = librosa.load(wav_path, sr=SR, mono=True, duration=DURATION)
    max_len = SR * DURATION
    y = np.pad(y, (0, max(0, max_len - len(y))))[:max_len]
    mel    = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS,
                                             hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img    = sk_resize(mel_db, (IMG_H, IMG_W), anti_aliasing=True)
    img    = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img.astype(np.float32)


# ─────────────────────────────────────────────────────────────
# ██  OS CONCEPT 4 — MULTIPROCESSING  (parallel preprocessing)
# ─────────────────────────────────────────────────────────────
def _worker_preprocess(args):
    """Top-level picklable worker function for multiprocessing.Pool.
    Must be at module level so Windows (spawn) can pickle it."""
    wav_path, label = args
    try:
        return (wav_to_melspec(wav_path), label, None)
    except Exception as e:
        return (None, label, str(e))


def preprocess_multiprocess(files, labels, n_workers: int):
    """Parallel preprocessing using OS-level process pool.
    Pool.imap() creates n_workers child processes via fork()/spawn().
    Each worker has its own address space — true parallelism, no GIL."""
    args  = list(zip(files, labels))
    specs, lbls = [], []
    with Pool(processes=n_workers) as pool:
        for spec, lbl, err in tqdm(
            pool.imap(_worker_preprocess, args, chunksize=8),
            total=len(args), desc=f"  [MP {n_workers} workers]"
        ):
            if err is None:
                specs.append(spec)
                lbls.append(lbl)
    return specs, lbls


def preprocess_single(files, labels):
    """Sequential single-process baseline for timing comparison."""
    specs, lbls = [], []
    for p, l in tqdm(zip(files, labels), total=len(files),
                     desc="  [Single process]"):
        try:
            specs.append(wav_to_melspec(p))
            lbls.append(l)
        except Exception as e:
            print(f"  ⚠  {p.name}: {e}")
    return specs, lbls


# ─────────────────────────────────────────────────────────────
# ██  OS CONCEPT 6 — SYNCHRONISATION  (mutex on shared counter)
# ─────────────────────────────────────────────────────────────
_progress_lock  = threading.Lock()
_progress_count = [0]

def _thread_safe_increment():
    """Mutex-guarded increment — demonstrates critical section."""
    with _progress_lock:        # ← acquire / release mutex
        _progress_count[0] += 1
        return _progress_count[0]


# ─────────────────────────────────────────────────────────────
# ██  PyTorch Dataset
# ─────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (N, H, W) → (N, 1, H, W)  channel-first for PyTorch Conv2d
        self.X = torch.tensor(X[:, np.newaxis, :, :], dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# ██  CNN Models  — shallow vs deep  (design trade-off)
# ─────────────────────────────────────────────────────────────
class ShallowCNN(nn.Module):
    """1 conv block — less memory, faster, lower accuracy.
    Like a minimal OS config: low resource usage, limited capability."""
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.30),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class DeepCNN(nn.Module):
    """3 conv blocks — more memory, slower, higher accuracy.
    Like a full OS: more resources used, better performance."""
    def __init__(self, num_classes):
        super().__init__()
        def conv_block(in_ch, out_ch, drop):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(),
                nn.MaxPool2d(2), nn.Dropout2d(drop),
            )
        self.features = nn.Sequential(
            conv_block(1,  32,  0.25),
            conv_block(32, 64,  0.25),
            conv_block(64, 128, 0.40),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────
# ██  Training & Evaluation helpers
# ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_b)
        correct    += (out.argmax(1) == y_b).sum().item()
        total      += len(y_b)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_true = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        out  = model(X_b)
        loss = criterion(out, y_b)
        total_loss += loss.item() * len(y_b)
        preds       = out.argmax(1)
        correct    += (preds == y_b).sum().item()
        total      += len(y_b)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(y_b.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_true


def run_training(model, train_loader, val_loader, name):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4, min_lr=1e-6)

    best_val_acc, best_state, patience_cnt = 0.0, None, 0
    history = {"train_acc": [], "val_acc": [],
               "train_loss": [], "val_loss": []}

    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc           = train_one_epoch(model, train_loader,
                                                    criterion, optimizer)
        vl_loss, vl_acc, _, _     = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={tr_acc:.3f}  val={vl_acc:.3f}  "
                  f"lr={optimizer.param_groups[0]['lr']:.1e}")

        if patience_cnt >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    train_time = time.perf_counter() - t0
    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), f"outputs/best_ser_{name}.pt")
    return history, train_time, best_val_acc


# ══════════════════════════════════════════════════════════════
# ██  MAIN  (Windows requires __main__ guard for multiprocessing)
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for Windows .exe packaging

    print("=" * 65)
    print("  Speech Emotion Recognition — OS-Aware Pipeline")
    print(f"  Python {sys.version.split()[0]}  |  "
          f"PyTorch {torch.__version__}  |  Device: {DEVICE}")
    print("=" * 65)

    # ── STEP 1 — Download dataset ─────────────────────────────
    print("\nSTEP 1 — Downloading RAVDESS dataset via KaggleHub")
    import kagglehub
    path = kagglehub.dataset_download(
        "uwrfkaggler/ravdess-emotional-speech-audio")
    print(f"  Dataset path: {path}")
    DATASET_PATH = Path(path)

    # ── STEP 2 — File Management (system calls) ───────────────
    print("\nSTEP 2 — File Management via POSIX system calls")
    wav_files, wav_labels = [], []
    for wav_path in sorted(DATASET_PATH.rglob("*.wav")):
        emotion = parse_emotion(wav_path.name)
        if emotion is not None:
            wav_files.append(wav_path)
            wav_labels.append(emotion)

    if wav_files:
        meta = file_metadata_via_syscall(wav_files[0])
        print(f"  Sample file  : {wav_files[0].name}")
        print(f"  Size (bytes) : {meta['size_bytes']:,}  [via os.stat() syscall]")
        print(f"  Inode        : {meta['inode']}")
    print(f"  Total files  : {len(wav_files)}")

    # ── STEP 3 — Memory Management (mmap) ────────────────────
    print("\nSTEP 3 — Memory Management: mmap vs standard read")
    sample = wav_files[0]

    mem_before = get_mem_mb()
    t0 = time.perf_counter()
    with open(sample, "rb") as f:
        _ = f.read()
    t_std   = time.perf_counter() - t0
    mem_std = get_mem_mb() - mem_before

    mem_before = get_mem_mb()
    t0 = time.perf_counter()
    _ = read_wav_mmap(sample)
    t_mm   = time.perf_counter() - t0
    mem_mm = get_mem_mb() - mem_before

    print(f"  Standard read : {t_std*1000:.2f} ms  | mem delta ≈ {mem_std:.2f} MB")
    print(f"  mmap read     : {t_mm*1000:.2f} ms  | mem delta ≈ {mem_mm:.2f} MB")
    print("  [mmap: OS maps file into virtual memory — demand paging]")

    # ── STEP 4 — I/O Management (buffered vs unbuffered) ──────
    print("\nSTEP 4 — I/O Management: buffered vs unbuffered read")
    N_TRIALS = 5
    times_buf, times_raw = [], []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        with open(sample, "rb") as f:
            _ = f.read()
        times_buf.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        fd = os.open(str(sample), flags)
        _ = os.read(fd, os.path.getsize(sample))  # ← read(2) syscall
        os.close(fd)
        times_raw.append(time.perf_counter() - t0)

    print(f"  Buffered   avg : {np.mean(times_buf)*1000:.2f} ms "
          f"(OS page-cache exploited after 1st access)")
    print(f"  Unbuffered avg : {np.mean(times_raw)*1000:.2f} ms "
          f"(direct read(2) syscall, no Python buffer)")

    # ── STEP 5 — CPU Scheduling ───────────────────────────────
    N_CPUS = os.cpu_count()
    print(f"\nSTEP 5 — CPU Scheduling")
    print(f"  Detected CPUs : {N_CPUS}")
    print(f"  Pool workers  : {N_CPUS}  (one OS process per logical CPU)")
    try:
        os.nice(3)
        print("  os.nice(+3) — main process yields to foreground tasks")
    except (PermissionError, AttributeError, OSError):
        print("  (os.nice() skipped — not supported on this platform)")

    # ── STEP 6 — Synchronisation demo ────────────────────────
    print("\nSTEP 6 — Synchronisation: threading.Lock() mutex demo")
    threads = [threading.Thread(target=_thread_safe_increment)
               for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    print(f"  Shared counter after 10 thread increments: "
          f"{_progress_count[0]}  (no race condition)")

    # ── STEP 7 — Performance trade-off: single vs multi ───────
    SAMPLE_N = min(100, len(wav_files))
    print(f"\nSTEP 7 — Performance Trade-off: single vs multi-process "
          f"({SAMPLE_N} files)")

    t0 = time.perf_counter()
    sp_specs, sp_labels = preprocess_single(
        wav_files[:SAMPLE_N], wav_labels[:SAMPLE_N])
    t_single = time.perf_counter() - t0
    print(f"  Single-process : {t_single:.2f}s")

    t0 = time.perf_counter()
    mp_specs, mp_labels = preprocess_multiprocess(
        wav_files[:SAMPLE_N], wav_labels[:SAMPLE_N], n_workers=N_CPUS)
    t_multi  = time.perf_counter() - t0
    speedup  = t_single / max(t_multi, 0.001)
    print(f"  Multi-process  : {t_multi:.2f}s  (speedup ≈ {speedup:.1f}×)")

    # Save performance chart
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Single-process", f"Multi-process\n({N_CPUS} workers)"],
        [t_single, t_multi],
        color=["#E07B54", "#4C9BE8"],
        edgecolor="black", linewidth=0.7, width=0.45,
    )
    for bar, t in zip(bars, [t_single, t_multi]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{t:.2f}s", ha="center", fontsize=12, fontweight="bold")
    ax.set_title(f"Preprocessing Time — Single vs Multi-process\n"
                 f"({SAMPLE_N} files, {N_CPUS} CPU cores)", fontsize=12)
    ax.set_ylabel("Time (seconds)")
    ax.set_ylim(0, max(t_single, t_multi) * 1.35)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/performance_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → outputs/performance_tradeoff.png")

    # ── STEP 8 — Preprocess ALL files ────────────────────────
    print(f"\nSTEP 8 — Preprocessing all {len(wav_files)} files (multi-process)")
    if len(wav_files) > SAMPLE_N:
        all_specs, all_labels = preprocess_multiprocess(
            wav_files, wav_labels, n_workers=N_CPUS)
    else:
        all_specs, all_labels = mp_specs, mp_labels

    X = np.array(all_specs, dtype=np.float32)
    label_to_idx = {lbl: i for i, lbl in enumerate(EMOTION_LABELS)}
    y = np.array([label_to_idx[l] for l in all_labels], dtype=np.int64)
    print(f"  Feature array : {X.shape}   Label array : {y.shape}")

    # ── STEP 9 — Sample spectrograms ─────────────────────────
    print("\nSTEP 9 — Saving sample Mel Spectrograms")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    shown = set()
    for img, lbl in zip(X, all_labels):
        if lbl not in shown:
            ax = axes[len(shown) // 4][len(shown) % 4]
            ax.imshow(img, aspect="auto", origin="lower", cmap="magma")
            ax.set_title(lbl, fontsize=11, fontweight="bold")
            ax.axis("off")
            shown.add(lbl)
        if len(shown) == NUM_CLASSES:
            break
    plt.suptitle("Sample Mel Spectrograms — RAVDESS", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/sample_spectrograms.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → outputs/sample_spectrograms.png")

    # ── STEP 10 — Train / Val / Test split ───────────────────
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp)
    print(f"\nSTEP 10 — Split  Train:{len(y_tr)}  "
          f"Val:{len(y_val)}  Test:{len(y_te)}")

    train_ds = EmotionDataset(X_tr,  y_tr)
    val_ds   = EmotionDataset(X_val, y_val)
    test_ds  = EmotionDataset(X_te,  y_te)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0)

    # ── STEP 11 — Train Shallow vs Deep (design trade-off) ───
    results = {}
    for name, ModelClass in [("shallow", ShallowCNN), ("deep", DeepCNN)]:
        print(f"\nSTEP 11 — Training {name.upper()} CNN …")
        model = ModelClass(NUM_CLASSES).to(DEVICE)
        print(f"  Parameters: {count_params(model):,}")
        hist, t_train, best_acc = run_training(
            model, train_loader, val_loader, name)

        _, test_acc, preds, trues = evaluate(
            model, test_loader, nn.CrossEntropyLoss())

        results[name] = {
            "model":      model,
            "history":    hist,
            "test_acc":   test_acc,
            "train_time": t_train,
            "params":     count_params(model),
            "preds":      preds,
            "trues":      trues,
        }
        print(f"  {name:7s} → test acc: {test_acc*100:.2f}%  |  "
              f"time: {t_train:.1f}s")

    # ── STEP 12 — Design trade-off chart ─────────────────────
    print("\nSTEP 12 — Saving design trade-off chart")
    depths   = list(results.keys())
    accs     = [results[d]["test_acc"] * 100 for d in depths]
    times    = [results[d]["train_time"]      for d in depths]
    params_k = [results[d]["params"] / 1000   for d in depths]
    palette  = ["#7DBEAA", "#4C72B0"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, vals, title, ylabel in zip(
        axes,
        [accs, times, params_k],
        ["Test Accuracy (%)", "Training Time (s)", "Parameters (K)"],
        ["Accuracy (%)", "Seconds", "Thousands"],
    ):
        bars = ax.bar(depths, vals, color=palette,
                      edgecolor="black", linewidth=0.7)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.01,
                    f"{v:.1f}", ha="center", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Design Trade-off: Shallow vs Deep CNN\n"
                 "(Memory footprint  vs  Accuracy  vs  Compute)", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/design_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → outputs/design_tradeoff.png")

    # ── STEP 13 — Training curves (deep model) ───────────────
    print("\nSTEP 13 — Saving training curves (deep model)")
    hist = results["deep"]["history"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (tr_key, vl_key), title in zip(
        axes,
        [("train_acc", "val_acc"), ("train_loss", "val_loss")],
        ["Accuracy", "Loss"],
    ):
        ax.plot(hist[tr_key], label="Train", color="#4C72B0")
        ax.plot(hist[vl_key], label="Val",   color="#DD8452")
        ax.set_title(f"{title} per Epoch — Deep CNN", fontsize=12)
        ax.set_xlabel("Epoch"); ax.set_ylabel(title)
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("Training History — SER Deep CNN (PyTorch)", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → outputs/training_curves.png")

    # ── STEP 14 — Classification report ──────────────────────
    preds = results["deep"]["preds"]
    trues = results["deep"]["trues"]
    print("\nSTEP 14 — Classification Report (Deep CNN):")
    print(classification_report(trues, preds,
                                 target_names=EMOTION_LABELS, digits=3))

    # ── STEP 15 — Confusion matrix ────────────────────────────
    print("STEP 15 — Saving confusion matrix")
    cm      = confusion_matrix(trues, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f",
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS,
                cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Confusion Matrix (Normalised) — Deep CNN", fontsize=13)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → outputs/confusion_matrix.png")

    # ── STEP 16 — Class distribution chart ───────────────────
    print("STEP 16 — Saving class distribution chart")
    unique_lbl, counts = np.unique(all_labels, return_counts=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(unique_lbl, counts,
                  color=sns.color_palette("Set2", len(unique_lbl)),
                  edgecolor="black", linewidth=0.7)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3, str(c), ha="center", fontsize=11)
    ax.set_title("RAVDESS — Emotion Class Distribution", fontsize=13)
    ax.set_xlabel("Emotion"); ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved → outputs/class_distribution.png")

    # ── OS CONCEPTS SUMMARY ───────────────────────────────────
    print("\n" + "=" * 65)
    print("  OS CONCEPTS IMPLEMENTED — SUMMARY")
    print("=" * 65)
    summary = [
        ("1. Multiprocessing",  f"Pool({N_CPUS} workers) fork/spawn — "
                                 f"speedup ≈ {speedup:.1f}×"),
        ("2. Synchronisation",  "threading.Lock() mutex — no race condition"),
        ("3. CPU Scheduling",   f"os.nice(+3); pool=cpu_count()={N_CPUS}"),
        ("4. Memory Mgmt",      "mmap zero-copy: OS demand-pages .wav files"),
        ("5. System Calls",     "open(2) read(2) fstat(2) stat(2) close(2)"),
        ("6. I/O Management",   "Buffered vs unbuffered read(2) timing"),
        ("7. Design Trade-off", "Shallow vs Deep CNN: accuracy/memory/speed"),
    ]
    for concept, detail in summary:
        print(f"  {concept:<22}  {detail}")
    print("=" * 65)

    deep_acc = results["deep"]["test_acc"] * 100
    print(f"\n  Deep CNN test accuracy : {deep_acc:.2f}%")
    print("  All outputs saved in   : ./outputs/")
    for f in [
        "sample_spectrograms.png", "performance_tradeoff.png",
        "design_tradeoff.png",     "training_curves.png",
        "confusion_matrix.png",    "class_distribution.png",
        "best_ser_deep.pt",        "best_ser_shallow.pt",
    ]:
        print(f"    • {f}")


    # ── Inference helper ──────────────────────────────────────
    def predict_emotion(wav_path: str) -> dict:
        """Predict emotion from a single .wav file using the deep model."""
        model = results["deep"]["model"]
        model.eval()
        spec  = wav_to_melspec(Path(wav_path))
        tensor = torch.tensor(spec[np.newaxis, np.newaxis, :, :],
                               dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        return {
            "emotion":       EMOTION_LABELS[idx],
            "confidence":    float(probs[idx]),
            "probabilities": dict(zip(EMOTION_LABELS, probs.tolist())),
        }

    # Example usage (uncomment to test):
    result = predict_emotion(r"C:\Users\ZIA\Documents\OS\testaudio.wav")
    print(result)
