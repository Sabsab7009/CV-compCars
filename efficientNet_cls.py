# train_effnetv2s_pytorch.py
# effinet_cls_v3.py
import argparse, os, time, gc
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score, f1_score

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

"""
pip install torch torchvision scikit-learn seaborn

# -------------------- Warm-up Phase --------------------
# Train classifier head only (backbone frozen)
python effinet_cls_v3.py \
  --data /path/to/data \
  --out runs_effv2s \
  --total-epochs 120 \
  --warmup-epochs 20 \
  --chunk 0 \
  --imgsz 320 \
  --batch 128

# -------------------- Fine-tuning Phase --------------------
# Chunk 1 (first fine-tuning block)
python effinet_cls_v3.py \
  --data /path/to/data \
  --out runs_effv2s \
  --total-epochs 120 \
  --warmup-epochs 20 \
  --chunk-size 60 \
  --chunk 1 \
  --imgsz 320 \
  --batch 128 \
  --test

# Optional: resume a run
python effinet_cls_v3.py \
  --data /path/to/data \
  --out runs_effv2s \
  --chunk 1 \
  --resume path/to/checkpoint.pt

# Later chunks (if needed)
python effinet_cls_v3.py \
  --data /path/to/data \
  --out runs_effv2s \
  --total-epochs 120 \
  --warmup-epochs 20 \
  --chunk-size 60 \
  --chunk 2 \
  --imgsz 320 \
  --batch 128 \
  --test

# -------------------- Notes --------------------
# - Dataset must follow ImageFolder structure:
#   root/
#     train/
#     val/
#     (optional) test/
#
# - Training is split into chunks:
#     total_epochs = warmup + fine-tuning
#     fine-tuning = total_epochs - warmup_epochs
#
# - You can run as many chunks as needed (chunk=1,2,3,...),
#   but total training will never exceed total_epochs.
#
# - If a chunk is already completed, it will automatically skip.
#
# - If you hit GPU OOM:
#     reduce batch size (e.g., --batch 64)
"""
# -----------------------
# Args
# -----------------------
def parse_args():
    p = argparse.ArgumentParser("EfficientNetV2-S Trainer (YOLOv8-aligned)")
    p.add_argument("--data", required=True, help="root dir with train/ val/ test/")
    p.add_argument("--out", default="boxCars/runs_effv2s_aligned")
    p.add_argument("--total-epochs", type=int, default=120)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=60, help="fine-tune epochs per run")
    p.add_argument("--chunk", type=int, default=0, help="0-based chunk idx (0 is warm-up)")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--throughput-batch", type=int, default=128, help="batch size for throughput timing")
    p.add_argument("--test", action="store_true", help="run final test evaluation if test/ exists")
    return p.parse_args()

# -----------------------
# Data
# -----------------------
def make_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.15)),
    ])
    # Deterministic val/test: keep aspect, center crop
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf

def get_dataloaders(data_path, img_size, batch):
    train_tf, eval_tf = make_transforms(img_size)
    train = datasets.ImageFolder(os.path.join(data_path, "train"), train_tf)
    val   = datasets.ImageFolder(os.path.join(data_path, "val"),   eval_tf)
    train_dl = DataLoader(train, batch_size=batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val,   batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_dl, val_dl, train.classes

def make_test_loader(data_path, img_size, batch):
    _, eval_tf = make_transforms(img_size)
    test_dir = os.path.join(data_path, "test")
    if not os.path.isdir(test_dir):
        return None, None
    test = datasets.ImageFolder(test_dir, eval_tf)
    test_dl = DataLoader(test, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return test_dl, test.classes

# -----------------------
# Model utils
# -----------------------
def freeze_backbone(model):
    for name, p in model.named_parameters():
        if "classifier" not in name:
            p.requires_grad = False

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

@torch.inference_mode()
def measure_speed(model, img_size, device="cuda", batch_for_throughput=128):
    model.eval().to(device)
    x1 = torch.randn(1, 3, img_size, img_size, device=device)
    xb = torch.randn(batch_for_throughput, 3, img_size, img_size, device=device)

    # warmup
    for _ in range(20):
        _ = model(x1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # latency (single image)
    iters = 200
    t0 = time.time()
    for _ in range(iters):
        _ = model(x1)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    lat_ms = (time.time() - t0) * 1000.0 / iters

    # throughput (batch)
    # keep it moderate to avoid OOM
    iters = 50
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    for _ in range(iters):
        _ = model(xb)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    imgs_per_s = (batch_for_throughput * iters) / (time.time() - t0)
    return lat_ms, imgs_per_s

# -----------------------
# Eval helpers
# -----------------------
@torch.no_grad()
def quick_validate(model, loader, criterion, device):
    model.eval().to(device)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(n, 1)

@torch.no_grad()
def evaluate_split(model, loader, device, classes, out_dir: Path, split_name: str):
    """Full evaluation: top-1, top-5, macro-F1, confusion matrix (raw + normalized)."""
    model.eval().to(device)
    logits_all, y_all = [], []
    for x, y in tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch"):
        x = x.to(device, non_blocking=True)
        logits_all.append(model(x).cpu())
        y_all.append(y)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)

    
    y_np      = y.numpy()
    logits_np = logits.numpy()
    preds_np  = logits_np.argmax(1)

    top1     = accuracy_score(y_np, preds_np)
    top5     = top_k_accuracy_score(y_np, logits_np, k=5)
    macro_f1 = f1_score(y_np, preds_np, average="macro")

    cm_raw  = confusion_matrix(y_np, preds_np, labels=range(len(classes)))
    cm_norm = confusion_matrix(y_np, preds_np, labels=range(len(classes)), normalize="true")

    # Save fig
    fig = plt.figure(figsize=(8, 6), dpi=200)
    sns.heatmap(cm_norm, vmin=0, vmax=1, cmap="Blues", cbar=True,
                xticklabels=False, yticklabels=False)
    plt.title(f"Confusion Matrix (norm) — {split_name}")
    cm_png = out_dir / f"confmat_{split_name}.png"
    fig.savefig(cm_png, dpi=200, bbox_inches="tight"); plt.close(fig)

    # Save raw cm CSV
    pd.DataFrame(cm_raw).to_csv(out_dir / f"confmat_{split_name}_raw.csv", index=False)

    # Log CSV
    row = {
        "split": split_name,
        "top1": round(top1, 6),
        "top5": round(top5, 6),
        "macro_f1": round(macro_f1, 6),
        "num_samples": int(len(y))
    }
    csv_path = out_dir / f"eval_{split_name}.csv"
    if not csv_path.exists():
        pd.DataFrame([row]).to_csv(csv_path, index=False)
    else:
        pd.concat([pd.read_csv(csv_path), pd.DataFrame([row])], ignore_index=True).to_csv(csv_path, index=False)

    print(f"✓ Saved confusion matrix → {cm_png}")
    print(f"✓ Saved raw CM CSV      → {out_dir / f'confmat_{split_name}_raw.csv'}")
    print(f"✓ Metrics CSV           → {csv_path}")
    return row

# -----------------------
# Main
# -----------------------
def main():
    gc.collect()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True

    args = parse_args()
    ROOT = Path(args.out)
    is_warmup = (args.chunk == 0)

    if is_warmup:
        run_dir = ROOT / "phase1_warmup"
        epochs_to_run = args.warmup_epochs
        start_ep_offset = 0
    else:
        run_dir = ROOT / f"phase2_chunk{args.chunk - 1}"
        ft_epochs_total = args.total_epochs - args.warmup_epochs
        ft_start_epoch = (args.chunk - 1) * args.chunk_size
        epochs_left = ft_epochs_total - ft_start_epoch
        epochs_to_run = max(0, min(args.chunk_size, epochs_left))
        start_ep_offset = args.warmup_epochs + ft_start_epoch

    if epochs_to_run == 0:
        print("Nothing to train in this chunk."); return

    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"

    # Data
    train_dl, val_dl, classes = get_dataloaders(args.data, args.imgsz, args.batch)
    test_dl, _ = make_test_loader(args.data, args.imgsz, args.batch)

    # Save class-id ↔ name mapping (useful for reports & confusion matrix reading)
    pd.DataFrame({
        "class_id": np.arange(len(classes), dtype=int),
        "class_name": classes,
    }).to_csv(run_dir / "classes.csv", index=False)
    # Model
    weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
    model = models.efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))

    best_acc = 0.0
    start_ep = 0

    # Resume
    resume_path = None
    if args.resume:
        resume_path = args.resume if Path(args.resume).is_file() else run_dir / "last.pt"
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_ep = ckpt.get("epoch", -1) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"Resuming training from epoch {start_ep}")
    elif not is_warmup and args.chunk == 1:
        warmup_best = ROOT / "phase1_warmup" / "best.pt"
        if warmup_best.exists():
            print(f"Starting fine-tune from best warm-up weights: {warmup_best}")
            model.load_state_dict(torch.load(warmup_best))
        else:
            print("WARNING: Best warm-up weights not found. Starting fine-tune from ImageNet weights.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    already_done = max(0, start_ep - start_ep_offset)   # start_ep is set when resuming; else 0
    remaining_epochs = epochs_to_run - already_done
    if remaining_epochs <= 0:
        print(f"This chunk is already complete (done {already_done}/{epochs_to_run}). Nothing to do.")
        return

    print(f"→ Chunk progress: done={already_done}, remaining={remaining_epochs} (of {epochs_to_run})")

    # Opt & sched (YOLOv8-aligned) — use remaining_epochs for prints and cosine horizon
    scheduler = None
    if is_warmup:
        print(f"► Phase-1 Warm-up for {remaining_epochs} epochs… (backbone frozen, SGD)")
        freeze_backbone(model)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=5e-4)
    else:
        print(f"► Phase-2 Fine-tune for {remaining_epochs} epochs… (AdamW + CosineAnnealingLR)")
        unfreeze_all(model)  # must be BEFORE creating optimizer
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)


    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # CSV header
    if not csv_path.exists() or start_ep == 0:
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_top1\n")


    for i in range(remaining_epochs):
        epoch = (start_ep_offset + already_done) + i
        # Train
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Train Ep {epoch}", unit="batch"):
            x, y = x.to(device, non_blocking=True), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # Step LR only in fine-tune
        if scheduler: scheduler.step()

        # Quick val (loss + top1)
        val_loss, val_top1 = quick_validate(model, val_dl, criterion, device)

        avg_train_loss = train_loss / max(len(train_dl), 1)
        print(f"Ep {epoch:03d}  TrainLoss {avg_train_loss:.4f}  ValLoss {val_loss:.4f}  ValTop1 {val_top1:.4f}")

        with open(csv_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{val_loss:.4f},{val_top1:.4f}\n")

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_acc": best_acc,
        }, run_dir / "last.pt")

        # Save best
        if val_top1 > best_acc:
            best_acc = val_top1
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"New best accuracy: {best_acc:.4f}. Saved to best.pt")

    # End-of-chunk: full VAL + TEST evaluation, speed, params
    params_m = count_params_m(model)
    lat_ms, ips = measure_speed(model, args.imgsz, device, args.throughput_batch)

    meta_row = {
        "params_m": round(params_m, 3),
        "latency_ms_1img": round(lat_ms, 3),
        "throughput_img_s": round(ips, 2),
        "imgsz": args.imgsz,
        "batch_throughput": args.throughput_batch
    }
    pd.DataFrame([meta_row]).to_csv(run_dir / "model_profile.csv", index=False)
    print("✓ Model profile:", meta_row)

    # Full VAL evaluate (top1/top5/macroF1 + CM)
    evaluate_split(model, val_dl, device, classes, run_dir, split_name="val")

    # Optional TEST evaluate
    if args.test and test_dl is not None:
        evaluate_split(model, test_dl, device, classes, run_dir, split_name="test")

    # Curves from metrics.csv
    df = pd.read_csv(csv_path)
    # Loss
    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"],   label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=200); plt.close()
    # Acc
    plt.figure(figsize=(8, 4))
    plt.plot(df["epoch"], df["val_top1"], color="green", label="Val top-1")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Val Top-1 vs Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "acc_curve.png", dpi=200); plt.close()
    print("✓ Figures saved:", run_dir / "loss_curve.png", " & ", run_dir / "acc_curve.png")

if __name__ == "__main__":
    main()
