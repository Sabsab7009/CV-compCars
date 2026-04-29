#!/usr/bin/env python3
# train_timm_cls.py — ConvNeXt-T / DeiT-S / MobileNetV3-Large
"""
# -------------------- Installation --------------------
pip install timm==0.9.16 torch torchvision scikit-learn seaborn

# -------------------- Warm-up Phase (chunk=0) --------------------
# Train classifier head only (backbone frozen)

# ConvNeXt-Tiny
python train_timm_cls.py --model convnext_tiny.fb_in22k_ft_in1k \
  --data /path/to/data \
  --out runs_timm \
  --total-epochs 120 \
  --warmup-epochs 20 \
  --chunk 0 \
  --imgsz 320 \
  --batch 128

# MobileNetV3-Large
python train_timm_cls.py --model mobilenetv3_large_100 \
  --data /path/to/data \
  --out runs_timm \
  --total-epochs 120 \
  --warmup-epochs 20 \
  --chunk 0 \
  --imgsz 320 \
  --batch 128


# -------------------- Fine-tuning Phase --------------------
# Continue training with full model (unfrozen)

# Example: Chunk 1 (first 60 epochs of fine-tuning)
python train_timm_cls.py --model convnext_tiny.fb_in22k_ft_in1k \
  --data /path/to/data \
  --out runs_timm \
  --total-epochs 120 \
  --warmup-epochs 20 \
  --chunk-size 60 \
  --chunk 1 \
  --imgsz 320 \
  --batch 128 \
  --test


# -------------------- Notes --------------------
# - Supported models:
#     * convnext_tiny.fb_in22k_ft_in1k
#     * mobilenetv3_large_100
#
# - Dataset must follow ImageFolder structure:
#     root/
#       train/
#       val/
#       (optional) test/
#
# - Training is split into chunks:
#     total training = total_epochs
#     fine-tuning    = total_epochs - warmup_epochs
#
# - You can run multiple chunks sequentially (chunk=1,2,3,...),
#   but total epochs will never exceed total_epochs.
#
# - If GPU runs out of memory:
#     reduce batch size (e.g., --batch 64)
"""
import argparse, os, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score, f1_score
import timm
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def parse_args():
    p = argparse.ArgumentParser("timm finetune (YOLO-aligned schedule)")
    p.add_argument("--data", required=True, help="root with train/ val/ (optional test/)")
    p.add_argument("--out", default="runs_timm")
    p.add_argument("--total-epochs", type=int, default=120)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=60)
    p.add_argument("--chunk", type=int, default=0, help="0=warmup, 1..N=fine-tune chunks")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--throughput-batch", type=int, default=128)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--test", action="store_true")
    return p.parse_args()

# ---------------- data ----------------
def make_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        transforms.RandomErasing(p=0.4, scale=(0.02,0.15)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def get_dataloaders(root, img_size, batch):
    train_tf, eval_tf = make_transforms(img_size)
    train = datasets.ImageFolder(os.path.join(root, "train"), train_tf)
    val   = datasets.ImageFolder(os.path.join(root, "val"),   eval_tf)
    train_dl = DataLoader(train, batch_size=batch, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val,   batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return train_dl, val_dl, train.classes

def make_test_loader(root, img_size, batch):
    _, eval_tf = make_transforms(img_size)
    test_dir = os.path.join(root, "test")
    if not os.path.isdir(test_dir): return None, None
    test = datasets.ImageFolder(test_dir, eval_tf)
    test_dl = DataLoader(test, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)
    return test_dl, test.classes

# --------------- model utils ---------------
def count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def freeze_backbone_timm(model):
    # Freeze all except classifier head
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze head/classifier parameters
    # timm exposes a convenience helper:
    for p in model.get_classifier().parameters():
        p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

@torch.inference_mode()
def measure_speed(model, img_size, device="cuda", batch_for_throughput=128):
    model.eval().to(device)
    x1 = torch.randn(1, 3, img_size, img_size, device=device)
    xb = torch.randn(batch_for_throughput, 3, img_size, img_size, device=device)
    for _ in range(20): _ = model(x1)  # warmup
    if torch.cuda.is_available(): torch.cuda.synchronize()
    iters = 200
    t0 = time.time()
    for _ in range(iters): _ = model(x1)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    lat_ms = (time.time() - t0) * 1000.0 / iters
    iters = 50
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters): _ = model(xb)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    ips = (batch_for_throughput * iters) / (time.time() - t0)
    return lat_ms, ips

# --------------- eval helpers ---------------
@torch.no_grad()
def quick_validate(model, loader, criterion, device):
    model.eval().to(device)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / max(len(loader), 1), correct / max(n, 1)

@torch.no_grad()
def evaluate_split(model, loader, device, classes, out_dir: Path, split_name: str):
    model.eval().to(device)
    logits_all, y_all = [], []
    for x, y in tqdm(loader, desc=f"Evaluating [{split_name}]", unit="batch"):
        x = x.to(device, non_blocking=True)
        logits_all.append(model(x).cpu())
        y_all.append(y)
    logits = torch.cat(logits_all)
    y      = torch.cat(y_all)

    y_np      = y.numpy()
    logits_np = logits.numpy()
    preds_np  = logits_np.argmax(1)

    top1     = accuracy_score(y_np, preds_np)
    top5     = top_k_accuracy_score(y_np, logits_np, k=5)
    macro_f1 = f1_score(y_np, preds_np, average="macro")

    cm_raw  = confusion_matrix(y_np, preds_np, labels=range(len(classes)))
    cm_norm = confusion_matrix(y_np, preds_np, labels=range(len(classes)), normalize="true")

    # save CM fig
    fig = plt.figure(figsize=(8,6), dpi=200)
    sns.heatmap(cm_norm, vmin=0, vmax=1, cmap="Blues", cbar=True,
                xticklabels=False, yticklabels=False)
    plt.title(f"Confusion Matrix (norm) — {split_name}")
    cm_png = out_dir / f"confmat_{split_name}.png"
    fig.savefig(cm_png, dpi=200, bbox_inches="tight"); plt.close(fig)

    pd.DataFrame(cm_raw).to_csv(out_dir / f"confmat_{split_name}_raw.csv", index=False)

    row = {"split": split_name, "top1": round(top1,6), "top5": round(top5,6),
           "macro_f1": round(macro_f1,6), "num_samples": int(len(y_np))}
    csv_path = out_dir / f"eval_{split_name}.csv"
    if not csv_path.exists():
        pd.DataFrame([row]).to_csv(csv_path, index=False)
    else:
        pd.concat([pd.read_csv(csv_path), pd.DataFrame([row])], ignore_index=True).to_csv(csv_path, index=False)

    print(f"Saved CM → {cm_png}")
    print(f"Saved raw CM → {out_dir / f'confmat_{split_name}_raw.csv'}")
    print(f"Metrics CSV → {csv_path}")
    return row

# ---------------- main ----------------
def main():
    gc.collect(); torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    model_name = "convnext_tiny.fb_in22k_ft_in1k" #or "mobilenetv3_large_100" or ""
    args = parse_args()
    ROOT = Path(args.out) / model_name.replace("/", "_")
    is_warmup = (args.chunk == 0)

    # schedule bookkeeping
    if is_warmup:
        run_dir = ROOT / "phase1_warmup"
        epochs_to_run = args.warmup_epochs
        start_ep_offset = 0
    else:
        run_dir = ROOT / f"phase2_chunk{args.chunk - 1}"
        ft_epochs_total = args.total_epochs - args.warmup_epochs
        ft_start_epoch  = (args.chunk - 1) * args.chunk_size
        epochs_left     = ft_epochs_total - ft_start_epoch
        epochs_to_run   = max(0, min(args.chunk_size, epochs_left))
        start_ep_offset = args.warmup_epochs + ft_start_epoch

    if epochs_to_run == 0:
        print("Nothing to train in this chunk."); return
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"

    # data
    train_dl, val_dl, classes = get_dataloaders(args.data, args.imgsz, args.batch)
    test_dl, _ = make_test_loader(args.data, args.imgsz, args.batch)

    # save classes
    pd.DataFrame({"class_id": np.arange(len(classes), dtype=int),
                  "class_name": classes}).to_csv(run_dir / "classes.csv", index=False)

    # model
    model = timm.create_model(model_name, pretrained=True, num_classes=len(classes))
    # (timm already replaces the head when num_classes is set)

    best_acc = 0.0
    start_ep = 0

    # resume
    # ------------------- load starting weights clearly -------------------
    start_source = "imagenet_pretrained"
    resume_path = None
    if args.resume:
        rp = Path(args.resume)
        resume_path = rp if rp.is_file() else (run_dir / "last.pt")

    if resume_path and resume_path.exists():
        # Highest priority: explicit resume
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_ep = ckpt.get("epoch", -1) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        start_source = f"resume:{resume_path}"
    elif (not is_warmup) and args.chunk == 1:
        # Next: start chunk-1 from warm-up best
        warmup_best = ROOT / "phase1_warmup" / "best.pt"
        if warmup_best.exists():
            model.load_state_dict(torch.load(warmup_best, map_location="cpu"))
            start_source = f"warmup_best:{warmup_best}"
        else:
            print("WARNING: phase1_warmup/best.pt not found; starting from ImageNet weights.")
    else:
        # Warm-up or later chunks without resume (rare)
        pass

    print(f"→ Starting weights: {start_source}")
    # ---------------------------------------------------------------------


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # --- figure out how much of this chunk is already done ---
    already_done = max(0, start_ep - start_ep_offset)   # start_ep set if resuming; else 0
    remaining_epochs = epochs_to_run - already_done
    if remaining_epochs <= 0:
        print(f"This chunk is already complete (done {already_done}/{epochs_to_run}). Nothing to do.")
        return

    print(f"→ Chunk progress: done={already_done}, remaining={remaining_epochs} (of {epochs_to_run})")

    # --- opt & sched (use remaining_epochs for prints AND scheduler horizon) ---
    scheduler = None
    if is_warmup:
        print(f"► Warm-up {remaining_epochs} epochs (freeze backbone, train head with SGD)")
        freeze_backbone_timm(model)  # or freeze_backbone(...) in your EffNet script
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937, weight_decay=5e-4)
    else:
        print(f"► Fine-tune {remaining_epochs} epochs (unfreeze, AdamW + Cosine)")
        unfreeze_all(model)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    # csv header
    if not csv_path.exists() or start_ep == 0:
        with open(csv_path, "w") as f:
            f.write("epoch,train_loss,val_loss,val_top1\n")

    # --- train only for the remaining epochs; keep global epoch numbering consistent ---
    for i in range(remaining_epochs):
        epoch = (start_ep_offset + already_done) + i

        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Train Ep {epoch}", unit="batch"):
            x, y = x.to(device, non_blocking=True), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        if scheduler:
            scheduler.step()

        # quick val
        val_loss, val_top1 = quick_validate(model, val_dl, criterion, device)
        avg_train_loss = train_loss / max(len(train_dl), 1)
        print(f"Ep {epoch:03d}  TrainLoss {avg_train_loss:.4f}  ValLoss {val_loss:.4f}  ValTop1 {val_top1:.4f}")
        with open(csv_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.4f},{val_loss:.4f},{val_top1:.4f}\n")

        # save last
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "best_acc": best_acc},
                   run_dir / "last.pt")
        # save best
        if val_top1 > best_acc:
            best_acc = val_top1
            torch.save(model.state_dict(), run_dir / "best.pt")
            print(f"New best accuracy: {best_acc:.4f} → saved best.pt")

    # profile + full evals
    params_m = count_params_m(model)
    lat_ms, ips = measure_speed(model, args.imgsz, device, args.throughput_batch)
    pd.DataFrame([{
        "params_m": round(params_m,3),
        "latency_ms_1img": round(lat_ms,3),
        "throughput_img_s": round(ips,2),
        "imgsz": args.imgsz,
        "batch_throughput": args.throughput_batch,
        "model": model_name,
    }]).to_csv(run_dir / "model_profile.csv", index=False)
    print("Profile saved:", run_dir / "model_profile.csv")

    evaluate_split(model, val_dl,  device, classes, run_dir, "val")
    if args.test and test_dl is not None:
        evaluate_split(model, test_dl, device, classes, run_dir, "test")

    # curves
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(8,4))
    plt.plot(df["epoch"], df["train_loss"], label="Train loss")
    plt.plot(df["epoch"], df["val_loss"],   label="Val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs Epoch")
    plt.legend(); plt.tight_layout(); plt.savefig(run_dir / "loss_curve.png", dpi=200); plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(df["epoch"], df["val_top1"], color="green", label="Val top-1")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Val Top-1 vs Epoch")
    plt.legend(); plt.tight_layout(); plt.savefig(run_dir / "acc_curve.png", dpi=200); plt.close()
    print("Curves saved")

if __name__ == "__main__":
    main()
