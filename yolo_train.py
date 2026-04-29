"""
Chunked YOLOv8 detection + classification training pipeline (BoxCars116k)

============================================================
PIPELINE OVERVIEW
============================================================

This script performs end-to-end training for vehicle:
- Detection (bounding boxes)
- Classification (make/model or class ID)

It is designed for large-scale datasets (e.g., BoxCars116k)
and supports curriculum-style chunked training.

============================================================
STAGES
============================================================

1) DATA PREPARATION (external step or prior script)
   - Converts BoxCars116k .pkl + atlas data into YOLO format:
       images/
       labels/
       data.yaml

   - Optional tiling mode:
       * Sliding window augmentation for small objects
       * Increases dataset density for training

   - Outputs:
       boxcars_yolo/
           images/train|val|test
           labels/train|val|test
           data.yaml

============================================================
2) PHASE 1 — WARMUP
============================================================

- Runs only once (job_index == 0)
- Purpose:
    * Stabilize backbone + detection heads
    * Learn coarse representations

- Settings:
    * Freeze backbone layers (freeze=10)
    * Strong augmentation (mosaic, mixup, copy-paste)
    * Optimizer: Adam
    * LR: 1e-3
    * Epochs: warmup_epochs (~20 or 20% of total)

Output:
    runs/phase0001_warmup/

============================================================
3) PHASE 2 — CHUNKED FINE-TUNING
============================================================

- Splits remaining training into chunks:
    Example:
        total_epochs = 120
        warmup       = 20
        fine-tune    = 100
        chunk_size   = 40

    → chunk 0: epochs 20–59
    → chunk 1: epochs 60–99
    → chunk 2: epochs 100–119

- Each chunk:
    * Continues from previous checkpoint
    * Uses last.pt or best.pt from previous phase
    * Fine-tunes full model (freeze=0)

- Settings:
    * Optimizer: SGD
    * LR: 1e-4
    * Cosine LR decay enabled
    * Strong augmentation retained

Output:
    runs/phase0002_chunkX/

============================================================
4) MODEL BEHAVIOR
============================================================

- Base model: YOLOv8 (e.g., yolov8m.pt)
- Task: object detection (+ implicit classification via class IDs)
- Training includes:
    * Mosaic augmentation
    * MixUp
    * Copy-Paste augmentation
    * HSV jitter
    * Geometric transforms

============================================================
5) RESUME & CHECKPOINTING
============================================================

- Supports resume training via:
    --resume

- Each chunk loads:
    * previous chunk last.pt OR
    * warmup best.pt (for chunk 0)

============================================================
6) OUTPUTS
============================================================

Each run produces:
- best.pt (best model)
- last.pt (latest checkpoint)
- training logs
- evaluation metrics (YOLO default outputs)

============================================================
EXAMPLE USAGE
============================================================

# Warmup (chunk 0)
python train_y8_detcls_chunks.py \
    --data data.yaml \
    --model yolov8m.pt \
    --out runs_detcls \
    --total-epochs 120 \
    --chunk-size 60 \
    --chunk 0

# Fine-tune chunk 1
python train_y8_detcls_chunks.py \
    --data data.yaml \
    --model yolov8m.pt \
    --out runs_detcls \
    --total-epochs 120 \
    --chunk-size 60 \
    --chunk 1 \
    --resume
"""

import argparse, os, glob
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser("Chunked YOLOv8 Det+Cls Trainer")
    p.add_argument(
        "--data", "-d", required=True,
        help="dataset YAML (with train/val/test + nc/names)"
    )
    p.add_argument(
        "--model", "-m", default="yolov8m.pt",
        help="base YOLOv8 detection checkpoint"
    )
    p.add_argument(
        "--out", "-o", default="runs_detcls",
        help="root output directory"
    )
    p.add_argument(
        "--total-epochs", type=int, default=120,
        help="total epochs (warm-up + fine-tune)"
    )
    p.add_argument(
        "--chunk-size", type=int, default=60,
        help="fine-tune epochs per chunk"
    )
    p.add_argument(
        "--chunk", type=int, default=0,
        help="0-based chunk index"
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="image size"
    )
    p.add_argument(
        "--batch", type=int, default=16,
        help="batch size"
    )
    p.add_argument(
        "--resume", action="store_true",
        help="resume last.pt of this chunk"
    )
    return p.parse_args()

def last_lastpt(chunk_dir):
    w = Path(chunk_dir) / "weights" / "last.pt"
    return str(w) if w.exists() else None

def main():
    args = parse_args()
    ROOT = Path(args.out); ROOT.mkdir(exist_ok=True, parents=True)

    # compute warmup vs fine-tune
    WARM = min(20, args.total_epochs // 5)
    FT    = args.total_epochs - WARM
    CH    = args.chunk
    CS    = args.chunk_size

    # ─── Phase 1 Warm-Up ─────────────────────────────
    warm_dir = ROOT / "phase1_warmup"
    if CH == 0 and not args.resume:
        print(f"► Phase 1 warm-up for {WARM} epochs (freeze=10)…")
        model = YOLO(args.model, task="detect")
        model.train(
            data       = args.data,
            epochs     = WARM,
            imgsz      = args.imgsz,
            batch      = args.batch,
            freeze     = 10,           # freeze backbone+neck
            optimizer  = "Adam",
            lr0        = 1e-3,
            momentum   = 0.9,
            weight_decay=5e-4,
            mosaic     = True,
            mixup      = 0.5,
            copy_paste = 0.5,
            hsv_h      = 0.015,
            hsv_s      = 0.7,
            hsv_v      = 0.4,
            translate  = 0.1,
            scale      = 0.5,
            perspective= 0.003,
            fliplr     = 0.5,
            flipud     = 0.0,
            project    = str(ROOT),
            name       = "phase1_warmup",
            exist_ok   = True,
        )
#python output_clean/yolo_train.py --data boxCars/boxcars_yolo_detcls/data.yaml  --total-epochs 200 --chunk-size 20 --chunk 0 --resume

    # ensure warm-up produced best.pt
    best_w = warm_dir / "weights" / "best.pt"
    if not best_w.exists():
        raise RuntimeError("Warm-up best.pt not found; run chunk 0 first")

    # ─── Phase 2 Fine-Tune ────────────────────────────
    start_ft = CH * CS
    remaining= FT - start_ft
    run_e    = max(0, min(CS, remaining))
    if run_e <= 0:
        print(f"No fine-tune epochs left for chunk {CH}.") 
        return

    # pick seed weights
    if CH == 0:
        seed = str(best_w)
    else:
        prev_dir = ROOT / f"phase2_chunk{CH-1}"
        prev_w   = last_lastpt(prev_dir)
        if prev_w is None:
            raise RuntimeError(f"Missing last.pt for chunk {CH-1}")
        seed = prev_w

    name = f"phase2_chunk{CH}"
    print(f"► Phase 2 {name}: {run_e} epochs, seed={Path(seed).name}")

    model = YOLO(seed, task="detect")
    model.train(
        data       = args.data,
        epochs     = run_e,
        imgsz      = args.imgsz,
        batch      = args.batch,
        optimizer  = "SGD",
        lr0        = 1e-4,
        momentum   = 0.937,
        cos_lr     = True,
        lrf        = 0.2,
        freeze     = 0,
        mosaic     = True,
        mixup      = 0.5,
        copy_paste = 0.5,
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        translate  = 0.1,
        scale      = 0.5,
        perspective= 0.003,
        fliplr     = 0.5,
        flipud     = 0.0,
        project    = str(ROOT),
        name       = name,
        exist_ok   = True,
        resume     = args.resume,
    )

if __name__ == "__main__":
    main()

