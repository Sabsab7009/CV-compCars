# Vehicle Recognition Framework 

This repository contains a unified deep learning framework for vehicle **detection and fine-grained classification**, designed for large-scale datasets such as **BoxCars116k** and surveillance-based imagery.

The project supports multiple architectures and a consistent training protocol based on **curriculum learning (warm-up + chunked fine-tuning)**.

---

# 1. Project Overview

The framework is divided into two main tasks:

## A. Vehicle Classification
Fine-grained classification of vehicles (e.g., make/model or class ID).

Supported models:
- EfficientNetV2-S
- ConvNeXt
- MobileNetV3
- DeiT (Vision Transformer)

All models use the same training philosophy:
- Warm-up phase (optional backbone freezing)
- Chunked fine-tuning
- Mixed precision training (AMP)
- Standardized evaluation pipeline

---

## B. Vehicle Detection (YOLOv8)
Object detection pipeline for vehicles with optional classification labels.

Features:
- YOLOv8-based detector


---

# 2. Training Strategy

All classification models follow the same 2-stage curriculum:

## Phase 1 — Warm-up
- Backbone partially or fully frozen
- Higher learning rate
- Strong augmentation
- Purpose: stabilize feature extraction

## Phase 2 — Chunked Fine-Tuning
- Full model training
- Lower learning rate
- Cosine or step LR schedule
- Training split into chunks for long runs

## Full Documentation

Complete thesis with methodology, results, and evaluation: 
See `reports/Traffic_Carbon_Footprint_Thesis.pdf`

