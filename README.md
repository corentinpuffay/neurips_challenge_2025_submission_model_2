# 🧠 NeurIPS EEG Foundation Challenge 2025 – Submission

This repository contains the code for a submission to the **NeurIPS 2025 EEG Foundation Challenge**.  
It implements an **end-to-end transformer-based model** for EEG decoding, performing both preprocessing and temporal modeling directly within PyTorch.  
The design replicates the MATLAB preprocessing pipeline in a differentiable, efficient form suitable for GPU inference.

---

## 🚀 Overview

The model, `EEGTransformerFull`, processes raw EEG signals sampled at **500 Hz** and produces per-trial predictions.  
It includes all preprocessing steps (high-pass filtering, referencing, resampling, normalization) followed by a transformer encoder for temporal feature extraction.  
Multiple model instances are combined via an ensemble (`SimpleEnsemble`) for improved robustness.

---

## 📦 Repository Structure


---

## ⚙️ Model Architecture

### EEGTransformerFull

An end-to-end EEG transformer model with embedded preprocessing and a lightweight transformer encoder.

#### 🔄 Preprocessing Pipeline

| Step | Description |
|------|--------------|
| **Channel selection** | Removes Cz and keeps 10 “interesting” channels. |
| **Common average reference (CAR)** | Subtracts the mean across channels to reduce spatial noise. |
| **High-pass filter** | 6th-order Butterworth filter (0.5 Hz cutoff, zero-phase `sosfiltfilt`). |
| **Resampling** | Downsamples from 500 Hz → 64 Hz using `scipy.signal.resample_poly`. |
| **Artifact suppression** | Applies `tanh(x / artifact_scale)` to limit extreme values. |
| **Z-score normalization** | Channel-wise normalization across time. |

#### 🧠 Transformer Encoder

- **Channel projection**: 1D convolution to project EEG channels into a `d_model`-dimensional feature space.  
- **Positional encoding**: standard sinusoidal encoding up to 3840 samples.  
- **Transformer encoder**: 1–2 layers (`nhead=1`, `dim_feedforward=32`), with dropout and LayerNorm.  
- **Pooling and output**: adaptive average pooling + linear head → scalar prediction.

#### ✨ Model Summary
- **Input**: `(batch_size, 128, timepoints)` EEG at 500 Hz  
- **Output**: `(batch_size, 1)` prediction (for classification or regression)
- **Default hyperparameters**:
  ```python
  d_model = 16
  nhead = 1
  num_layers = 1
  dropout = 0.3
  dim_feedforward = 32
