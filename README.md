# MUSE-Net

**Uncertainty-Aware Dynamic Routing and Continuous-Time State Evolution for Cross-Modal RGB-Event Segmentation**

Official PyTorch implementation of the paper accepted by **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2026**.

> Zhuoxian Li, Hengyi Lv, Ming Sun, Yisa Zhang, Yang Feng, and Junyao Sang.
> DOI: [10.1109/TCSVT.2026.3716495](https://doi.org/10.1109/TCSVT.2026.3716495)

MUSE-Net establishes a systematic new paradigm for robust cross-modal RGB-event semantic segmentation. It breaks down the representation gap between dense RGB frames and asynchronous sparse event streams across three interconnected physical dimensions:

1. **Reliability** — *Uncertainty-Aware Dynamic Routing (UADR)* acts as a circuit breaker against modality-specific noise (RGB low-light blur, event static unreliability).
2. **Temporal Dynamics** — *Cross-Modal State Evolution (CM-SE)* replaces quadratic static attention (O(L²)) with an ODE-inspired physical evolution (O(L)) driven by an event-based differential control gate.
3. **Spatial Topology** — *Sparse-Dense Graph Synergistic Representation (SDGSR)* recovers the high-frequency micro-edges lost during voxelization.

MUSE-Net achieves **79.86% mIoU on DDD17** and **76.29% mIoU on DSEC-Semantic**, with absolute mIoU improvements of **+3.30%** and **+1.65%** over the previous SOTA, while running at **216.86 FPS** with only **0.74 GB** peak GPU memory on an RTX 3090.

---

## Method

<p align="center"><i>Architecture and module details are described in the paper (Fig. 2–6).</i></p>

MUSE-Net adopts an asymmetric dual-stream hierarchical encoder-decoder built on pre-trained SegFormer backbones: **MiT-B2** for the RGB (image) branch and lightweight **MiT-B0** for the event branch.

The raw event stream is first processed by two event representation modules:
- **DSEM** — *Density-guided Spatiotemporal Event Modulation*: treats the event activity map as a spatial density prior, applies a large-kernel depth-wise density extraction, and conditionally modulates the event voxel grid via a zero-initialized affine transformation (scale γ / shift β), followed by temporal-bin mixing with 1-D convolutions.
- **SDGSR** — *Sparse-Dense Graph Synergistic Representation*: builds an implicit directed 3×3 local graph (K = 9) on the event feature map, defines directed edge features as exact state differences `[x_q − x_p, x_p]`, and performs group graph convolution with mean aggregation to restore local geometric topology. Fixed-window unfolding keeps strictly linear O(N) complexity.

At each of the four hierarchical stages, features enter the **UADR** reliability gate, then a **Multi-dimensional Synergistic Fusion Engine (MSFE)**:

| Dimension | Module | Role |
|---|---|---|
| Spatial | **Deformable Cross-Attention** | Event queries predict 2-D sampling offsets to align RGB textures for fine-grained spatial calibration |
| Temporal | **Cross-Modal State Evolution (CM-SE)** | Formulates the event stream as a data-driven differential control gate Δt within a continuous-time ODE (`dh/dt = −h + x_evt`, ZOH discretization), driving the RGB state evolution with linear O(L) complexity |
| Frequency | **Cross-Modal Spectral Fusion** | Extracts FFT amplitude spectra from both modalities as an illumination-invariant global structural gate |

The multi-scale fused features pass through an **MLP decoder** (SegFormer head). During training, a **Cross-Modal Semantic Contrastive Alignment (CM-SECA)** loss (InfoNCE, temperature 0.1) is applied **only at Stage 4** to align high-level abstract semantics. Finally, an **Event Boundary Refinement Module (EBRM)** uses high-resolution low-level event features to generate class-specific boundary gates (zero-initialized) that sharpen the predicted contours.

> **Note on naming:** the model entry-point class is kept as `EGHFNet` for backward compatibility with existing checkpoints; it implements MUSE-Net.

---

## Datasets

### DSEC-Semantic — 11 classes

Built on the large-scale [DSEC](https://dsec.ifi.uzh.ch/) dataset with the official 11-class semantic annotation (road, sidewalk, vehicle, pedestrian, etc.). Event camera + standard RGB camera, covering driving scenes from intense sunlight to nighttime low-light.

- Effective input resolution **640 × 440**: the left RGB image is perspective-warped to the left event camera frame using extrinsic/intrinsic calibration, then the bottom 40 pixel rows (not covered by the frame sensor) are cropped.
- Event encoding: **5 time bins** → 5-channel spatiotemporal voxel grid + 5-channel activity map = **10-channel** composite input.
- Preprocess with `tools/preprocess_align.py` (edit `DSEC_ROOT` / `OUTPUT_ROOT` first). Each sample `.pt` contains `rgb` `[3, 440, 640]`, `event` `[10, 440, 640]`, `mask` `[440, 640]`.

### DDD17 — 6 classes

Recorded by the DAVIS346B sensor (grayscale APS images + asynchronous events, native resolution 346 × 260). We adopt pixel-level pseudo-labels with **6 semantic categories** (DDD17-Seg).

- Effective input resolution **346 × 200**: the bottom 60 pixel rows (vehicle hood) are cropped.
- Grayscale APS images are duplicated to 3 channels to match the pre-trained backbone input.
- Event encoding: **3 time bins** → 6-channel composite tensor (voxel + activity).
- Preprocess with `tools/preprocess_ddd17.py` (edit the paths in `DATA_PATHS` / `OUTPUT_ROOT` first). Each sample `.pt` contains `rgb`, `event`, `mask`, `filename`.

Point the loaders at the preprocessed data via `PREPROCESSED_ROOT` in `configs/config.py` (DSEC) and `configs/config_ddd17.py` (DDD17).

---

## Installation

```bash
conda create -n musenet python=3.10 -y
conda activate musenet
pip install -r requirements.txt
```

Tested with PyTorch ≥ 2.0 on a single NVIDIA GPU (RTX 3090, `cuda:0`).

### Pretrained backbones

MUSE-Net starts from the SegFormer ImageNet weights:

- RGB backbone (MiT-B2) → `./pretrained/mit_b2.pth`
- Event backbone (MiT-B0) → `./pretrained/mit_b0.pth`

Both are loaded with `strict=False`, so training also works from scratch. Put the `.pth` files under `./pretrained/` (already git-ignored).

---

## Training

Training uses AdamW (weight decay 0.01), a polynomial LR decay with 1500-step linear warmup, base learning rate 2 × 10⁻⁴, and 100 epochs on both datasets, with mixed precision (AMP) and gradient accumulation. The primary loss is cross-entropy (OHEM, threshold 0.7, for DSEC) plus `0.1 × L_CM-SECA` computed at Stage 4. Synchronous joint augmentation (random multi-scale 0.5×–2.0×, random crop, horizontal flip, Copy-Paste) is applied to both modalities.

### DSEC

```bash
python train.py            # train from scratch
python train.py --resume   # resume from latest_checkpoint.pth
```

### DDD17

```bash
python train_ddd17.py            # train from scratch
python train_ddd17.py --resume   # resume
```

Checkpoints are saved under `checkpoints/<suffix>/`:
- `latest_checkpoint.pth` — every epoch (resume point, includes optimizer/scheduler/scaler states).
- `best_miou_checkpoint.pth` — best validation mIoU.

TensorBoard logs → `logs/`, text logs → `txt_logs/`. Key hyper-parameters (batch size, image size, class count, time bins) are set per dataset in the corresponding `configs/*.py`.

---

## Evaluation

Evaluation runs inference on the test split, computes global mIoU and pixel accuracy, and saves per-sequence colorized predictions under `runs/`.

### DSEC

```bash
python test.py                                   # uses best_miou_checkpoint.pth
python test.py --ckpt /path/to/checkpoint.pth    # explicit checkpoint
```

### DDD17

```bash
python test_ddd17.py                                   # uses best_miou_checkpoint.pth
python test_ddd17.py --ckpt /path/to/checkpoint.pth    # explicit checkpoint
```

---

## Results

### Comparison with state-of-the-art (paper Table I)

| Method | Modality | Event Rep. | DDD17 mIoU (%) | DDD17 Acc (%) | DSEC mIoU (%) | DSEC Acc (%) |
|---|---|---|---|---|---|---|
| SegFormer-B2 | Image | – | 71.05 | 95.73 | 71.99 | 94.97 |
| SegNeXt-B | Image | – | 71.46 | 95.97 | 71.55 | 94.89 |
| EV-SegNet | Event | 6-ch Image | 54.81 | 89.76 | 51.76 | 88.61 |
| ESS | Event | Voxel Grid | 61.37 | 91.08 | 51.57 | 89.25 |
| EDCNet-S2D | E + I | Voxel Grid | 61.99 | 93.80 | 56.75 | 92.39 |
| HALSIE | E + I | Voxel Grid | 60.66 | 92.50 | 52.43 | 89.01 |
| CMX | E + I | Voxel Grid | 71.88 | 95.64 | 72.42 | 95.07 |
| CMNeXt | E + I | Voxel Grid | 72.67 | 95.74 | 72.54 | 95.10 |
| EISNet | E + I | AET | 75.03 | 96.04 | 73.07 | 95.12 |
| EIFNet | E + I | AEFRM | 76.56 | 96.18 | 74.64 | 95.61 |
| MambaSeg | E + I | Voxel Grid | 77.56 | 96.33 | 75.10 | 95.71 |
| **Ours (MUSE-Net)** | E + I | DSEM + SDGSR | **79.86** | **96.86** | **76.29** | **95.71** |

### Efficiency (paper Table X, RTX 3090, 346 × 200, FP16)

| Method | Backbone | Params (M) | MACs (G) | Mem (GB) | FPS | mIoU (%) |
|---|---|---|---|---|---|---|
| CMX | 2×MiT-B2 | 66.56 | 16.29 | 1.51 | 191.63 | 71.88 |
| CMNeXt | PPX + MiT-B2 | 58.68 | 16.32 | 1.20 | 192.92 | 72.67 |
| EISNet | MiT-B0 + B2 | 34.39 | 17.30 | 1.58 | 135.21 | 75.03 |
| MambaSeg | Mamba | 25.44 | 15.59 | 1.64 | 211.49 | 77.56 |
| **Ours (MUSE-Net)** | MiT-B0 + B2 | 47.39 | **14.09** | **0.74** | **216.86** | **79.86** |

Ablation studies (constructive step-by-step, leave-one-out, MSFE internal synergy, temporal mechanisms, routing strategies, contrastive levels, SDGSR graph construction, time-bin selection) are reported in paper Tables II–IX.

---

## Repository Structure

```
.
├── configs/
│   ├── config.py           # DSEC configuration
│   └── config_ddd17.py     # DDD17 configuration
├── network/
│   ├── backbone.py         # Dual SegFormer (MiT-B2 RGB / MiT-B0 event) backbones
│   ├── modules.py          # Reusable building blocks (DSEM, SDGSR, UADR, MSFE branches, EBRM)
│   ├── muse_net.py         # Shared MUSE-Net architecture
│   ├── model.py            # DSEC model entry point (EGHFNet)
│   └── model_DDD17.py      # DDD17 model entry point (EGHFNet)
├── tools/
│   ├── loader.py           # DSEC dataloader
│   ├── loader_DDD17.py     # DDD17 dataloader
│   ├── loss.py             # OHEM CE + CM-SECA contrastive loss
│   ├── lovasz_losses.py    # Lovász-Softmax (kept for compatibility)
│   ├── utils.py            # mIoU confusion matrix, color maps, event encoding (AET)
│   ├── preprocess.py       # DSEC sample preprocessors
│   ├── preprocess_align.py # DSEC calibration & preprocessing (→ 640 × 440 .pt samples)
│   └── preprocess_ddd17.py # DDD17 preprocessing (→ 346 × 200 .pt samples)
├── train.py                # DSEC training
├── test.py                 # DSEC evaluation
├── train_ddd17.py          # DDD17 training
├── test_ddd17.py           # DDD17 evaluation
├── requirements.txt
└── README.md
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026muse,
  title   = {Uncertainty-Aware Dynamic Routing and Continuous-Time State Evolution
             for Cross-Modal RGB-Event Segmentation},
  author  = {Li, Zhuoxian and Lv, Hengyi and Sun, Ming and Zhang, Yisa and Feng, Yang and Sang, Junyao},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  year    = {2026},
  publisher = {IEEE},
  doi     = {10.1109/TCSVT.2026.3716495}
}
```

---

## Acknowledgement

This work was supported by the Jilin Provincial Department of Science and Technology under Grant 20250201053GX.
