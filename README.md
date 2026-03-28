
# DCTformer: Frequency-to-Time-Domain Cyclic Learning for Coupling Noise Suppression in DAS Vertical Seismic Profile Data

<div align="center">

</div>

> **Paper under review** — Training code will be released upon acceptance.
> Currently, only **inference/testing code** is open-sourced.

---

## 📄 Citation

> **[Paper Title: DCTformer: Frequency-to-Time-Domain Cyclic Learning for Coupling Noise
Suppression in DAS Vertical Seismic Profile Data — will be updated upon acceptance]**
>
> *Abstract: (Distributed Acoustic Sensing (DAS) technology has been increasingly adopted
for Vertical Seismic Profile (VSP) data acquisition. However, coupling noise caused
by poor wellbore cementation severely degrades the data quality and hampers
subsequent interpretation. Coupling noise is particularly challenging to suppress
because it shares overlapping frequency bands, similar waveform morphology, and
comparable energy levels with signals, making it difficult for conventional methods to
distinguish between the two. To address this challenge, we propose DCTformer, a
deep learning architecture that employs a frequency-to-time-domain cyclic learning
strategy based on the Discrete Cosine Transform (DCT). DCTformer integrates a
DCT-domain frequency-convolution mixer (FDCM) for multi-scale spectral feature
extraction, a prior-guided fusion network (PGFN) with a residue channel prior (RCP)
for adaptive noise-aware gating, and a dual-domain loss function that jointly
optimizes the spatial and DCT domains to prevent signal energy leakage during
domain transformation. The core of DCTformer is a cyclic architecture in which each
processing block alternates between DCT-domain spectral feature extraction and
time-domain spatial structure preservation, enabling progressive refinement of noise- signal separation across scales. Tests on synthetic and field data from three distinct
regions demonstrate that DCTformer achieves an SNR of 10.98 dB, and generalizes
well across three regions’ DAS data.)*
>
> *(Shuhan Li，Qiankun Feng(corresponding author)，Yuanzhong Chen, Yue Li, 2026)*

The core model architecture is inspired by and built upon:

> **Efficient Frequency-Domain Image Deraining with Contrastive Regularization**
> *(please cite the original work if you use this codebase)*

---

## 🔍 Overview

This repository provides the **testing code** for our seismic noise attenuation method targeting **DAS-VSP (Distributed Acoustic Sensing Vertical Seismic Profile) coupling noise**. The model, **DCTformer**, replaces conventional Fourier-domain processing with a **2-D Discrete Cosine Transform (DCT)** backbone embedded in a U-Net-style encoder-decoder architecture, enabling robust signal preservation while suppressing coherent and random noise in seismic field data.

Key features:

- **DCT-domain global mixing** via `DCTUnit` for energy-compacted frequency representation
- **Fused local-global token mixer** combining depthwise convolutions with DCT-domain attention
- **Prior-Gated Feed-forward Network** that leverages noise residual priors at training time
- **SEGY I/O support** for direct integration with standard seismic processing workflows

---

## 🖼️ Results

### Synthetic / Simulated Data

<!-- Replace the placeholder below with your actual result figure -->
*Synthetic data testing results will be shown here. A simulated test dataset is provided in `data/synthetic/` — see [Quick Start](#-quick-start).*

| Input (Noisy) | Predicted (Clean) | Residual (Noise) |
|:---:|:---:|:---:|
| *(figure placeholder)* | *(figure placeholder)* | *(figure placeholder)* |

### Field Data (DASVSP)

> ⚠️ **Note:** Field seismic data cannot be publicly released due to confidentiality agreements with the oil & gas company that provided the data. The experimental results on field data are shown below for reference only.

<!-- Add your field data result figures here -->

**Figure 1 — Field data denoising example:**

*![实际数据1](figures/result_comparison.png)*

**Figure 2 — Frequency spectrum analysis:**

*![实际数据2](figures/result_comparison.png)*

**Figure 3 — Field data denoising example:**

*![实际数据3](figures/result_comparison.png)*

---

## 🗂️ Repository Structure

```
DCTformer/
├── model.py                  # DCTformer model definition
├── denoise_test.py           # Inference / testing script
├── utils/
│   ├── GetPatches.py         # SEGY I/O: read_segy_data, create_segy
│   └── Cut_combine.py        # Patch cutting and Gaussian recombination
├── data/
│   ├── synthetic/            # Simulated test data (provided)
│   │   └── synthetic_test.sgy
│   └── sgy_data/             # Directory for your own SEGY data (not included)
├── model/
│   └── save/                 # Place your pre-trained weights here
│       └── latest_model_epoch99.pth   # (download separately — see below)
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

**Python 3.8+** is recommended. Install all dependencies via:

```bash
pip install -r requirements.txt
```

Core dependencies:

```
torch>=1.12.0
torchvision
numpy
matplotlib
segyio
torch_dct
```

> **GPU**: A CUDA-compatible GPU is strongly recommended. The code auto-detects GPU availability and falls back to CPU if unavailable. On CPU, large SEGY files may be slow.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DCTformer.git
cd DCTformer
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download pre-trained weights

> Pre-trained model weights (`latest_model_epoch99.pth`) can be downloaded from:
> **[Google Drive / Zenodo / BaiduNetDisk link — TBD]**

Place the downloaded weights at:

```
model/save/latest_model_epoch99.pth
```

### 4. Run inference on the provided synthetic data

```bash
python denoise_test.py
```

By default, the script reads from `data/synthetic/synthetic_test.sgy`. Edit the path variables at the top of the `__main__` block in `denoise_test.py` to point to your own data:

```python
data_path  = 'data/synthetic/synthetic_test.sgy'   # Input SEGY
model_path = 'model/save/latest_model_epoch99.pth'  # Pre-trained weights
save_path  = 'results/output_denoised.sgy'          # Output: denoised
save_path_noisepre = 'results/output_noise.sgy'     # Output: separated noise
```


---



## 📝 Notes for Users

- **SEGY convention**: The code reads and writes standard SEGY format via `segyio`. Make sure your input SEGY traces are stored as `[n_traces × n_samples]` (i.e., the data matrix is `[width × height]`).
- **Normalization**: Data is automatically normalized to `[-1, 1]` before inference and rescaled back afterward. No manual preprocessing is needed.
- **Large files**: For very large SEGY files, the Gaussian-weighted recombination (`combine_gaussian`) ensures seamless stitching of overlapping patches. Overlap is controlled by `stride_x` and `stride_y` — smaller strides yield smoother results at the cost of more computation.

---

## 🔒 Training Code

> **Training code is not yet publicly available.**
> It will be released here upon paper acceptance.


---

## 📬 Contact

If you have questions, please open a GitHub Issue or contact:

**[Qiankun Feng]** — [fengqk@jlu.edu.cn]

---

## 📃 License

> The field seismic data used in this paper is proprietary and subject to a confidentiality agreement with the data provider. Only the synthetic test dataset included in this repository can be freely used and redistributed.
