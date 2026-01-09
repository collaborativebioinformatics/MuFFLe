# CHIMERA Task 3: Unsupervised Multimodal Fusion Pipeline

> **Inference-Only Risk Stratification** for Bladder Cancer Patients using WSI Histopathology + RNA Transcriptomics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

This pipeline stratifies 176 bladder cancer patients into discrete recurrence risk groups by fusing:
- **WSI Features**: Variable-sized patch embeddings (N × 1024) from UNI encoder
- **RNA Embeddings**: Pre-computed transcriptomics signatures (256-d)

**Key Principle**: **No Training Required** - The pipeline uses self-normalizing heuristics for attention and density-based clustering for stratification.

## 📁 Project Structure

```
Fusion model/
├── data/
│   ├── clinical_data.csv           # Patient clinical outcomes
│   ├── rna_embeddings/             # 256-d RNA embeddings (.pt files)
│   └── wsi_data/
│       ├── coordinates/            # Patch spatial coordinates (.npy)
│       └── features/               # N×1024 UNI embeddings (.pt)
│
├── model/
│   ├── __init__.py                 # Package exports
│   ├── config.py                   # Configuration dataclasses
│   ├── data_loader.py              # Local/S3 data loading
│   ├── unsupervised_fusion.py      # Core pipeline logic
│   └── visualization.py            # Plots and survival analysis
│
├── processing/
│   ├── patient_signatures/         # Fused 1280-d vectors
│   └── attention_results/          # Per-patient attention weights
│
├── analysis/
│   ├── clusters.csv                # Patient → Cluster mapping
│   ├── survival_plots/             # Kaplan-Meier curves
│   └── attention_heatmaps/         # Spatial attention visualizations
│
├── run_pipeline.py                 # Main entry point
├── download_wsi_data.py            # S3 data download utility
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd "Fusion model"
pip install -r requirements.txt
```

### 2. Download WSI Features (if not already present)

```bash
# Download all WSI features from S3 (public bucket)
python download_wsi_data.py --all

# Or download specific patients
python download_wsi_data.py --patients 3A_001 3A_002 3A_003
```

### 3. Run the Pipeline

```bash
# Basic execution
python run_pipeline.py

# With options
python run_pipeline.py --min-cluster-size 5 --verbose
```

## 🔬 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL FUSION PIPELINE                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   WSI Patches (N × 1024)                                            │
│         │                                                            │
│         ▼                                                            │
│   ┌─────────────────────────────┐                                   │
│   │  Gated Attention Pooling    │  Weight_i = σ(var) · tanh(mean)   │
│   │  (Heuristic, No Training)   │                                   │
│   └─────────────┬───────────────┘                                   │
│                 │                                                    │
│                 ▼                                                    │
│         Slide Embedding (1024-d)                                     │
│                 │                                                    │
│                 ├──────────────────────────────────┐                │
│                 │                                  │                │
│                 ▼                                  ▼                │
│         ┌───────────────┐                  ┌───────────────┐       │
│         │  Z-Score WSI  │                  │  Z-Score RNA  │       │
│         └───────┬───────┘                  └───────┬───────┘       │
│                 │                                  │                │
│                 └──────────────┬───────────────────┘                │
│                                │                                    │
│                                ▼                                    │
│                    Concatenation (1280-d)                           │
│                                │                                    │
│                                ▼                                    │
│                    ┌───────────────────────┐                        │
│                    │       HDBSCAN         │                        │
│                    │  Density Clustering   │                        │
│                    └───────────┬───────────┘                        │
│                                │                                    │
│                                ▼                                    │
│                    Risk Clusters + Survival Validation              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 📊 Key Components

### 1. WSI Aggregation (Variance-Weighted Attention)

High-variance patches represent complex morphologies (tumor nests, pleomorphism):

```python
# Pseudo-code
patch_variance = torch.var(features, dim=1)  # [N]
attention_weights = softmax(patch_variance)   # [N]
slide_embedding = sum(features * weights)     # [1024]
```

### 2. Gated Attention (Alternative)

More nuanced attention using tanh-sigmoid gating:

```python
tanh_branch = tanh(patch_mean / std)
sigmoid_branch = sigmoid((variance - mean) / std)
attention = softmax(tanh_branch * sigmoid_branch)
```

### 3. Multimodal Fusion

Dimensionality-aligned concatenation with cohort-level normalization:

```python
wsi_normalized = StandardScaler().fit_transform(wsi_embeddings)  # [n, 1024]
rna_normalized = StandardScaler().fit_transform(rna_embeddings)  # [n, 256]
signatures = concatenate([wsi_normalized, rna_normalized])        # [n, 1280]
```

### 4. Clustering (HDBSCAN)

Density-based clustering that automatically:
- Determines optimal number of clusters
- Identifies outlier patients (noise)
- Handles varying cluster densities

## 📈 Output Files

| File | Description |
|------|-------------|
| `clusters.csv` | Patient ID → Cluster label mapping |
| `signatures.npy` | Final 1280-d patient signature matrix |
| `*_attention.npz` | Per-patient attention weights and top-k indices |
| `kaplan_meier_curves.png` | Survival curves with log-rank p-value |
| `signature_tsne.png` | t-SNE visualization of patient signatures |
| `*_attention_heatmap.png` | Spatial attention maps per patient |

## 🔧 Configuration

Edit `model/config.py` or pass CLI arguments:

```python
# Clustering
min_cluster_size = 5        # Minimum patients per cluster
fallback_n_clusters = 3     # KMeans fallback

# Features
top_k_attention_percent = 0.01  # Top 1% patches for heatmaps

# Survival Analysis
significance_level = 0.05   # Log-rank test threshold
```

## 📋 Command Line Options

```bash
python run_pipeline.py [OPTIONS]

Options:
  --use-variance-attention    Use simple variance (default: gated attention)
  --min-cluster-size INT      HDBSCAN min_cluster_size (default: 5)
  --n-clusters INT            KMeans fallback clusters (default: 3)
  --s3-fallback               Enable S3 for missing local data
  --patient-ids ID [ID ...]   Process specific patients only
  --skip-visualization        Skip plot generation
  --output-dir PATH           Override output directory
  -v, --verbose               Verbose logging
  -q, --quiet                 Suppress output
```

## 🧪 Validation

The pipeline validates clusters against clinical outcomes:

1. **Kaplan-Meier Curves**: Visual separation of survival probabilities
2. **Log-Rank Test**: Statistical significance (p < 0.05)
3. **Concordance Index**: C-index > 0.65 indicates clinical utility

## 🔍 Interpretability

### Attention Heatmaps

The pipeline generates spatial attention maps showing which tissue regions drove the classification:

```python
from model import AttentionHeatmapGenerator, get_config

heatmap_gen = AttentionHeatmapGenerator(get_config())
heatmap_gen.generate_heatmap("3A_001")  # Patient ID
```

### Biological Validation

High-attention patches should localize to:
- Tumor nests
- Areas of high pleomorphism  
- Regions with significant morphological features

**NOT** to:
- Background stroma
- Artifact regions
- Outside the tissue mask

## 🌐 Cloud Compute

For running on cloud instances with S3 data:

```bash
# Enable S3 fallback (public bucket)
python run_pipeline.py --s3-fallback

# Or set environment variables for private buckets
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

## 📚 Citation

If you use this pipeline, please cite:

```bibtex
@software{chimera_fusion_2024,
  title={Unsupervised Multimodal Fusion for Bladder Cancer Stratification},
  author={CHIMERA Challenge Team},
  year={2024},
  url={https://github.com/...}
}
```

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

---

**Questions?** Open an issue or contact the maintainers.

