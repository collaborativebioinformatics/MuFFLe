# Brev Cloud Setup Guide for CHIMERA Task 3

## 📊 Compute Requirements Summary

Based on the analysis of the pipeline:

### Memory Requirements
| Metric | Value |
|--------|-------|
| Peak GPU Memory (inference) | **~1.2 GB** |
| Peak CPU RAM | **~2 GB** |
| Average WSI per patient | ~730 MB |
| Max WSI per patient | ~1 GB |
| Final signature matrix | < 1 MB |

### Storage Requirements
| Data Type | Size |
|-----------|------|
| WSI Features (176 patients) | **~160 GB** |
| RNA Embeddings | ~0.2 MB |
| Coordinates | ~2.3 GB |
| **Total raw data** | **~165 GB** |
| Output files | < 100 MB |

### Processing Time Estimate
- Per patient: ~2-5 seconds (CPU) / ~1-2 seconds (GPU)
- Full cohort (176 patients): ~10-15 minutes
- Including visualization: ~20 minutes total

---

## 🖥️ Recommended Brev Instance

### Option 1: CPU-Only (Cheapest) ✅ RECOMMENDED
Since this is an **inference-only** pipeline with no training, CPU is sufficient:

| Instance | Specs | Monthly Cost |
|----------|-------|--------------|
| **n1-standard-4** | 4 vCPU, 15 GB RAM | ~$100/mo |
| **n1-standard-8** | 8 vCPU, 30 GB RAM | ~$200/mo |

**Why CPU works:**
- No neural network training (no backpropagation)
- Operations are: variance computation, softmax, matrix multiply, concatenation
- HDBSCAN clustering is CPU-bound anyway
- Lifelines survival analysis is CPU-only

### Option 2: GPU (Faster, more expensive)
If you want faster processing or future training:

| Instance | GPU | VRAM | Monthly Cost |
|----------|-----|------|--------------|
| **g4dn.xlarge** | T4 | 16 GB | ~$150/mo |
| **g5.xlarge** | A10G | 24 GB | ~$300/mo |

### Storage
- Use at least **200 GB** disk for raw data + buffer
- Consider attaching an external volume for data persistence

---

## 🚀 Quick Setup Commands

### 1. Clone/Upload Project
```bash
# If uploading from local
scp -r "Fusion model" user@brev-instance:/home/user/

# Or git clone your repo
git clone <your-repo-url>
cd "Fusion model"
```

### 2. Create Environment
```bash
# Create conda environment
conda create -n chimera python=3.10 -y
conda activate chimera

# Install dependencies
pip install -r requirements.txt
```

### 3. Download Data from S3
```bash
# Run the download script (public bucket, no credentials needed)
cd "Fusion model"
python utility/download_image_embeddings.py

# This downloads all 176 patients (~160 GB)
# Takes approximately 30-60 minutes depending on bandwidth
```

### 4. Run Pipeline
```bash
# Full pipeline
python run_pipeline.py

# With options
python run_pipeline.py --verbose --min-cluster-size 5
```

---

## 📁 Expected Directory Structure After Setup

```
Fusion model/
├── data/
│   ├── clinical_data.csv          ✓ Already present
│   ├── rna_embeddings/            ✓ Already present (176 files)
│   └── wsi_data/
│       ├── coordinates/           ✓ Already present (176 files)
│       └── features/              📥 Download required (176 files, ~160GB)
│
├── model/                         ✓ Pipeline code
├── processing/                    📤 Generated output
├── analysis/                      📤 Generated output
└── run_pipeline.py               ✓ Entry point
```

---

## 🔧 Environment Variables

Add to your `.bashrc` or `.zshrc` for convenience:

```bash
# Prevent OpenMP conflicts (common on conda)
export KMP_DUPLICATE_LIB_OK=TRUE

# Optional: Limit threads for memory efficiency
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Optional: CUDA settings (if using GPU)
export CUDA_VISIBLE_DEVICES=0
```

---

## ⚡ Performance Tips

1. **Memory Management**: The pipeline processes one patient at a time to minimize memory usage
2. **Skip visualizations** for faster runs: `python run_pipeline.py --skip-visualization`
3. **Parallel downloads**: The download script uses 4 workers by default
4. **Check progress**: Monitor with `ls -la data/wsi_data/features/ | wc -l`

---

## 🐛 Troubleshooting

### "No module named 'torch'"
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu  # For CPU-only
# or
pip install torch  # For GPU
```

### OpenMP Error
```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

### S3 Download Timeout
```bash
# Increase timeout in download script or use AWS CLI directly:
aws s3 sync s3://chimera-challenge/v2/task3/features/ data/wsi_data/ --no-sign-request
```

### Out of Memory
- Reduce `--min-cluster-size` to process fewer patients
- Use `--skip-visualization` to skip memory-intensive plots
- Process patients in batches by specifying `--patient-ids`

---

## ✅ Verification Checklist

Before running the full pipeline, verify:

- [ ] All dependencies installed: `pip list | grep -E "torch|hdbscan|lifelines"`
- [ ] Clinical data present: `ls data/clinical_data.csv`
- [ ] RNA embeddings present: `ls data/rna_embeddings/ | wc -l` (should be 176)
- [ ] WSI features downloaded: `ls data/wsi_data/features/ | wc -l` (should be 176)
- [ ] Coordinates present: `ls data/wsi_data/coordinates/ | wc -l` (should be 176)

Quick test:
```bash
python -c "from model import run_pipeline; print('✅ All imports work!')"
```

