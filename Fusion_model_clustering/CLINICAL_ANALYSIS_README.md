# Clinical Relevance Analysis

To generate the clinical analysis visualizations, run:

```bash
cd /home/ubuntu/Fusion_model/MuFFLe/Fusion_model_clustering
python3 create_clinical_analysis.py
```

The script will:
1. Load clusters from `analysis/clusters.csv`
2. Find clinical data (checks multiple locations automatically)
3. Merge the data and perform statistical analyses
4. Generate visualizations in `analysis/clinical_analysis/`

If the clinical data is not found, make sure it exists at one of these locations:
- `data/clinical_data.csv` (relative to script)
- `/home/ubuntu/Fusion_model/Fusion_model_clustering/data/clinical_data.csv`
- `/home/ubuntu/Fusion_model/MuFFLe/src/data/raw/clinical_data.csv`

The analysis generates:
- `progression_by_cluster.png` - Progression rates by cluster
- `BRS_distribution.png` - BRS category distribution
- `age_distribution.png` - Age distributions by cluster
- `EORTC_distribution.png` - EORTC risk category distribution  
- `clinical_relevance_summary.png` - Summary figure with statistics

