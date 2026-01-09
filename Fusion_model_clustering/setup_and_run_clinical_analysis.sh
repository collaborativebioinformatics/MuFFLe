#!/bin/bash
# Setup and run clinical analysis

cd /home/ubuntu/Fusion_model/MuFFLe/Fusion_model_clustering

echo "Installing required packages..."
pip3 install pandas numpy matplotlib seaborn scipy --quiet

echo "Running clinical analysis..."
python3 create_clinical_analysis.py

echo "Done! Check analysis/clinical_analysis/ for results."

