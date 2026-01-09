#!/bin/bash
# =============================================================================
# CHIMERA Task 3 - Brev Cloud Setup Script
# =============================================================================
# This script sets up the complete environment on a Brev cloud instance
#
# Usage:
#   chmod +x setup_brev.sh
#   ./setup_brev.sh
#
# Requirements:
#   - Brev instance with at least 200GB disk space
#   - Internet connection for downloading dependencies and data
# =============================================================================

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     CHIMERA TASK 3 - BREV CLOUD SETUP                                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "📁 Working directory: $SCRIPT_DIR"
echo ""

# =============================================================================
# Step 1: Check Python version
# =============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Checking Python environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "${GREEN}✓${NC} Found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python3 not found. Please install Python 3.9+"
    exit 1
fi

# =============================================================================
# Step 2: Create virtual environment (optional)
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Setting up Python environment..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Conda detected. Creating conda environment..."
    
    # Create environment if it doesn't exist
    if ! conda env list | grep -q "chimera"; then
        conda create -n chimera python=3.10 -y
        echo -e "${GREEN}✓${NC} Created conda environment 'chimera'"
    else
        echo -e "${GREEN}✓${NC} Conda environment 'chimera' already exists"
    fi
    
    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate chimera
    echo -e "${GREEN}✓${NC} Activated conda environment"
else
    # Use venv instead
    echo "Conda not found, using venv..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo -e "${GREEN}✓${NC} Created virtual environment"
    else
        echo -e "${GREEN}✓${NC} Virtual environment already exists"
    fi
    
    source venv/bin/activate
    echo -e "${GREEN}✓${NC} Activated virtual environment"
fi

# =============================================================================
# Step 3: Install dependencies
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Installing Python dependencies..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Upgrade pip
pip install --upgrade pip --quiet

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}✓${NC} Installed dependencies from requirements.txt"
else
    echo -e "${YELLOW}!${NC} requirements.txt not found, installing core packages..."
    pip install torch numpy pandas scikit-learn hdbscan lifelines matplotlib seaborn boto3 tqdm --quiet
fi

# Verify critical imports
python3 -c "import torch; import hdbscan; import lifelines; print('✓ All critical packages imported successfully')"

# =============================================================================
# Step 4: Set environment variables
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Setting environment variables..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

echo -e "${GREEN}✓${NC} Set KMP_DUPLICATE_LIB_OK=TRUE"
echo -e "${GREEN}✓${NC} Set thread limits (OMP_NUM_THREADS=4, MKL_NUM_THREADS=4)"

# Add to bashrc for persistence
if ! grep -q "KMP_DUPLICATE_LIB_OK" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# CHIMERA Task 3 environment" >> ~/.bashrc
    echo "export KMP_DUPLICATE_LIB_OK=TRUE" >> ~/.bashrc
    echo "export OMP_NUM_THREADS=4" >> ~/.bashrc
    echo "export MKL_NUM_THREADS=4" >> ~/.bashrc
    echo -e "${GREEN}✓${NC} Added environment variables to ~/.bashrc"
fi

# =============================================================================
# Step 5: Check data status
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 5: Checking data availability..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check directories
mkdir -p data/wsi_data/features
mkdir -p data/wsi_data/coordinates
mkdir -p data/rna_embeddings
mkdir -p processing/patient_signatures
mkdir -p processing/attention_results
mkdir -p analysis/survival_plots
mkdir -p analysis/attention_heatmaps

echo -e "${GREEN}✓${NC} Created output directories"

# Count existing files
CLINICAL_EXISTS=$([ -f "data/clinical_data.csv" ] && echo "✓" || echo "✗")
RNA_COUNT=$(ls -1 data/rna_embeddings/*.pt 2>/dev/null | wc -l || echo 0)
WSI_COUNT=$(ls -1 data/wsi_data/features/*.pt 2>/dev/null | wc -l || echo 0)
COORD_COUNT=$(ls -1 data/wsi_data/coordinates/*.npy 2>/dev/null | wc -l || echo 0)

echo ""
echo "📊 Data Status:"
echo "   Clinical data:     $CLINICAL_EXISTS"
echo "   RNA embeddings:    $RNA_COUNT / 176"
echo "   WSI features:      $WSI_COUNT / 176"
echo "   Coordinates:       $COORD_COUNT / 176"

# =============================================================================
# Step 6: Download missing WSI data
# =============================================================================
if [ "$WSI_COUNT" -lt 176 ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Step 6: Downloading WSI features from S3..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo -e "${YELLOW}⚠️  This will download approximately 160 GB of data.${NC}"
    echo "   Estimated time: 30-60 minutes (depending on bandwidth)"
    echo ""
    read -p "Do you want to download now? [y/N]: " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting download..."
        # Auto-confirm in non-interactive mode
        echo "y" | python3 utility/download_image_embeddings.py --workers 8
    else
        echo -e "${YELLOW}!${NC} Skipping download. Run later with:"
        echo "   python utility/download_image_embeddings.py"
    fi
else
    echo -e "${GREEN}✓${NC} All WSI features already downloaded"
fi

# =============================================================================
# Step 7: Verify installation
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 7: Verifying installation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -c "
import sys
sys.path.insert(0, '.')
try:
    from model import run_pipeline, get_config
    config = get_config()
    print('✓ Pipeline imports successful')
    print(f'  Project root: {config.paths.root}')
except Exception as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Installation verified successfully"
else
    echo -e "${RED}✗${NC} Installation verification failed"
    exit 1
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                    SETUP COMPLETE                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 Next Steps:"
echo ""
echo "   1. If you skipped data download, run:"
echo "      python utility/download_image_embeddings.py"
echo ""
echo "   2. Run the pipeline:"
echo "      python run_pipeline.py"
echo ""
echo "   3. View results in:"
echo "      - analysis/clusters.csv"
echo "      - analysis/survival_plots/"
echo "      - analysis/attention_heatmaps/"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

