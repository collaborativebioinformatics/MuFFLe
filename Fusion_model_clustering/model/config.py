"""
Configuration file for CHIMERA Task 3 Unsupervised Multimodal Fusion Pipeline.

This configuration manages all paths, hyperparameters, and constants for the 
inference-only unsupervised pipeline that stratifies bladder cancer patients
into recurrence risk groups.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

# ============================================================================
# BASE PATHS
# ============================================================================

# Automatically determine project root
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

@dataclass
class PathConfig:
    """All path configurations for the pipeline."""
    
    # Project root
    root: Path = PROJECT_ROOT
    
    # Data directories
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    clinical_data: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "clinical_data.csv")
    rna_embeddings_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "rna_embeddings")
    wsi_features_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "wsi_data" / "features")
    wsi_coordinates_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "wsi_data" / "coordinates")
    metadata: Path = field(default_factory=lambda: PROJECT_ROOT / "metadata.csv")
    
    # Processing output directories
    processing_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "processing")
    patient_signatures_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "processing" / "patient_signatures")
    attention_results_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "processing" / "attention_results")
    
    # Analysis output directories
    analysis_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "analysis")
    clusters_output: Path = field(default_factory=lambda: PROJECT_ROOT / "analysis" / "clusters.csv")
    survival_plots_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "analysis" / "survival_plots")
    
    # S3 Configuration (for cloud compute)
    s3_bucket: str = "chimera-challenge"
    s3_prefix: str = "v2/task3/"
    
    def ensure_dirs(self):
        """Create all output directories if they don't exist."""
        for attr_name in ['processing_dir', 'patient_signatures_dir', 
                          'attention_results_dir', 'analysis_dir', 'survival_plots_dir']:
            path = getattr(self, attr_name)
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class FeatureConfig:
    """Feature dimensions and configurations."""
    
    # WSI (Whole Slide Image) Features
    wsi_patch_dim: int = 1024  # UNI embedding dimension per patch
    wsi_output_dim: int = 1024  # Aggregated slide representation
    
    # RNA Features
    rna_embedding_dim: int = 256  # Pre-computed MLP embedding dimension
    
    # Fused representation
    fused_dim: int = 1280  # WSI (1024) + RNA (256)
    
    # Attention pooling
    top_k_attention_percent: float = 0.01  # Top 1% patches for interpretability


@dataclass
class ClusteringConfig:
    """HDBSCAN clustering hyperparameters."""
    
    # HDBSCAN parameters
    min_cluster_size: int = 5  # Minimum members to form a cluster
    min_samples: int = 3  # Core point threshold
    cluster_selection_epsilon: float = 0.0  # No distance threshold
    cluster_selection_method: str = "eom"  # Excess of Mass (default)
    metric: str = "euclidean"
    
    # Fallback to K-Means if HDBSCAN produces too few clusters
    fallback_n_clusters: int = 3  # For survival analysis (Low/Medium/High risk)
    use_kmeans_fallback: bool = True


@dataclass
class ProcessingConfig:
    """Processing and computation settings."""
    
    # Device configuration
    device: str = "cuda"  # Will fallback to CPU if unavailable
    
    # Batch processing
    batch_size: int = 1  # Process one patient at a time (memory efficient for large WSIs)
    num_workers: int = 4
    
    # Normalization
    use_z_score_normalization: bool = True  # StandardScaler
    use_robust_scaler: bool = False  # Alternative: RobustScaler
    
    # Random seed for reproducibility
    seed: int = 42
    
    # Logging
    verbose: bool = True
    save_intermediate: bool = True  # Save attention weights and signatures


@dataclass 
class SurvivalConfig:
    """Survival analysis configuration."""
    
    # Clinical columns
    time_column: str = "Time_to_prog_or_FUend"
    event_column: str = "progression"
    patient_id_column: str = "chimera_id_t3"
    
    # Statistical tests
    significance_level: float = 0.05
    
    # Visualization
    figure_dpi: int = 150
    figure_size: tuple = (10, 8)


@dataclass
class PipelineConfig:
    """Master configuration combining all sub-configs."""
    
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    survival: SurvivalConfig = field(default_factory=SurvivalConfig)
    
    def __post_init__(self):
        """Ensure output directories exist."""
        self.paths.ensure_dirs()


# ============================================================================
# DEFAULT CONFIGURATION INSTANCE
# ============================================================================

def get_config() -> PipelineConfig:
    """Factory function to get default pipeline configuration."""
    return PipelineConfig()


# For quick access
DEFAULT_CONFIG = PipelineConfig()

