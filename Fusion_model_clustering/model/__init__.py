"""
CHIMERA Task 3: Unsupervised Multimodal Fusion Pipeline

This package provides inference-only unsupervised clustering of bladder cancer
patients using fused WSI histopathology and RNA transcriptomics features.

Main Components:
- UnsupervisedFusionPipeline: Complete pipeline for patient stratification
- WSIAggregator: Variance-weighted attention pooling for slide features
- MultimodalFusion: Z-score normalized feature concatenation
- ClusteringEngine: HDBSCAN-based density clustering

Usage:
    from model import run_pipeline
    results = run_pipeline()

    # Or with custom configuration:
    from model import PipelineConfig, UnsupervisedFusionPipeline
    config = PipelineConfig()
    pipeline = UnsupervisedFusionPipeline(config)
    results = pipeline.run()
"""

from .config import (
    PipelineConfig,
    PathConfig,
    FeatureConfig,
    ClusteringConfig,
    ProcessingConfig,
    SurvivalConfig,
    get_config,
    DEFAULT_CONFIG,
)

from .data_loader import (
    LocalDataLoader,
    S3DataLoader,
    HybridDataLoader,
    PatientData,
    get_data_loader,
)

from .unsupervised_fusion import (
    UnsupervisedFusionPipeline,
    WSIAggregator,
    GatedAttentionAggregator,
    MultimodalFusion,
    ClusteringEngine,
    AggregationResult,
    FusionResult,
    run_pipeline,
)

from .visualization import (
    AttentionHeatmapGenerator,
    SurvivalAnalyzer,
    ClusterVisualizer,
    generate_all_visualizations,
)


__version__ = "1.0.0"
__author__ = "CHIMERA Challenge Team"

__all__ = [
    # Configuration
    "PipelineConfig",
    "PathConfig",
    "FeatureConfig",
    "ClusteringConfig",
    "ProcessingConfig",
    "SurvivalConfig",
    "get_config",
    "DEFAULT_CONFIG",
    # Data Loading
    "LocalDataLoader",
    "S3DataLoader",
    "HybridDataLoader",
    "PatientData",
    "get_data_loader",
    # Pipeline
    "UnsupervisedFusionPipeline",
    "WSIAggregator",
    "GatedAttentionAggregator",
    "MultimodalFusion",
    "ClusteringEngine",
    "AggregationResult",
    "FusionResult",
    "run_pipeline",
    # Visualization
    "AttentionHeatmapGenerator",
    "SurvivalAnalyzer",
    "ClusterVisualizer",
    "generate_all_visualizations",
]

