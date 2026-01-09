#!/usr/bin/env python3
"""
CHIMERA Task 3: Unsupervised Multimodal Fusion Pipeline Runner

This script executes the complete inference-only unsupervised pipeline:
1. Aggregates WSI patch features (N x 1024) → slide embedding (1024-d)
2. Fuses with RNA embeddings (256-d) → patient signature (1280-d)
3. Clusters patients using HDBSCAN
4. Validates with survival analysis
5. Generates attention heatmaps for interpretability

Usage:
    # Basic execution
    python run_pipeline.py
    
    # With specific options
    python run_pipeline.py --use-variance-attention --skip-visualization
    
    # Cloud compute mode (S3 fallback enabled)
    python run_pipeline.py --s3-fallback

Output Files:
    - analysis/clusters.csv: Patient → Cluster mapping
    - processing/patient_signatures/signatures.npy: 1280-d signature matrix
    - analysis/survival_plots/: Kaplan-Meier curves
    - analysis/attention_heatmaps/: Spatial attention visualizations
"""

import argparse
import sys
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model.config import get_config, PipelineConfig
from model.unsupervised_fusion import UnsupervisedFusionPipeline
from model.visualization import generate_all_visualizations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print pipeline banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CHIMERA TASK 3: MULTIMODAL FUSION                         ║
║                    Unsupervised Risk Stratification                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WSI Features (N × 1024) ──┬──> Variance Attention Pooling ──┐               ║
║                            │                                  │               ║
║  RNA Embedding (1 × 256) ──┴──> Z-Score Normalization ──────>│──> HDBSCAN    ║
║                                                               │    Clustering ║
║                                     Fusion (1280-d) <────────┘               ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='CHIMERA Task 3 Unsupervised Multimodal Fusion Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Attention mechanism
    parser.add_argument(
        '--use-variance-attention',
        action='store_true',
        help='Use simple variance attention instead of gated attention'
    )
    
    # Clustering options
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=5,
        help='Minimum cluster size for HDBSCAN (default: 5)'
    )
    
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=3,
        help='Number of clusters for KMeans fallback (default: 3)'
    )
    
    # Data loading
    parser.add_argument(
        '--s3-fallback',
        action='store_true',
        help='Enable S3 fallback for missing local WSI features'
    )
    
    parser.add_argument(
        '--patient-ids',
        nargs='+',
        help='Specific patient IDs to process (default: all available)'
    )
    
    # Output options
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--skip-survival-analysis',
        action='store_true',
        help='Skip survival analysis'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )
    
    # Verbosity
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    return parser.parse_args()


def main():
    """Main pipeline execution."""
    args = parse_args()
    
    # Set logging level
    if args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print_banner()
    
    # Configure pipeline
    config = get_config()
    
    # Override config with command line args
    if args.min_cluster_size:
        config.clustering.min_cluster_size = args.min_cluster_size
    if args.n_clusters:
        config.clustering.fallback_n_clusters = args.n_clusters
    if args.output_dir:
        config.paths.analysis_dir = Path(args.output_dir)
        config.paths.clusters_output = Path(args.output_dir) / "clusters.csv"
    
    # Initialize pipeline
    logger.info("Initializing pipeline...")
    pipeline = UnsupervisedFusionPipeline(
        config=config,
        use_gated_attention=not args.use_variance_attention
    )
    
    # Check data availability
    logger.info("\n📊 Data Availability Check:")
    local_loader = pipeline.data_loader.local_loader
    
    all_patients = local_loader.patient_ids
    wsi_available = local_loader.get_available_wsi_ids()
    rna_available = local_loader.get_available_rna_ids()
    complete = local_loader.get_complete_ids()
    
    logger.info(f"   Total patients in clinical data: {len(all_patients)}")
    logger.info(f"   WSI features available locally: {len(wsi_available)}")
    logger.info(f"   RNA embeddings available: {len(rna_available)}")
    logger.info(f"   Complete (both modalities): {len(complete)}")
    
    if len(complete) == 0:
        logger.error("\n❌ No patients with complete data found!")
        logger.error("   Please ensure WSI features and RNA embeddings are downloaded.")
        sys.exit(1)
    
    # Run pipeline
    try:
        logger.info("\n🚀 Starting pipeline execution...")
        results_df = pipeline.run(
            patient_ids=args.patient_ids,
            save_results=True
        )
        
        # Generate visualizations
        if not args.skip_visualization:
            logger.info("\n📈 Generating visualizations...")
            try:
                generate_all_visualizations(
                    cluster_df=results_df,
                    signatures=pipeline.patient_signatures,
                    patient_ids=pipeline.processed_ids,
                    config=config
                )
            except Exception as e:
                logger.warning(f"Visualization generation failed: {e}")
                logger.warning("Continuing without visualizations...")
        
        # Print final summary
        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"\n📁 Output Files:")
        print(f"   • Cluster assignments: {config.paths.clusters_output}")
        print(f"   • Patient signatures:  {config.paths.patient_signatures_dir}/signatures.npy")
        print(f"   • Attention results:   {config.paths.attention_results_dir}/")
        print(f"   • Survival plots:      {config.paths.survival_plots_dir}/")
        print(f"\n📊 Results Summary:")
        
        cluster_counts = results_df['cluster'].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            label = f"Cluster {cluster}" if cluster != -1 else "Noise"
            pct = 100 * count / len(results_df)
            print(f"   • {label}: {count} patients ({pct:.1f}%)")
        
        print(f"\n   Total patients processed: {len(results_df)}")
        print("=" * 70)
        
        return results_df
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    results = main()

