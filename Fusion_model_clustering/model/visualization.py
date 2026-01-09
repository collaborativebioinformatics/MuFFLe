"""
Visualization Module for CHIMERA Task 3 Pipeline.

Provides:
1. Attention heatmap generation (spatial mapping of patch importance)
2. Kaplan-Meier survival curves
3. Cluster distribution visualizations
4. Concordance index calculation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings

# Optional survival analysis imports
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test, multivariate_logrank_test
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    warnings.warn("lifelines not installed. Survival analysis disabled.")

from .config import PipelineConfig, get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CUSTOM COLORMAPS
# ============================================================================

def get_attention_cmap():
    """Create custom colormap for attention heatmaps."""
    colors = [
        (0.1, 0.1, 0.3),      # Dark blue (low attention)
        (0.2, 0.4, 0.6),      # Medium blue
        (0.3, 0.7, 0.5),      # Teal
        (0.9, 0.9, 0.3),      # Yellow
        (0.95, 0.5, 0.1),     # Orange
        (0.9, 0.1, 0.1),      # Red (high attention)
    ]
    return LinearSegmentedColormap.from_list('attention', colors, N=256)


def get_cluster_colors(n_clusters: int) -> List[str]:
    """Generate distinct colors for clusters."""
    base_colors = [
        '#E63946',  # Red
        '#457B9D',  # Blue
        '#2A9D8F',  # Teal
        '#E9C46A',  # Yellow
        '#F4A261',  # Orange
        '#9B5DE5',  # Purple
        '#00BBF9',  # Cyan
        '#00F5D4',  # Mint
    ]
    
    if n_clusters <= len(base_colors):
        return base_colors[:n_clusters]
    else:
        # Generate additional colors if needed
        import colorsys
        colors = base_colors.copy()
        for i in range(n_clusters - len(base_colors)):
            hue = (i * 0.618033988749895) % 1  # Golden ratio
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
            colors.append('#{:02x}{:02x}{:02x}'.format(
                int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)
            ))
        return colors


# ============================================================================
# ATTENTION HEATMAP GENERATION
# ============================================================================

class AttentionHeatmapGenerator:
    """
    Generate spatial attention heatmaps by mapping attention weights to coordinates.
    
    This validates the biological grounding of the clustering by showing
    whether high-attention patches focus on tumor regions.
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = config.paths.analysis_dir / "attention_heatmaps"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cmap = get_attention_cmap()
        
    def load_attention_data(self, patient_id: str) -> Optional[Dict]:
        """Load saved attention weights and indices."""
        attn_path = self.config.paths.attention_results_dir / f"{patient_id}_attention.npz"
        
        if not attn_path.exists():
            logger.warning(f"No attention data for {patient_id}")
            return None
        
        data = np.load(attn_path)
        return {
            'weights': data['weights'],
            'top_k_indices': data['top_k_indices'],
            'n_patches': int(data['n_patches'])
        }
    
    def load_coordinates(self, patient_id: str) -> Optional[np.ndarray]:
        """Load patch coordinates."""
        coord_path = self.config.paths.wsi_coordinates_dir / f"{patient_id}_HE.npy"
        
        if not coord_path.exists():
            logger.warning(f"No coordinates for {patient_id}")
            return None
        
        return np.load(coord_path, allow_pickle=True)
    
    def generate_heatmap(self, patient_id: str,
                         highlight_top_k: bool = True,
                         figsize: Tuple[int, int] = (14, 10),
                         save: bool = True) -> Optional[plt.Figure]:
        """
        Generate attention heatmap for a patient.
        
        Args:
            patient_id: Patient identifier
            highlight_top_k: Highlight top-k attention patches
            figsize: Figure size
            save: Save to file
            
        Returns:
            Matplotlib figure or None
        """
        # Load data
        attn_data = self.load_attention_data(patient_id)
        coords = self.load_coordinates(patient_id)
        
        if attn_data is None or coords is None:
            return None
        
        weights = attn_data['weights']
        top_k_indices = attn_data['top_k_indices']
        
        # Extract x, y coordinates
        x = coords['x']
        y = coords['y']
        
        # Ensure lengths match
        if len(weights) != len(x):
            logger.error(f"Length mismatch for {patient_id}: weights={len(weights)}, coords={len(x)}")
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Full attention heatmap
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            x, y, 
            c=weights, 
            cmap=self.cmap,
            s=1,  # Small point size for dense plots
            alpha=0.7
        )
        ax1.set_title(f'{patient_id} - Attention Heatmap\n({len(weights):,} patches)', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.invert_yaxis()  # Match image orientation
        ax1.set_aspect('equal')
        plt.colorbar(scatter1, ax=ax1, label='Attention Weight')
        
        # Right plot: Top-k highlighted
        ax2 = axes[1]
        
        # Plot all patches in gray
        ax2.scatter(x, y, c='lightgray', s=1, alpha=0.3)
        
        # Highlight top-k in red
        if highlight_top_k and len(top_k_indices) > 0:
            ax2.scatter(
                x[top_k_indices], 
                y[top_k_indices],
                c='red',
                s=3,
                alpha=0.8,
                label=f'Top {len(top_k_indices)} patches'
            )
        
        ax2.set_title(f'{patient_id} - Top 1% High-Attention Regions\n(Potential Tumor Focus)', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        ax2.invert_yaxis()
        ax2.set_aspect('equal')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Save
        if save:
            save_path = self.output_dir / f"{patient_id}_attention_heatmap.png"
            fig.savefig(save_path, dpi=self.config.survival.figure_dpi, 
                       bbox_inches='tight', facecolor='white')
            logger.info(f"Saved heatmap: {save_path}")
        
        return fig
    
    def generate_cluster_heatmaps(self, cluster_df: pd.DataFrame,
                                   samples_per_cluster: int = 2):
        """
        Generate heatmaps for representative patients from each cluster.
        
        Args:
            cluster_df: DataFrame with patient_id and cluster columns
            samples_per_cluster: Number of samples per cluster
        """
        unique_clusters = cluster_df['cluster'].unique()
        
        for cluster in sorted(unique_clusters):
            if cluster == -1:  # Skip noise
                continue
                
            cluster_patients = cluster_df[
                cluster_df['cluster'] == cluster
            ]['patient_id'].tolist()
            
            # Sample patients
            samples = cluster_patients[:samples_per_cluster]
            
            logger.info(f"\nGenerating heatmaps for Cluster {cluster}:")
            for pid in samples:
                self.generate_heatmap(pid)


# ============================================================================
# SURVIVAL ANALYSIS
# ============================================================================

class SurvivalAnalyzer:
    """
    Perform survival analysis to validate cluster clinical significance.
    
    Methods:
    - Kaplan-Meier curves per cluster
    - Log-rank test for statistical significance
    - Concordance index calculation
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.survival_config = config.survival
        self.output_dir = config.paths.survival_plots_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if not LIFELINES_AVAILABLE:
            raise ImportError("lifelines package required for survival analysis")
    
    def prepare_data(self, cluster_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge cluster assignments with clinical survival data.
        
        Args:
            cluster_df: DataFrame with patient_id and cluster
            
        Returns:
            Merged DataFrame with survival columns
        """
        # Load clinical data
        clinical_df = pd.read_csv(self.config.paths.clinical_data)
        
        # Merge
        merged = cluster_df.merge(
            clinical_df,
            left_on='patient_id',
            right_on=self.survival_config.patient_id_column,
            how='inner'
        )
        
        # Clean survival columns
        time_col = self.survival_config.time_column
        event_col = self.survival_config.event_column
        
        # Handle missing/invalid values
        merged = merged[merged[time_col] > 0]
        merged[event_col] = merged[event_col].astype(int)
        
        logger.info(f"Prepared survival data for {len(merged)} patients")
        
        return merged
    
    def plot_kaplan_meier(self, data: pd.DataFrame,
                         figsize: Tuple[int, int] = (12, 8),
                         save: bool = True) -> plt.Figure:
        """
        Plot Kaplan-Meier survival curves for each cluster.
        
        Args:
            data: DataFrame with cluster, time, and event columns
            figsize: Figure size
            save: Save to file
            
        Returns:
            Matplotlib figure
        """
        time_col = self.survival_config.time_column
        event_col = self.survival_config.event_column
        
        # Filter out noise cluster
        data_clean = data[data['cluster'] != -1].copy()
        unique_clusters = sorted(data_clean['cluster'].unique())
        n_clusters = len(unique_clusters)
        
        colors = get_cluster_colors(n_clusters)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        kmf = KaplanMeierFitter()
        
        for i, cluster in enumerate(unique_clusters):
            cluster_data = data_clean[data_clean['cluster'] == cluster]
            
            kmf.fit(
                cluster_data[time_col],
                event_observed=cluster_data[event_col],
                label=f'Cluster {cluster} (n={len(cluster_data)})'
            )
            
            kmf.plot_survival_function(
                ax=ax,
                color=colors[i],
                ci_show=True,
                ci_alpha=0.15,
                linewidth=2.5
            )
        
        # Styling
        ax.set_xlabel('Time (months)', fontsize=12)
        ax.set_ylabel('Progression-Free Survival Probability', fontsize=12)
        ax.set_title('Kaplan-Meier Survival Curves by Cluster\n'
                    '(Multimodal WSI+RNA Stratification)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        # Add statistical test result
        if n_clusters >= 2:
            result = multivariate_logrank_test(
                data_clean[time_col],
                data_clean['cluster'],
                data_clean[event_col]
            )
            
            p_value = result.p_value
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            
            ax.text(
                0.95, 0.95,
                f'Log-rank p = {p_value:.4f} {significance}',
                transform=ax.transAxes,
                fontsize=11,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "kaplan_meier_curves.png"
            fig.savefig(save_path, dpi=self.config.survival.figure_dpi,
                       bbox_inches='tight', facecolor='white')
            logger.info(f"Saved KM curves: {save_path}")
        
        return fig
    
    def calculate_concordance(self, data: pd.DataFrame) -> float:
        """
        Calculate concordance index (C-index) for cluster risk ordering.
        
        A C-index > 0.65 suggests clinically meaningful stratification.
        
        Args:
            data: DataFrame with cluster, time, and event columns
            
        Returns:
            Concordance index value
        """
        time_col = self.survival_config.time_column
        event_col = self.survival_config.event_column
        
        # Filter out noise
        data_clean = data[data['cluster'] != -1].copy()
        
        # Calculate median survival per cluster to order risk
        cluster_medians = {}
        for cluster in data_clean['cluster'].unique():
            cluster_data = data_clean[data_clean['cluster'] == cluster]
            kmf = KaplanMeierFitter()
            kmf.fit(cluster_data[time_col], event_observed=cluster_data[event_col])
            cluster_medians[cluster] = kmf.median_survival_time_
        
        # Create risk scores (lower median = higher risk = higher score)
        sorted_clusters = sorted(cluster_medians.keys(), 
                                key=lambda x: cluster_medians[x])
        risk_mapping = {c: i for i, c in enumerate(sorted_clusters)}
        
        data_clean['risk_score'] = data_clean['cluster'].map(risk_mapping)
        
        c_index = concordance_index(
            data_clean[time_col],
            -data_clean['risk_score'],  # Negative because higher risk should correlate with shorter time
            data_clean[event_col]
        )
        
        logger.info(f"Concordance Index: {c_index:.4f}")
        
        return c_index
    
    def run_full_analysis(self, cluster_df: pd.DataFrame) -> Dict:
        """
        Run complete survival analysis.
        
        Args:
            cluster_df: DataFrame with patient_id and cluster
            
        Returns:
            Dictionary with analysis results
        """
        # Prepare data
        data = self.prepare_data(cluster_df)
        
        results = {
            'n_patients': len(data),
            'n_events': data[self.survival_config.event_column].sum(),
        }
        
        # Plot KM curves
        self.plot_kaplan_meier(data)
        
        # Calculate C-index
        try:
            c_index = self.calculate_concordance(data)
            results['c_index'] = c_index
            results['clinically_significant'] = c_index > 0.65
        except Exception as e:
            logger.warning(f"Could not calculate C-index: {e}")
            results['c_index'] = None
        
        # Perform log-rank test
        data_clean = data[data['cluster'] != -1]
        if len(data_clean['cluster'].unique()) >= 2:
            result = multivariate_logrank_test(
                data_clean[self.survival_config.time_column],
                data_clean['cluster'],
                data_clean[self.survival_config.event_column]
            )
            results['logrank_p_value'] = result.p_value
            results['significant'] = result.p_value < self.survival_config.significance_level
        
        return results


# ============================================================================
# CLUSTER VISUALIZATION
# ============================================================================

class ClusterVisualizer:
    """Visualize cluster distributions and characteristics."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.output_dir = config.paths.analysis_dir
        
    def plot_cluster_distribution(self, cluster_df: pd.DataFrame,
                                   save: bool = True) -> plt.Figure:
        """
        Plot cluster size distribution.
        
        Args:
            cluster_df: DataFrame with cluster column
            save: Save to file
            
        Returns:
            Matplotlib figure
        """
        cluster_counts = cluster_df['cluster'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = get_cluster_colors(len(cluster_counts))
        labels = [f'Cluster {c}' if c != -1 else 'Noise' 
                 for c in cluster_counts.index]
        
        bars = ax.bar(range(len(cluster_counts)), cluster_counts.values, 
                     color=colors)
        
        ax.set_xticks(range(len(cluster_counts)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel('Number of Patients', fontsize=12)
        ax.set_title('Patient Distribution Across Clusters', 
                    fontsize=14, fontweight='bold')
        
        # Add count labels on bars
        for bar, count in zip(bars, cluster_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   str(count), ha='center', fontsize=11, fontweight='bold')
        
        ax.set_ylim(0, max(cluster_counts.values) * 1.15)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "cluster_distribution.png"
            fig.savefig(save_path, dpi=self.config.survival.figure_dpi,
                       bbox_inches='tight', facecolor='white')
            logger.info(f"Saved cluster distribution: {save_path}")
        
        return fig
    
    def plot_signature_tsne(self, signatures: np.ndarray,
                            labels: np.ndarray,
                            patient_ids: List[str],
                            save: bool = True) -> plt.Figure:
        """
        Plot t-SNE visualization of patient signatures.
        
        Args:
            signatures: Patient signature matrix [n, 1280]
            labels: Cluster labels
            patient_ids: Patient identifiers
            save: Save to file
            
        Returns:
            Matplotlib figure
        """
        from sklearn.manifold import TSNE
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=self.config.processing.seed,
                   perplexity=min(30, len(signatures) - 1))
        embeddings = tsne.fit_transform(signatures)
        
        unique_labels = sorted(set(labels))
        colors = get_cluster_colors(len(unique_labels))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_name = f'Cluster {label}' if label != -1 else 'Noise'
            
            ax.scatter(
                embeddings[mask, 0],
                embeddings[mask, 1],
                c=colors[i],
                label=f'{cluster_name} (n={mask.sum()})',
                s=80,
                alpha=0.7,
                edgecolors='white',
                linewidth=0.5
            )
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title('t-SNE Visualization of Multimodal Patient Signatures\n'
                    '(WSI 1024-d + RNA 256-d → 1280-d fused)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / "signature_tsne.png"
            fig.savefig(save_path, dpi=self.config.survival.figure_dpi,
                       bbox_inches='tight', facecolor='white')
            logger.info(f"Saved t-SNE visualization: {save_path}")
        
        return fig


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_all_visualizations(cluster_df: pd.DataFrame,
                                signatures: np.ndarray,
                                patient_ids: List[str],
                                config: Optional[PipelineConfig] = None):
    """
    Generate all visualization outputs.
    
    Args:
        cluster_df: DataFrame with patient_id and cluster
        signatures: Patient signature matrix
        patient_ids: Patient identifiers
        config: Optional configuration
    """
    config = config or get_config()
    
    logger.info("\n" + "=" * 60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 60)
    
    # Cluster distribution
    cluster_viz = ClusterVisualizer(config)
    cluster_viz.plot_cluster_distribution(cluster_df)
    cluster_viz.plot_signature_tsne(signatures, cluster_df['cluster'].values, patient_ids)
    
    # Survival analysis (if lifelines available)
    if LIFELINES_AVAILABLE:
        try:
            survival = SurvivalAnalyzer(config)
            results = survival.run_full_analysis(cluster_df)
            
            logger.info("\nSurvival Analysis Results:")
            logger.info(f"  - Patients analyzed: {results['n_patients']}")
            logger.info(f"  - Events (progressions): {results['n_events']}")
            if results.get('c_index'):
                logger.info(f"  - C-Index: {results['c_index']:.4f}")
                logger.info(f"  - Clinically significant: {results.get('clinically_significant', 'N/A')}")
            if results.get('logrank_p_value'):
                logger.info(f"  - Log-rank p-value: {results['logrank_p_value']:.4f}")
                logger.info(f"  - Statistically significant: {results.get('significant', 'N/A')}")
        except Exception as e:
            logger.warning(f"Survival analysis failed: {e}")
    
    # Attention heatmaps for samples
    heatmap_gen = AttentionHeatmapGenerator(config)
    heatmap_gen.generate_cluster_heatmaps(cluster_df, samples_per_cluster=2)
    
    logger.info("\nVisualization generation complete!")

