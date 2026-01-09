"""
Unsupervised Multimodal Fusion Pipeline for CHIMERA Task 3.

This module implements:
1. Weighted Variance Pooling (Heuristic Attention) for WSI aggregation
2. Multimodal fusion of WSI (1024-d) and RNA (256-d) features
3. HDBSCAN clustering for patient stratification
4. Spatial attention mapping for interpretability

Key Principle: INFERENCE-ONLY, NO TRAINING
- No model weights are updated
- No loss functions or optimizers
- Uses self-normalizing heuristics for attention
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
import logging
from dataclasses import dataclass
import warnings

# Optional HDBSCAN import
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    warnings.warn("hdbscan not installed. Will use KMeans as fallback.")

from .config import PipelineConfig, get_config
from .data_loader import get_data_loader, PatientData, HybridDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class AggregationResult:
    """Result of WSI feature aggregation."""
    patient_id: str
    slide_embedding: np.ndarray  # [1024]
    attention_weights: np.ndarray  # [N]
    top_k_indices: np.ndarray  # Indices of highest attention patches
    n_patches: int


@dataclass
class FusionResult:
    """Result of multimodal fusion for a single patient."""
    patient_id: str
    fused_signature: np.ndarray  # [1280]
    wsi_embedding: np.ndarray  # [1024] (Z-scored)
    rna_embedding: np.ndarray  # [256] (Z-scored)
    attention_weights: Optional[np.ndarray] = None
    top_k_indices: Optional[np.ndarray] = None


class WSIAggregator:
    """
    Aggregate variable-sized WSI patch features into fixed slide representation.
    
    Uses Weighted Variance Pooling (Heuristic Attention):
    - High-variance patches represent complex morphologies (tumor nests, pleomorphism)
    - Low-variance patches are typically background/stroma
    
    Formula: Weight_i = Softmax(Var(Patch_i))
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.top_k_percent = config.features.top_k_attention_percent
        
    def compute_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute heuristic attention weights based on patch variance.
        
        High-variance patches typically represent:
        - Tumor regions with complex cellular structures
        - Areas of high pleomorphism
        - Regions with significant morphological features
        
        Args:
            features: Tensor of shape [N, 1024]
            
        Returns:
            Attention weights of shape [N], summing to 1
        """
        # Compute variance across feature dimension for each patch
        # Shape: [N]
        patch_variance = torch.var(features, dim=1)
        
        # Apply softmax for normalization
        # Using temperature scaling to control sharpness
        temperature = 1.0
        attention_weights = F.softmax(patch_variance / temperature, dim=0)
        
        return attention_weights
    
    def aggregate(self, features: torch.Tensor, 
                  patient_id: str) -> AggregationResult:
        """
        Aggregate patch features into slide-level representation.
        
        Args:
            features: Tensor of shape [N, 1024]
            patient_id: Patient identifier
            
        Returns:
            AggregationResult with slide embedding and attention info
        """
        n_patches = features.shape[0]
        
        # Compute attention weights
        attention_weights = self.compute_attention_weights(features)
        
        # Weighted sum pooling -> [1024]
        # Multiply each patch by its attention weight and sum
        slide_embedding = torch.sum(
            features * attention_weights.unsqueeze(1), 
            dim=0
        )
        
        # Get top-k attention indices for interpretability
        k = max(1, int(n_patches * self.top_k_percent))
        top_k_indices = torch.topk(attention_weights, k=k).indices
        
        return AggregationResult(
            patient_id=patient_id,
            slide_embedding=slide_embedding.numpy(),
            attention_weights=attention_weights.numpy(),
            top_k_indices=top_k_indices.numpy(),
            n_patches=n_patches
        )


class GatedAttentionAggregator(WSIAggregator):
    """
    Alternative aggregator using Gated Attention mechanism.
    
    Uses Tanh-Sigmoid gating without learned parameters:
    - Tanh branch: captures feature relevance
    - Sigmoid branch: gating mechanism
    
    This provides more nuanced attention than pure variance.
    """
    
    def compute_attention_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute gated attention weights using self-normalizing heuristic.
        
        Args:
            features: Tensor of shape [N, 1024]
            
        Returns:
            Attention weights of shape [N]
        """
        # Compute per-patch statistics
        patch_mean = features.mean(dim=1)  # [N]
        patch_std = features.std(dim=1)    # [N]
        patch_max = features.max(dim=1).values  # [N]
        
        # Tanh branch: based on normalized mean (relevance)
        tanh_branch = torch.tanh(patch_mean / (patch_mean.std() + 1e-8))
        
        # Sigmoid branch: based on variance (confidence/informativeness)
        patch_variance = patch_std ** 2
        sigmoid_branch = torch.sigmoid(
            (patch_variance - patch_variance.mean()) / (patch_variance.std() + 1e-8)
        )
        
        # Gated attention: element-wise product
        gated_scores = tanh_branch * sigmoid_branch
        
        # Add max activation as additional signal for highly activated regions
        max_normalized = (patch_max - patch_max.mean()) / (patch_max.std() + 1e-8)
        combined_scores = gated_scores + 0.5 * torch.sigmoid(max_normalized)
        
        # Softmax normalization
        attention_weights = F.softmax(combined_scores, dim=0)
        
        return attention_weights


class MultimodalFusion:
    """
    Fuse WSI and RNA modalities into unified patient signature.
    
    Steps:
    1. Z-Score normalize WSI embedding (1024-d)
    2. Z-Score normalize RNA embedding (256-d)
    3. Concatenate to form 1280-d patient signature
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.wsi_scaler = StandardScaler()
        self.rna_scaler = StandardScaler()
        self._fitted = False
        
    def fit_scalers(self, wsi_embeddings: np.ndarray, 
                    rna_embeddings: np.ndarray):
        """
        Fit scalers on cohort data for Z-score normalization.
        
        Args:
            wsi_embeddings: Array of shape [n_patients, 1024]
            rna_embeddings: Array of shape [n_patients, 256]
        """
        self.wsi_scaler.fit(wsi_embeddings)
        self.rna_scaler.fit(rna_embeddings)
        self._fitted = True
        
    def transform(self, wsi_embedding: np.ndarray,
                  rna_embedding: np.ndarray,
                  patient_id: str,
                  attention_weights: Optional[np.ndarray] = None,
                  top_k_indices: Optional[np.ndarray] = None) -> FusionResult:
        """
        Transform and fuse a single patient's embeddings.
        
        Args:
            wsi_embedding: Shape [1024]
            rna_embedding: Shape [256]
            patient_id: Patient identifier
            attention_weights: Optional attention weights for saving
            top_k_indices: Optional top-k indices for saving
            
        Returns:
            FusionResult with fused signature
        """
        if not self._fitted:
            raise RuntimeError("Scalers must be fitted before transform")
        
        # Reshape for sklearn
        wsi_2d = wsi_embedding.reshape(1, -1)
        rna_2d = rna_embedding.reshape(1, -1)
        
        # Z-score normalization
        wsi_normalized = self.wsi_scaler.transform(wsi_2d).flatten()
        rna_normalized = self.rna_scaler.transform(rna_2d).flatten()
        
        # Concatenate: [1024 + 256] = [1280]
        fused = np.concatenate([wsi_normalized, rna_normalized])
        
        return FusionResult(
            patient_id=patient_id,
            fused_signature=fused,
            wsi_embedding=wsi_normalized,
            rna_embedding=rna_normalized,
            attention_weights=attention_weights,
            top_k_indices=top_k_indices
        )


class ClusteringEngine:
    """
    Density-based clustering using HDBSCAN with KMeans fallback.
    
    HDBSCAN advantages:
    - Automatically determines number of clusters
    - Identifies outlier patients (noise, label=-1)
    - Handles clusters of varying densities
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cluster_config = config.clustering
        self._clusterer = None
        self._labels = None
        self._probabilities = None
        
    def fit_predict(self, signatures: np.ndarray) -> np.ndarray:
        """
        Cluster patient signatures.
        
        Args:
            signatures: Array of shape [n_patients, 1280]
            
        Returns:
            Cluster labels array of shape [n_patients]
        """
        n_patients = signatures.shape[0]
        
        if HDBSCAN_AVAILABLE:
            logger.info("Using HDBSCAN for clustering")
            self._clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min(self.cluster_config.min_cluster_size, 
                                    n_patients // 3),
                min_samples=min(self.cluster_config.min_samples, 
                               n_patients // 5),
                cluster_selection_epsilon=self.cluster_config.cluster_selection_epsilon,
                cluster_selection_method=self.cluster_config.cluster_selection_method,
                metric=self.cluster_config.metric,
                gen_min_span_tree=True
            )
            
            self._labels = self._clusterer.fit_predict(signatures)
            self._probabilities = self._clusterer.probabilities_
            
            n_clusters = len(set(self._labels)) - (1 if -1 in self._labels else 0)
            n_noise = (self._labels == -1).sum()
            
            logger.info(f"HDBSCAN found {n_clusters} clusters, {n_noise} noise points")
            
            # Fallback to KMeans if HDBSCAN produces poor results
            if (n_clusters < 2 or n_noise > n_patients * 0.3) and \
               self.cluster_config.use_kmeans_fallback:
                logger.warning("HDBSCAN produced suboptimal clusters, using KMeans fallback")
                return self._kmeans_fallback(signatures)
        else:
            return self._kmeans_fallback(signatures)
        
        return self._labels
    
    def _kmeans_fallback(self, signatures: np.ndarray) -> np.ndarray:
        """KMeans fallback clustering."""
        logger.info(f"Using KMeans with k={self.cluster_config.fallback_n_clusters}")
        
        kmeans = KMeans(
            n_clusters=self.cluster_config.fallback_n_clusters,
            random_state=self.config.processing.seed,
            n_init=10
        )
        
        self._labels = kmeans.fit_predict(signatures)
        self._probabilities = None
        self._clusterer = kmeans
        
        return self._labels
    
    @property
    def labels(self) -> np.ndarray:
        return self._labels
    
    @property
    def probabilities(self) -> Optional[np.ndarray]:
        return self._probabilities


class UnsupervisedFusionPipeline:
    """
    Complete unsupervised pipeline for multimodal patient stratification.
    
    Pipeline Flow:
    1. Load WSI features (N x 1024) and RNA embeddings (256) for each patient
    2. Aggregate WSI using variance-weighted attention -> 1024-d slide embedding
    3. Z-score normalize both modalities across cohort
    4. Concatenate to form 1280-d patient signatures
    5. Cluster using HDBSCAN
    6. Save results for survival validation
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None,
                 use_gated_attention: bool = True):
        """
        Args:
            config: Pipeline configuration (uses default if None)
            use_gated_attention: Use gated attention (True) or simple variance (False)
        """
        self.config = config or get_config()
        
        # Initialize components
        self.data_loader = get_data_loader(self.config, mode="hybrid")
        
        if use_gated_attention:
            self.aggregator = GatedAttentionAggregator(self.config)
        else:
            self.aggregator = WSIAggregator(self.config)
            
        self.fusion = MultimodalFusion(self.config)
        self.clustering = ClusteringEngine(self.config)
        
        # Results storage
        self.aggregation_results: Dict[str, AggregationResult] = {}
        self.fusion_results: Dict[str, FusionResult] = {}
        self.patient_signatures: Optional[np.ndarray] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.processed_ids: List[str] = []
        
    def _process_patient(self, patient: PatientData) -> Optional[Tuple[np.ndarray, np.ndarray, AggregationResult]]:
        """
        Process a single patient through aggregation.
        
        Returns:
            Tuple of (wsi_embedding, rna_embedding, aggregation_result) or None
        """
        if not patient.has_wsi or not patient.has_rna:
            logger.warning(f"Skipping {patient.patient_id}: missing modality")
            return None
        
        # Aggregate WSI features
        agg_result = self.aggregator.aggregate(
            patient.wsi_features, 
            patient.patient_id
        )
        
        # Get RNA embedding as numpy
        rna_embedding = patient.rna_embedding.numpy()
        if rna_embedding.ndim > 1:
            rna_embedding = rna_embedding.flatten()
        
        return agg_result.slide_embedding, rna_embedding, agg_result
    
    def run(self, patient_ids: Optional[List[str]] = None,
            save_results: bool = True) -> pd.DataFrame:
        """
        Execute the complete pipeline.
        
        Args:
            patient_ids: Specific patients to process (None = all available)
            save_results: Whether to save intermediate results
            
        Returns:
            DataFrame with patient_id and cluster assignments
        """
        logger.info("=" * 60)
        logger.info("Starting Unsupervised Multimodal Fusion Pipeline")
        logger.info("=" * 60)
        
        # Get patient IDs
        if patient_ids is None:
            # Get patients with both modalities available locally
            available_wsi = set(self.data_loader.local_loader.get_available_wsi_ids())
            available_rna = set(self.data_loader.local_loader.get_available_rna_ids())
            patient_ids = sorted(list(available_wsi & available_rna))
            logger.info(f"Found {len(patient_ids)} patients with complete data")
        
        if len(patient_ids) == 0:
            logger.error("No patients with complete data found!")
            raise ValueError("No patients with complete data")
        
        # Phase 1: Aggregate WSI features for all patients
        logger.info("\n[Phase 1] Aggregating WSI features...")
        wsi_embeddings = []
        rna_embeddings = []
        valid_ids = []
        
        for pid in patient_ids:
            patient = self.data_loader.load_patient(pid)
            result = self._process_patient(patient)
            
            if result is not None:
                wsi_emb, rna_emb, agg_result = result
                wsi_embeddings.append(wsi_emb)
                rna_embeddings.append(rna_emb)
                valid_ids.append(pid)
                self.aggregation_results[pid] = agg_result
                
                logger.info(f"  ✓ {pid}: {agg_result.n_patches:,} patches aggregated")
        
        self.processed_ids = valid_ids
        n_patients = len(valid_ids)
        logger.info(f"\nProcessed {n_patients} patients successfully")
        
        # Convert to arrays
        wsi_matrix = np.stack(wsi_embeddings)  # [n_patients, 1024]
        rna_matrix = np.stack(rna_embeddings)  # [n_patients, 256]
        
        # Phase 2: Fit scalers and fuse modalities
        logger.info("\n[Phase 2] Fusing modalities with Z-score normalization...")
        self.fusion.fit_scalers(wsi_matrix, rna_matrix)
        
        signatures = []
        for i, pid in enumerate(valid_ids):
            agg = self.aggregation_results[pid]
            fusion_result = self.fusion.transform(
                wsi_embeddings[i],
                rna_embeddings[i],
                pid,
                attention_weights=agg.attention_weights,
                top_k_indices=agg.top_k_indices
            )
            self.fusion_results[pid] = fusion_result
            signatures.append(fusion_result.fused_signature)
        
        self.patient_signatures = np.stack(signatures)  # [n_patients, 1280]
        logger.info(f"Created {n_patients} x {self.patient_signatures.shape[1]} signature matrix")
        
        # Phase 3: Clustering
        logger.info("\n[Phase 3] Clustering patient signatures...")
        self.cluster_labels = self.clustering.fit_predict(self.patient_signatures)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'patient_id': valid_ids,
            'cluster': self.cluster_labels
        })
        
        # Add cluster probabilities if available
        if self.clustering.probabilities is not None:
            results_df['cluster_probability'] = self.clustering.probabilities
        
        # Phase 4: Save results
        if save_results:
            self._save_results(results_df)
        
        # Summary
        self._print_summary(results_df)
        
        return results_df
    
    def _save_results(self, results_df: pd.DataFrame):
        """Save all pipeline outputs."""
        logger.info("\n[Phase 4] Saving results...")
        
        # Ensure directories exist
        self.config.paths.ensure_dirs()
        
        # Save cluster assignments
        results_df.to_csv(self.config.paths.clusters_output, index=False)
        logger.info(f"  → Saved clusters to {self.config.paths.clusters_output}")
        
        # Save patient signatures
        sig_path = self.config.paths.patient_signatures_dir / "signatures.npy"
        np.save(sig_path, self.patient_signatures)
        logger.info(f"  → Saved signatures to {sig_path}")
        
        # Save patient ID mapping
        id_path = self.config.paths.patient_signatures_dir / "patient_ids.npy"
        np.save(id_path, np.array(self.processed_ids))
        logger.info(f"  → Saved patient IDs to {id_path}")
        
        # Save attention results for each patient
        for pid, agg in self.aggregation_results.items():
            attn_path = self.config.paths.attention_results_dir / f"{pid}_attention.npz"
            np.savez(
                attn_path,
                weights=agg.attention_weights,
                top_k_indices=agg.top_k_indices,
                n_patches=agg.n_patches
            )
        logger.info(f"  → Saved {len(self.aggregation_results)} attention files")
    
    def _print_summary(self, results_df: pd.DataFrame):
        """Print pipeline summary."""
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        
        cluster_counts = results_df['cluster'].value_counts().sort_index()
        
        for cluster, count in cluster_counts.items():
            if cluster == -1:
                label = "Noise/Outliers"
            else:
                label = f"Cluster {cluster}"
            logger.info(f"  {label}: {count} patients ({100*count/len(results_df):.1f}%)")
        
        logger.info(f"\nTotal patients processed: {len(results_df)}")
        logger.info(f"Signature dimensions: {self.patient_signatures.shape[1]}")
        logger.info("=" * 60)


def run_pipeline(config: Optional[PipelineConfig] = None,
                 patient_ids: Optional[List[str]] = None,
                 use_gated_attention: bool = True) -> pd.DataFrame:
    """
    Convenience function to run the complete pipeline.
    
    Args:
        config: Optional pipeline configuration
        patient_ids: Optional specific patient IDs to process
        use_gated_attention: Use gated attention mechanism
        
    Returns:
        DataFrame with clustering results
    """
    pipeline = UnsupervisedFusionPipeline(
        config=config,
        use_gated_attention=use_gated_attention
    )
    return pipeline.run(patient_ids=patient_ids)


if __name__ == "__main__":
    # Run pipeline with default settings
    results = run_pipeline()
    print(f"\nResults saved to: {get_config().paths.clusters_output}")

