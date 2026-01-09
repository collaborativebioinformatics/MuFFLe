"""
Data Loading Utilities for CHIMERA Task 3 Multimodal Fusion Pipeline.

Handles loading of:
- WSI features (N x 1024 UNI embeddings) from local or S3
- RNA embeddings (256-d pre-computed vectors)
- Clinical data (survival outcomes)
- Patch coordinates (for spatial mapping)
"""

import os
import io
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatientData:
    """Container for a single patient's multimodal data."""
    patient_id: str
    wsi_features: Optional[torch.Tensor] = None  # [N, 1024]
    rna_embedding: Optional[torch.Tensor] = None  # [256]
    coordinates: Optional[np.ndarray] = None  # Structured array
    clinical_data: Optional[Dict] = None
    
    @property
    def has_wsi(self) -> bool:
        return self.wsi_features is not None
    
    @property
    def has_rna(self) -> bool:
        return self.rna_embedding is not None
    
    @property
    def n_patches(self) -> int:
        return self.wsi_features.shape[0] if self.has_wsi else 0


class LocalDataLoader:
    """
    Data loader for local file system.
    Use this when data is downloaded to local storage.
    """
    
    def __init__(self, config):
        """
        Args:
            config: PipelineConfig instance with path configurations
        """
        self.config = config
        self.paths = config.paths
        self._clinical_df = None
        self._patient_ids = None
        
    @property
    def clinical_df(self) -> pd.DataFrame:
        """Lazy load clinical data."""
        if self._clinical_df is None:
            self._clinical_df = pd.read_csv(self.paths.clinical_data)
        return self._clinical_df
    
    @property
    def patient_ids(self) -> List[str]:
        """Get list of all patient IDs from clinical data."""
        if self._patient_ids is None:
            self._patient_ids = self.clinical_df[
                self.config.survival.patient_id_column
            ].tolist()
        return self._patient_ids
    
    def get_available_wsi_ids(self) -> List[str]:
        """Get patient IDs that have WSI features available locally."""
        available = []
        for pid in self.patient_ids:
            feat_path = self.paths.wsi_features_dir / f"{pid}_HE.pt"
            if feat_path.exists():
                available.append(pid)
        return available
    
    def get_available_rna_ids(self) -> List[str]:
        """Get patient IDs that have RNA embeddings available locally."""
        available = []
        for pid in self.patient_ids:
            rna_path = self.paths.rna_embeddings_dir / f"{pid}.pt"
            if rna_path.exists():
                available.append(pid)
        return available
    
    def get_complete_ids(self) -> List[str]:
        """Get patient IDs with both WSI and RNA data available."""
        wsi_ids = set(self.get_available_wsi_ids())
        rna_ids = set(self.get_available_rna_ids())
        return sorted(list(wsi_ids & rna_ids))
    
    def load_wsi_features(self, patient_id: str) -> Optional[torch.Tensor]:
        """
        Load WSI patch features for a patient.
        
        Args:
            patient_id: Patient identifier (e.g., "3A_001")
            
        Returns:
            Tensor of shape [N, 1024] or None if not available
        """
        feat_path = self.paths.wsi_features_dir / f"{patient_id}_HE.pt"
        
        if not feat_path.exists():
            logger.warning(f"WSI features not found for {patient_id}")
            return None
        
        try:
            features = torch.load(feat_path, map_location="cpu", weights_only=False)
            
            # Handle different storage formats
            if isinstance(features, dict):
                features = features.get("features", features.get("embeddings", None))
            
            if features is not None and not isinstance(features, torch.Tensor):
                features = torch.tensor(features)
            
            logger.debug(f"Loaded WSI features for {patient_id}: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"Error loading WSI features for {patient_id}: {e}")
            return None
    
    def load_rna_embedding(self, patient_id: str) -> Optional[torch.Tensor]:
        """
        Load pre-computed RNA embedding for a patient.
        
        Expected format: {"patient_id": str, "embedding": Tensor[256], 
                         "genes": List[str], "encoder": str}
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Tensor of shape [256] or None if not available
        """
        rna_path = self.paths.rna_embeddings_dir / f"{patient_id}.pt"
        
        if not rna_path.exists():
            logger.warning(f"RNA embedding not found for {patient_id}")
            return None
        
        try:
            data = torch.load(rna_path, map_location="cpu", weights_only=False)
            
            # Handle different storage formats
            if isinstance(data, dict):
                embedding = data.get("embedding", None)
                if embedding is None:
                    # Try other common keys
                    for key in ["rna_embedding", "features", "vector"]:
                        if key in data:
                            embedding = data[key]
                            break
            elif isinstance(data, torch.Tensor):
                embedding = data
            else:
                embedding = torch.tensor(data)
            
            if embedding is not None and not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding)
            
            # Ensure 1D
            if embedding is not None and embedding.dim() > 1:
                embedding = embedding.squeeze()
            
            logger.debug(f"Loaded RNA embedding for {patient_id}: {embedding.shape}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error loading RNA embedding for {patient_id}: {e}")
            return None
    
    def load_coordinates(self, patient_id: str) -> Optional[np.ndarray]:
        """
        Load patch coordinates for a patient.
        
        Coordinate format: structured array with fields
            ('x', 'y', 'tile_size_resized', 'tile_level', 
             'resize_factor', 'tile_size_lv0', 'target_spacing')
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Structured numpy array or None if not available
        """
        coord_path = self.paths.wsi_coordinates_dir / f"{patient_id}_HE.npy"
        
        if not coord_path.exists():
            logger.warning(f"Coordinates not found for {patient_id}")
            return None
        
        try:
            coordinates = np.load(coord_path, allow_pickle=True)
            logger.debug(f"Loaded coordinates for {patient_id}: {coordinates.shape}")
            return coordinates
            
        except Exception as e:
            logger.error(f"Error loading coordinates for {patient_id}: {e}")
            return None
    
    def load_clinical_data(self, patient_id: str) -> Optional[Dict]:
        """
        Load clinical data for a patient.
        
        Args:
            patient_id: Patient identifier
            
        Returns:
            Dictionary with clinical features
        """
        patient_row = self.clinical_df[
            self.clinical_df[self.config.survival.patient_id_column] == patient_id
        ]
        
        if patient_row.empty:
            logger.warning(f"Clinical data not found for {patient_id}")
            return None
        
        return patient_row.iloc[0].to_dict()
    
    def load_patient(self, patient_id: str, 
                     load_wsi: bool = True,
                     load_rna: bool = True,
                     load_coords: bool = True,
                     load_clinical: bool = True) -> PatientData:
        """
        Load all available data for a single patient.
        
        Args:
            patient_id: Patient identifier
            load_wsi: Whether to load WSI features
            load_rna: Whether to load RNA embedding
            load_coords: Whether to load coordinates
            load_clinical: Whether to load clinical data
            
        Returns:
            PatientData instance with available data
        """
        patient = PatientData(patient_id=patient_id)
        
        if load_wsi:
            patient.wsi_features = self.load_wsi_features(patient_id)
        
        if load_rna:
            patient.rna_embedding = self.load_rna_embedding(patient_id)
        
        if load_coords:
            patient.coordinates = self.load_coordinates(patient_id)
        
        if load_clinical:
            patient.clinical_data = self.load_clinical_data(patient_id)
        
        return patient
    
    def iterate_patients(self, patient_ids: Optional[List[str]] = None,
                         require_complete: bool = False):
        """
        Generator that yields PatientData for each patient.
        
        Args:
            patient_ids: Specific IDs to iterate, or None for all
            require_complete: If True, only yield patients with both modalities
            
        Yields:
            PatientData instances
        """
        if patient_ids is None:
            if require_complete:
                patient_ids = self.get_complete_ids()
            else:
                patient_ids = self.patient_ids
        
        for pid in patient_ids:
            yield self.load_patient(pid)


class S3DataLoader:
    """
    Data loader for S3 cloud storage.
    Use this when running on cloud compute with data in S3.
    """
    
    def __init__(self, config, anon: bool = True):
        """
        Args:
            config: PipelineConfig instance
            anon: Whether to use anonymous access (public bucket)
        """
        self.config = config
        self.paths = config.paths
        self.bucket = self.paths.s3_bucket
        self.prefix = self.paths.s3_prefix
        
        # Initialize S3 filesystem
        try:
            import s3fs
            self.fs = s3fs.S3FileSystem(anon=anon)
            self._s3_available = True
        except ImportError:
            logger.warning("s3fs not installed. S3 loading disabled.")
            self._s3_available = False
        
        self._clinical_df = None
    
    @property
    def is_available(self) -> bool:
        return self._s3_available
    
    def _s3_path(self, subpath: str) -> str:
        """Construct full S3 path."""
        return f"{self.bucket}/{self.prefix}{subpath}"
    
    def load_wsi_features(self, patient_id: str) -> Optional[torch.Tensor]:
        """Load WSI features from S3."""
        if not self._s3_available:
            return None
        
        path = self._s3_path(f"features/features/{patient_id}_HE.pt")
        
        try:
            with self.fs.open(path, "rb") as f:
                features = torch.load(io.BytesIO(f.read()), 
                                     map_location="cpu", 
                                     weights_only=False)
            
            if isinstance(features, dict):
                features = features.get("features", features.get("embeddings", None))
            
            return features
            
        except Exception as e:
            logger.error(f"Error loading WSI from S3 for {patient_id}: {e}")
            return None
    
    def load_coordinates(self, patient_id: str) -> Optional[np.ndarray]:
        """Load coordinates from S3."""
        if not self._s3_available:
            return None
        
        path = self._s3_path(f"features/coordinates/{patient_id}_HE.npy")
        
        try:
            with self.fs.open(path, "rb") as f:
                coordinates = np.load(io.BytesIO(f.read()), allow_pickle=True)
            return coordinates
            
        except Exception as e:
            logger.error(f"Error loading coordinates from S3 for {patient_id}: {e}")
            return None


class HybridDataLoader:
    """
    Hybrid data loader that tries local first, then falls back to S3.
    Optimal for cloud compute with partial local cache.
    """
    
    def __init__(self, config, use_s3_fallback: bool = True):
        """
        Args:
            config: PipelineConfig instance
            use_s3_fallback: Whether to try S3 if local file not found
        """
        self.config = config
        self.local_loader = LocalDataLoader(config)
        self.s3_loader = S3DataLoader(config) if use_s3_fallback else None
        self._clinical_df = None
    
    @property
    def clinical_df(self) -> pd.DataFrame:
        return self.local_loader.clinical_df
    
    @property
    def patient_ids(self) -> List[str]:
        return self.local_loader.patient_ids
    
    def load_wsi_features(self, patient_id: str) -> Optional[torch.Tensor]:
        """Try local first, then S3."""
        features = self.local_loader.load_wsi_features(patient_id)
        
        if features is None and self.s3_loader and self.s3_loader.is_available:
            logger.info(f"Falling back to S3 for WSI: {patient_id}")
            features = self.s3_loader.load_wsi_features(patient_id)
        
        return features
    
    def load_rna_embedding(self, patient_id: str) -> Optional[torch.Tensor]:
        """RNA embeddings are only local (pre-computed)."""
        return self.local_loader.load_rna_embedding(patient_id)
    
    def load_coordinates(self, patient_id: str) -> Optional[np.ndarray]:
        """Try local first, then S3."""
        coords = self.local_loader.load_coordinates(patient_id)
        
        if coords is None and self.s3_loader and self.s3_loader.is_available:
            logger.info(f"Falling back to S3 for coordinates: {patient_id}")
            coords = self.s3_loader.load_coordinates(patient_id)
        
        return coords
    
    def load_clinical_data(self, patient_id: str) -> Optional[Dict]:
        return self.local_loader.load_clinical_data(patient_id)
    
    def load_patient(self, patient_id: str, **kwargs) -> PatientData:
        """Load patient with hybrid approach."""
        patient = PatientData(patient_id=patient_id)
        
        if kwargs.get('load_wsi', True):
            patient.wsi_features = self.load_wsi_features(patient_id)
        
        if kwargs.get('load_rna', True):
            patient.rna_embedding = self.load_rna_embedding(patient_id)
        
        if kwargs.get('load_coords', True):
            patient.coordinates = self.load_coordinates(patient_id)
        
        if kwargs.get('load_clinical', True):
            patient.clinical_data = self.load_clinical_data(patient_id)
        
        return patient


def get_data_loader(config, mode: str = "hybrid") -> Union[LocalDataLoader, 
                                                            S3DataLoader, 
                                                            HybridDataLoader]:
    """
    Factory function to get appropriate data loader.
    
    Args:
        config: PipelineConfig instance
        mode: One of "local", "s3", or "hybrid"
        
    Returns:
        Data loader instance
    """
    if mode == "local":
        return LocalDataLoader(config)
    elif mode == "s3":
        return S3DataLoader(config)
    elif mode == "hybrid":
        return HybridDataLoader(config)
    else:
        raise ValueError(f"Unknown loader mode: {mode}")

