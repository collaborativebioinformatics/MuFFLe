import torch
import json
import os
from torch.utils.data import Dataset

class BaseMedicalDataset(Dataset):
    """Helper class to load JSON files."""
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)
    
    def _load_json(self, idx):
        with open(self.file_paths[idx], 'r') as f:
            return json.load(f)

class ClinicalDataset(BaseMedicalDataset):
    """Dataloader for Clinical data (Dimension: 13)"""
    def _process_features(self, data):
        # Categorical mappings based on provided JSON structure
        sex_map = {"Male": 0, "Female": 1}
        tumor_map = {"Primary": 0, "Recurrent": 1}
        reTUR_map = {"No": 0, "Yes": 1}
        lvi_map = {"No": 0, "Yes": 1}
        grade_map = {"G1": 1, "G2": 2, "G3": 3}
        
        # Mapping 13 features
        features = [
            float(data['age']),
            float(sex_map.get(data['sex'], 0)),
            float(data['smoking']),
            float(tumor_map.get(data['tumor'], 0)),
            1.0 if "T1" in data['stage'] else 0.0,
            1.0 if data['substage'] == "T1e" else 0.0,
            float(grade_map.get(data['grade'], 0)),
            float(reTUR_map.get(data['reTUR'], 0)),
            float(lvi_map.get(data['LVI'], 0)),
            1.0 if data['variant'] == "UCC" else 0.0,
            float(data['no_instillations']),
            float(data['progression']),
            float(data['Time_to_prog_or_FUend'])
        ]
        return torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, idx):
        data = self._load_json(idx)
        return self._process_features(data)

class RNADataset(BaseMedicalDataset):
    """Dataloader for RNA data (Dimension: 19359)"""
    def __init__(self, file_paths, gene_list=None):
        super().__init__(file_paths)
        # Ensure consistent gene order across all samples
        if gene_list is None:
            first_sample = self._load_json(0)
            self.gene_list = sorted(first_sample.keys())
        else:
            self.gene_list = gene_list

    def __getitem__(self, idx):
        data = self._load_json(idx)
        # Extract values in the specific gene order
        rna_values = [data.get(gene, 0.0) for gene in self.gene_list]
        return torch.tensor(rna_values, dtype=torch.float32)