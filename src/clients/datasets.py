import torch
import json
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
    """Dataloader for Clinical data (Dimension: 14)"""
    def _process_features(self, data):
        # NOTE: we will ignore the columns with keys
        # ['progression' and 'Time_to_prog_or_FUend']
        # see Methods for why. 
        # TLDR: those keysare labels, not features.

        # NOTE: we also throw out the "tumor" column,
        # since there is only one unique value for
        # that key (see exploration.ipynb). There might 
        # be "recurrent" in real clinical data,
        # but this is not helpful because we are trying
        # to predict tumor recurrence in our simulated task.

        """
        Feature Format (see 3A_001_CD.json for an example) 
        Descriptions and mappings generated from Google Gemini  
        {
          "age": int,
          "sex": {"Male", "Female"},
          "smoking": {"No", "Yes", "-1"},
          "tumor": "Primary",                       # IGNORED
          "stage": {"T1HG", "TaHG},                 # T1 is more invasive than Ta
          "substage": {"T1m", "T1e", "-1"},         # m - micro, e - extensive
          "grade": {"G2", "G3"},
          "reTUR": {"No", "Yes"},
          "LVI": {"No", "Yes", "-1"},               # Lymphovascular Invasion
          "variant": {"UCC", "UCC + Variant"},      # Variants usually imply higher risk
          "EORTC": {"High risk", "Highest risk"},   # Ordinal risk scale
          "no_instillations": float, could be -1,
          "BRS": {"BRS1", "BRS2", "BRS3},           # No order, so 1-hot encode
          "progression": {0, 1},                    # IGNORED
          "Time_to_prog_or_FUend": int              # IGNORED
        }
        This leaves us with 14 dimensions overall, 
        13 - 1 = 12 features + 2 extra dimensions
        for the one-hot encoded BRS feature.
        """

        mapping = {
            "sex": {"Male": 0, "Female": 1},
            "smoking": {"No": 0, "Yes": 1, "-1": 0.5},
            "stage": {"TaHG": 0, "T1HG": 1},
            "substage": {"T1m": 0, "T1e": 1, "-1": 0.5},
            "grade": {"G2": 0, "G3": 1},
            "reTUR": {"No": 0, "Yes": 1},
            "LVI": {"No": 0, "Yes": 1, "-1": 0.5},
            "variant": {"UCC": 0, "UCC + Variant": 1},
            "EORTC": {"High risk": 0, "Highest risk": 1}
        }

        vector = [
            data["age"],                          
            mapping["sex"][data["sex"]],
            mapping["smoking"][data["smoking"]],
            mapping["stage"][data["stage"]],
            mapping["substage"][data["substage"]],
            mapping["grade"][data["grade"]],
            mapping["reTUR"][data["reTUR"]],
            mapping["LVI"][data["LVI"]],
            mapping["variant"][data["variant"]],
            mapping["EORTC"][data["EORTC"]],
            data["no_instillations"]
        ]
        # NOTE: in DataLoader, we have a collate_fn that normalizes both
        # age and no_instillations features, per-batch. This collate-fn
        # is provided below as `clinical_collate_fn`
        
        brs_map = {"BRS1": [1, 0, 0], "BRS2": [0, 1, 0], "BRS3": [0, 0, 1]}
        vector.extend(brs_map.get(data["BRS"], [0, 0, 0]))

        assert len(vector) == 14

        return torch.tensor(vector, dtype=torch.float32)
  
    def _process_label(self, data):
        return float(data['progression'])

    def __getitem__(self, idx):
        data = self._load_json(idx)
        return self._process_features(data), self._process_label(data)

def clinical_collate_fn(batch):
    """
    Performs per-batch normalization on 
    Age (index 0) and Instillations (index 10).
    """
    features, labels = zip(*batch)
    features = torch.stack(features)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Indices to normalize (Age and no_instillations)
    norm_indices = [0, 10]

    for idx in norm_indices:
        col = features[:, idx]
        mean = col.mean()
        std = col.std()
        # Prevent division by zero if all values in batch are the same
        if std > 0:
            features[:, idx] = (col - mean) / std
        else:
            features[:, idx] = col - mean 

    return features, labels

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