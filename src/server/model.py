"""
Adapted from CHIMERA Task 2 baseline 
https://github.com/DIAGNijmegen/CHIMERA/tree/main/task2_baseline/prediction_model/Aggregators/training/mil_models
model_abmil_fusion.py
tabular_snn.py
"""
import torch
import torch.nn as nn

from server.embedRNA import RNANet
from server.embedCD import CDNet

class FusionNet(nn.Module):
    def __init__(
            self,
            rna_in_dim=19359,
            clinical_in_dim=14, 
            embedding_dim=512, # Target dim for RNA embedding
            dropout_CD=0.3,
            dropout_RNA=0.3
        ):
        super().__init__()

        # === 1. Embeddings ===
        # RNA Branch
        self.rna_out_dim = embedding_dim 
        self.rna_embedding = RNANet(
            in_dim=rna_in_dim, 
            out_dim=self.rna_out_dim, 
            dropout_p=dropout_RNA
        )

        # Clinical Branch
        # Note: CDNet output is hardcoded to 512 in embedCD.py
        self.tabular_net = CDNet(
            in_dim=clinical_in_dim, 
            dropout_p=dropout_CD
        )

        # Determine Clinical Output Dimension dynamically
        with torch.no_grad():
            self.tabular_net.eval()
            dummy_input = torch.zeros(1, clinical_in_dim)
            self.clinical_out_dim = self.tabular_net(dummy_input).shape[1]
            self.tabular_net.train()

        # === 2. Interpretable Weights Per-Modality ===
        # Learnable scalar weights, initialized to 1.0
        self.w_rna = nn.Parameter(torch.tensor(1.0))
        self.w_clinical = nn.Parameter(torch.tensor(1.0))
        
        # === 4. Attention Backbone ===
        # "Learn what parts of the embedding are most important"
        # We use a Gating mechanism (Sigmoid) effectively acting as feature attention
        fusion_dim = self.rna_out_dim + self.clinical_out_dim
        self.attention_backbone = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.Sigmoid() 
        )

        # === 5. Classifier ===
        self.classifier = nn.Linear(fusion_dim, 1)

    def forward(self, x_clinical, x_rna=None):
        
        # --- Level 1: Embeddings ---
        # Clinical
        z_tab = self.tabular_net(x_clinical)

        # RNA (Handle missing modality)
        if x_rna is not None:
            z_rna = self.rna_embedding(x_rna)
        else:
            # Create zeros on the correct device
            z_rna = torch.zeros(
                (x_clinical.shape[0], self.rna_out_dim), 
                device=x_clinical.device
            )

        # --- Level 2: Per-Modality Weights ---
        z_rna = z_rna * self.w_rna
        z_tab = z_tab * self.w_clinical

        # --- Level 3: Concatenation ---
        z_fusion = torch.cat((z_rna, z_tab), dim=-1)

        # --- Level 4: Attention Backbone ---
        # Reweight the fused vector based on feature importance
        attention_weights = self.attention_backbone(z_fusion)
        z_attended = z_fusion * attention_weights

        # --- Level 5: Prediction ---
        logits = self.classifier(z_attended)

        return {
            'logits': logits, 
            'modality_weights': {'rna': self.w_rna, 'clinical': self.w_clinical},
            'attention_weights': attention_weights
        }