import torch
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, rna_dim=19359, clinical_dim=13, risk_output_dim=1):
        super().__init__()

        assert rna_dim is not None, "rna_dim must be provided"
        assert clinical_dim is not None, "clinical_dim must be provided"

        # --- Dimensions for Fusion ---
        self.rna_out_dim = 256
        self.clinical_out_dim = 64

        # RNA projection network
        self.rna_net = nn.Sequential(
            nn.Linear(rna_dim, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.rna_out_dim) 
        ) # Output: 256

        # Clinical projection network
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.clinical_out_dim) 
        ) # Output: 64

        # Final risk prediction head
        self.risk_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.rna_out_dim + self.clinical_out_dim, risk_output_dim)
        ) # Input size: 512 (Path) + 256 (RNA) + 64 (Clinical) = 832

    def forward(self, x_rna=None, x_clinical=None, return_l1=False):
        """
        Args:
            x_rna: [B, rna_dim] - RNA sequencing data
            x_clinical: [B, clinical_dim] - Clinical features
        """
        
        # 1. Determine Batch Size and Device from whichever input is available
        # We assume at least one modality is present.
        if x_rna is not None:
            B, _ = x_rna.shape
            device = x_rna.device
        elif x_clinical is not None:
            B, _ = x_clinical.shape
            device = x_clinical.device
        else:
            raise ValueError("At least one input modality must be provided.")

        # === RNA Branch ===
        if x_rna is not None:
            # x_rna: [B, rna_dim]
            z_rna = self.rna_net(x_rna)                       # [B, 256]
        else:
            # Bypass: Create zero embedding directly
            z_rna = torch.zeros((B, self.rna_out_dim), device=device)

        # === Clinical Branch ===
        if x_clinical is not None:
            # x_clinical: [B, clinical_dim]
            z_clinical = self.clinical_net(x_clinical)        # [B, 64]
        else:
            # Bypass: Create zero embedding directly
            z_clinical = torch.zeros((B, self.clinical_out_dim), device=device)

        # === Feature Fusion ===
        # Concatenate: [B, 512] + [B, 256] + [B, 64] -> [B, 832]
        z = torch.cat([z_rna, z_clinical], dim=-1)

        # === Risk Prediction ===
        risk = self.risk_head(z).squeeze(1)                   # [B]

        return risk
        # # === Optional L1 penalty ===
        # # Note: If RNA is missing, we usually skip the L1 penalty for it to save compute
        # if return_l1:
        #     l1_penalty = sum(torch.norm(p, p=1) for p in self.rna_net.parameters())
        #     return {'risk': risk}, l1_penalty
        # else:
        #     return {'risk': risk}, None