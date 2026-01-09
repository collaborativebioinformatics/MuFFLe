# CHIMERA Task 3: Architecture Explanation

## 🔗 RNA + Image Embedding Concatenation

### Step-by-Step Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL FUSION PROCESS                     │
└─────────────────────────────────────────────────────────────────┘

STEP 1: WSI Aggregation (Variable → Fixed)
───────────────────────────────────────────
Input:  WSI Patches (N × 1024)  [N varies per patient: 24k-250k patches]
        │
        ▼
[Gated Attention Pooling - HEURISTIC]
        │
        ├─ Compute patch statistics:
        │   • Mean across 1024 features → (N,)
        │   • Std across 1024 features → (N,)
        │   • Max across 1024 features → (N,)
        │
        ├─ Compute attention weights:
        │   tanh_branch = tanh(mean / std(mean))
        │   sigmoid_branch = sigmoid((variance - mean(variance)) / std(variance))
        │   attention = softmax(tanh_branch * sigmoid_branch)
        │
        └─ Weighted sum: sum(patches × attention) → (1024,)

Output: Slide Embedding (1024,)  [FIXED SIZE]


STEP 2: Z-Score Normalization (Per Modality)
──────────────────────────────────────────────
WSI Embedding (1024,)
        │
        ▼
[StandardScaler.fit() on ALL patients]
        │
        ├─ Compute mean: μ_WSI = mean(all WSI embeddings)
        ├─ Compute std:  σ_WSI = std(all WSI embeddings)
        │
        └─ Normalize: (WSI - μ_WSI) / σ_WSI → (1024,)

RNA Embedding (256,)
        │
        ▼
[StandardScaler.fit() on ALL patients]
        │
        ├─ Compute mean: μ_RNA = mean(all RNA embeddings)
        ├─ Compute std:  σ_RNA = std(all RNA embeddings)
        │
        └─ Normalize: (RNA - μ_RNA) / σ_RNA → (256,)


STEP 3: Concatenation
──────────────────────
WSI Normalized: [w₁, w₂, ..., w₁₀₂₄]  (1024 values)
RNA Normalized: [r₁, r₂, ..., r₂₅₆]   (256 values)
        │
        ▼
[Concatenate]
        │
        └─→ [w₁, w₂, ..., w₁₀₂₄, r₁, r₂, ..., r₂₅₆]  (1280 values)

Output: Patient Signature (1280,)
```

### Code Implementation

```python
# From model/unsupervised_fusion.py

class MultimodalFusion:
    def fit_scalers(self, wsi_embeddings, rna_embeddings):
        """Fit Z-score scalers on the ENTIRE cohort"""
        # wsi_embeddings: (n_patients, 1024)
        # rna_embeddings: (n_patients, 256)
        
        self.wsi_scaler.fit(wsi_embeddings)  # Learns μ_WSI, σ_WSI
        self.rna_scaler.fit(rna_embeddings)  # Learns μ_RNA, σ_RNA
    
    def transform(self, wsi_embedding, rna_embedding):
        """Normalize and concatenate for ONE patient"""
        # Step 1: Z-score normalize each modality
        wsi_normalized = self.wsi_scaler.transform(wsi_embedding.reshape(1, -1))
        rna_normalized = self.rna_scaler.transform(rna_embedding.reshape(1, -1))
        
        # Step 2: Concatenate
        fused = np.concatenate([wsi_normalized, rna_normalized])
        # Result: (1280,) = (1024,) + (256,)
        
        return fused
```

### Why Z-Score Normalization Before Concatenation?

**Problem**: WSI (1024-d) and RNA (256-d) have different scales and distributions.

**Example**:
- WSI features might range: [-5, 5]
- RNA features might range: [0, 1000]

Without normalization, RNA features would dominate the concatenated vector!

**Solution**: Z-score normalization ensures both modalities:
1. Have mean = 0
2. Have std = 1
3. Are on the same scale

This prevents one modality from overwhelming the other in clustering.

---

## 🧠 Heuristic-Based vs Neural Network

### What is "Heuristic-Based"?

A **heuristic** is a rule-of-thumb or algorithm that doesn't require training. It uses fixed mathematical operations based on domain knowledge.

### Comparison Table

| Aspect | **Heuristic-Based (Current)** | **Neural Network** |
|--------|------------------------------|-------------------|
| **Parameters** | 0 trainable parameters | Millions of trainable weights |
| **Training** | None required | Requires backpropagation |
| **Attention** | Variance-based formula | Learned attention network |
| **Fusion** | Simple concatenation | Learned fusion layers |
| **Interpretability** | High (explicit formulas) | Lower (black box) |
| **Data Requirements** | Works immediately | Needs labeled data |

### Current Pipeline: Heuristic Attention

```python
# HEURISTIC APPROACH (No Training)
class GatedAttentionAggregator:
    def compute_attention_weights(self, features):
        # Fixed formula - no learned weights!
        patch_mean = features.mean(dim=1)
        patch_std = features.std(dim=1)
        
        # Heuristic: High variance = more informative
        tanh_branch = torch.tanh(patch_mean / patch_mean.std())
        sigmoid_branch = torch.sigmoid(
            (patch_std**2 - patch_std**2.mean()) / patch_std**2.std()
        )
        
        # Combine heuristics
        attention = softmax(tanh_branch * sigmoid_branch)
        return attention
```

**Key Insight**: The attention weights are computed using **statistical properties** (mean, variance) rather than learned neural network weights.

### If It Were a Neural Network:

```python
# NEURAL NETWORK APPROACH (Requires Training)
class AttentionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # LEARNED weights - these would be trained!
        self.attention_net = nn.Sequential(
            nn.Linear(1024, 512),  # Weight matrix W1
            nn.ReLU(),
            nn.Linear(512, 256),   # Weight matrix W2
            nn.ReLU(),
            nn.Linear(256, 1)      # Weight matrix W3
        )
    
    def forward(self, features):
        # Attention computed through learned transformations
        attention_scores = self.attention_net(features)
        attention = softmax(attention_scores)
        return attention
```

**Key Difference**: The neural network has **learnable weight matrices** (W1, W2, W3) that are optimized during training. The heuristic has **no weights** - it's just math!

---

## 📊 Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    PATIENT PROCESSING PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

Patient: 3A_001
────────────────

1. LOAD DATA
   ├─ WSI: (238,673 patches × 1024 features) = 238,673 × 1024
   └─ RNA: (256 features) = 1 × 256

2. AGGREGATE WSI (Heuristic Attention)
   Input:  (238,673, 1024)
   Process:
     • Compute variance per patch: var(1024 features) → (238,673,)
     • Apply gated attention formula (heuristic)
     • Weighted sum: sum(patches × attention) → (1024,)
   Output: (1024,) slide embedding

3. NORMALIZE (Z-Score)
   WSI: (1024,) → normalize → (1024,) [mean=0, std=1]
   RNA: (256,)  → normalize → (256,)  [mean=0, std=1]

4. CONCATENATE
   [WSI_normalized (1024,) + RNA_normalized (256,)] → (1280,)

5. CLUSTER (HDBSCAN)
   Input: All patients' (1280,) signatures
   Output: Cluster labels (e.g., Cluster 0, Cluster 1, Noise)
```

---

## 🎯 Why This Approach?

### Advantages of Heuristic-Based:

1. **No Training Required**: Works immediately on new data
2. **Interpretable**: We know exactly why patches get high attention (high variance)
3. **Fast**: No backpropagation, just matrix operations
4. **Robust**: Doesn't overfit to training data
5. **Domain Knowledge**: Uses biological insight (variance = morphological complexity)

### When You'd Use Neural Networks Instead:

- If you have **labeled training data** (e.g., "this patient has recurrence")
- If you want to **learn complex interactions** between modalities
- If you need **higher accuracy** and can afford training time
- If you want **end-to-end optimization** from raw data to prediction

---

## 🔍 Key Takeaway

**Current Pipeline**:
- **Heuristic Attention**: Uses variance/statistics (no training)
- **Simple Concatenation**: Just stacks normalized vectors
- **Z-Score Normalization**: Ensures fair fusion of modalities
- **HDBSCAN Clustering**: Finds natural groups (unsupervised)

**Result**: A fully unsupervised pipeline that works without any labeled data or training!

