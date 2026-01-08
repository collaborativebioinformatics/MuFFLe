# Task and Setup
We modify Task 2 of CHIMERA, using Task 3 dataset. For the proof-of-concept, we will only use RNASeq and Clinical Data (CD). If that works, we can use images (or leave that for future directions).

## Examining the Data
We first examine the clinical data:
```json
{
  "age": 72,
  "sex": "Male",
  "smoking": "No",
  "tumor": "Primary",
  "stage": "T1HG",
  "substage": "T1m",
  "grade": "G3",
  "reTUR": "Yes",
  "LVI": "No",
  "variant": "UCC",
  "EORTC": "Highest risk",
  "no_instillations": 24.0,
  "BRS": "BRS2",
  // the 13 features above comprise the 13-dimension inputs to our CD embedding submodel
  "progression": 0,               // Task 2 Answer, binary classification objective. 1 if the disease progressed, 0 otherwise.
  "Time_to_prog_or_FUend": 110    // Task 3 Answer, used in Cox survival loss
}
```
And now we examine a (truncated) RNASeq data:
```json
{
  "A1BG": 7.61361045919288,     // log-transformed gene expression data 
  "A1CF": 6.7250557547282,      // for each of the ~20k (n = 19359)  
  "A2M": 10.6539141169293,      // protein-coding genes in the human genome
  "A2ML1": 8.25324694246224,    
  "A3GALT2": 3.69463296517318,
  "A4GALT": 8.21555743274119,
  "A4GNT": 4.72369439314022,
  "AAAS": 8.22980706206347,
  "AACS": 10.4268475492017,
  "AADAC": 6.59704897422347,
  "AADACL2": 3.69463296517318,
  "AADACL3": 4.65012559819883,
  "AADACL4": 3.69463296517318,
  "AADAT": 7.85988840290319,
  "AAGAB": 8.28543364189896,
  "AAK1": 12.0198717741375,
  "AAMDC": 7.42771364410383,
  "AAMP": 9.33111699113079,
  "AANAT": 5.44797154135986,
  "AAR2": 9.12070329741041,
  "AARD": 5.517116299221,
  .
  .
  .
}
```
Each datapoint has an ID. The task 3 data has 177 IDs, which can be accessed using the notebook `helpers/get_ids.py`.

## Example Use Case
Consider we have 2 hospitals $X$ and $Y$, each with different screening capabilities. Each is able to provide clinical data and image histology data, but hospital $Y$ can additionally provide RNA sequencing data. We'd like to train a model to detect the likelihood of disease progression, so we will train a binary classifier based on the clinical data's ground truth `progression` key. 

We adapt the diagram from [CHIMERA Task 3](https://chimera.grand-challenge.org/task-3-bladder-cancer-recurrence-prediction/) to summarize the task:

![Adapted from CHIMERA Task 3](figures/chimera-task-3-adapted.png)

### Federation + Data Splits
We will simulate federation by splitting the data, assuming that patients are either from hospital $X$ or hospital $Y$.

We will split the `n=177` datapoints into 4 buckets:
1. IDs for hospital $X$
2. IDs for hospital $Y$
4. validation data between federated learning rounds
5. test data for the final evaluation


## Fusion Model Architecture
We will explain each part of the diagram below:

![Model Architecture](figures/model-architecture.png)

### Level 1: Embeddings
Each data modality needs to be embedded in some way. An example might be:
- Images can use a pretrained CNN,
- RNASeq can use some sort of MLP
- Clinical Data can also use some sort of MLP

### Level 2: Interpretable + Learnable Weights Per-Modality
We create a learnable layer that gives an easily-interpreted weight, or importance score, to each modality's embedding.

### Level 3: Concatenation
We just concatenate each modality's embedding together.

### Level 4: Attention Backbone
We use an attention mechanism (simplest example: a linear layer) to let the fusion model learn what parts of the embedding are most important

### Level 5: Sigmoid for Binary Classification
We then use an MLP to project the result of the attention backbone down to a single logit, which is then clamped to a 0 or 1 prediction.

## Training + Evaluation
The overall workflow is as follows:
- Clients receive a copy of the fusion model
- They run $k$ epochs of training using only the modalities they have 
  - for example, Hospital $X$ will just have zero'd RNASeq embeddings,
  - whereas Hospital $Y$ will have all modalities used.
- these updated weights are sent to the NVFlare server
  - and updated using FedAvg
  - then the server runs a validation to see the performance of the model
- after $n$ rounds of federated learning, we evaluate on the test set. Recall that since our fusion model is a binary classifier, this is just binary classification accuracy, plus confidence scores using the non-sigmoid logits.