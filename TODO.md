# Task and Setup
We modify Task 2 of CHIMERA, using Task 3 dataset. For the proof-of-concept, we will only use RNASeq and Clinical Data (CD). If that works, we can use images (or leave that for future directions).

## Examining the Data
We first examine the clinical data for a single id `3A_001`:
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

Looking at `raw/metadata.csv`, The distribution of the data is `n=126` people in Cohort A, and `n=50` people in Cohort B. 

## Example Use Case
> At a high level: we simulate an example where federated + multimodal learning 
> mitigates data gaps stemming from unequal access to healthcare, bypassing
> the data harmonization step when certain data modalities are missing.

Consider we have 2 hospitals, one in San Diego and one in Memphis. 

For the sake of this example, we make some assumptions. Since San Diego is a larger city, and thus its hospital will (1) see more people and (2) have more screening equipment than the hospital in Memphis.

To simulate such an environment, we will assign `n=100` people in Cohort A to the San Diego hospital, and all `n=50` people in Cohort B to the Memphis hospital. 

The remaining `n=26` people from CHIMERA Task 3, Cohort A will be held out as an evaluation set.

For the San Diego hospital, we are given access to both RNA sequencing and clinical data (CD), whereas for the Memphis hospital, we are only given access to clinical data.

We would like to train a federated model to predict the cases where the disease will progress, akin to Task 2 of the CHIMERA challenge. Formally, we are training a binary classifier, using the `progression` key of the clinical data as the label.

For now, we have elected to avoid processing images. But this is an easily extendable part of the pipeline as we will soon see.

We adapt the diagram from [CHIMERA Task 3](https://chimera.grand-challenge.org/task-3-bladder-cancer-recurrence-prediction/) to summarize the task:

![Adapted from CHIMERA Task 3](figures/chimera-task-3-adapted.png)


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
