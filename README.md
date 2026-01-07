# Multimodal Framework for Federated Learning (MuFFLe)

## TODO List + References:
- [X] Reference previous years' repos as inspo for this GitHub repo
  - [GeneVar2](https://github.com/collaborativebioinformatics/GeneVar2), a little *too* good because it was a 2nd followup project
  - [Haploblock Clusters](https://github.com/collaborativebioinformatics/Haploblock_Clusters_ElixirBH25)
- [X] Stick w the default MIT License, just don't use GPL or LGPL because industry can't use it 
- [ ] Scaffold

![flowchart](figures/flowchart.png)

# Quick Start
- [ ] 

# Directory Structure
- [ ] Generate with Tree

# Introduction (1 para)

# Methods (2 paras)

## Selecting a Multimodal Dataset for Validation
We decided to go with the data for the [CHIMERA Challenge](https://registry.opendata.aws/chimera), which stands for
> Combining HIstology, Medical imaging (radiology) and molEcular data for medical pRognosis and diAgnosis
Details for the challenge are [here](https://chimera.grand-challenge.org/).

*author's note, what a forced acronym :-)*

This data was open-access and easily available on [AWS Open Data](https://registry.opendata.aws/). 

We opted for [Task 3](https://chimera.grand-challenge.org/task-3-bladder-cancer-recurrence-prediction/) of this challenge. See [How we built this tool](#how-to-use-this-tool) for the reasons why we chose this task.


# How we built this tool
We started by brainstorming potential ways to integrate multimodal data. We considered natively multimodal models, like Vision Transformers (ViT), but we opted not to do such a thing for several considerations:
1. Cost: fine-tuning large ViTs, even open-source ones such as [Qwen-3 VL](https://github.com/QwenLM/Qwen3-VL) is computationally expensive 
2. Catastrophic forgetting: similar to how RL updates may undo themselves over time, updates from different modalities might "cancel out" and lead to more confusion than actually meaningful results. 

As a result, we opted for an approach that better leverages smaller, more specialized models. This led us to the diagram below:
![Methods Architecture Draft 1](figures/methods-architecture-draft-1.png)

Which naturally lent itself to the very similar diagram from [CHIMERA, task 3](https://chimera.grand-challenge.org/task-3-bladder-cancer-recurrence-prediction/). **(Half serious) we want to emphasize that we dreamt up the diagram above before running into CHIMERA.**
![CHIMERA Task 3 Diagram](figures/chimera-task-3.png)

# How to use this tool

# Future Directions
There are some low-hanging fruit that this could be applied to. While searching for instances to create our proof-of-concept, we came across some data from the Real-time Analysis and Discovery in Integrated And Networked Technologies (RADIANT) group, which 
> seeks to develop an extensible, federated framework for rapid exchange of multimodal clinical and research data on behalf of accelerated discovery and patient impact. 
[RADIANT Public Data (AWS)](https://registry.opendata.aws/radiant/)

We elected not to use this dataset because the S3 bucket had "controlled access," which required filling out a form for approval and did not fit the fast-paced nature of the Hackathon. However, our federated learning framework could be easily extended to RADIANT's data, which contains
> Clinical data, Imaging data, Histology data, Genomic data, Proteomics data, and more [Children's Brain Tumor Network (CBTN)](https://cbtn.org/research-resources)

# References (BibTeX)