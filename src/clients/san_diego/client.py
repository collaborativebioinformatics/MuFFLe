"""
Adapted from NVFlare "Hello PyTorch" (hello-pt) example code.

Simulates the San Diego Hospital as outlined in the Methods.
"""
import os 

import torch 
from torch import nn 

from clients.datasets import ClinicalDataset, RNADataset, clinical_collate_fn

def update_model(
    client_name: str, 
    dataset_path: str,
    model, 
    device,
    summary_writer,
    current_round : int,
):
    # ----------------------------------------------------------------------- #
    # Client-Specific Constants
    # ----------------------------------------------------------------------- #
    IDS = ['3A_001', '3A_002', '3A_003', '3A_004', '3A_005', '3A_006', '3A_007', '3A_008', '3A_009', '3A_010', '3A_011', '3A_012', '3A_013', '3A_014', '3A_015', '3A_016', '3A_017', '3A_018', '3A_019', '3A_020', '3A_021', '3A_022', '3A_023', '3A_024', '3A_025', '3A_026', '3A_027', '3A_028', '3A_029', '3A_030', '3A_031', '3A_033', '3A_034', '3A_035', '3A_036', '3A_037', '3A_038', '3A_039', '3A_040', '3A_041', '3A_042', '3A_043', '3A_044', '3A_045', '3A_046', '3A_047', '3A_049', '3A_050', '3A_052', '3A_053', '3A_055', '3A_056', '3A_057', '3A_058', '3A_059', '3A_060', '3A_061', '3A_062', '3A_063', '3A_064', '3A_066', '3A_067', '3A_068', '3A_070', '3A_071', '3A_072', '3A_073', '3A_074', '3A_075', '3A_076', '3A_077', '3A_087', '3A_088', '3A_089', '3A_091', '3A_092', '3A_093', '3A_094', '3A_095', '3A_097', '3A_098', '3A_100', '3A_105', '3A_108', '3A_110', '3A_111', '3A_113', '3A_114', '3A_115', '3A_116', '3A_123', '3A_124', '3A_125', '3A_126', '3A_127', '3A_129', '3A_130', '3A_134', '3A_135', '3A_136']
    assert len(IDS) == 100, f"expected 100 IDS in San Diego (CHIMERA Task 3 Cohort A subset), got {len(IDS)}"

    # ----------------------------------------------------------------------- #
    # Initialize Model
    # ----------------------------------------------------------------------- #

    # Free for Client to modify
    config = {
        "batch_size": 32,
        "epochs": 2,
        "learning_rate": 1e-2
    }
    loss = nn.BCEWithLogitsLoss() # NOTE: model outputs probability, not binary classification
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)

    # ----------------------------------------------------------------------- #
    # Load Client-Specific Data for Federated Training
    # ----------------------------------------------------------------------- #
    clinical_files = [
        os.path.join(dataset_path, f"{id}/{id}_CD.json") for id in IDS
    ]  # Add all clinical file paths here
    clinical_ds = ClinicalDataset(clinical_files)

    rna_files = [
        os.path.join(dataset_path, f"{id}/{id}_RNA.json") for id in IDS
    ]  # Add all RNA file paths here
    rna_ds = RNADataset(rna_files)

    CDloader = torch.utils.data.DataLoader(
        clinical_ds, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=clinical_collate_fn
    )

    RNAloader = torch.utils.data.DataLoader(
        rna_ds, 
        batch_size=config['batch_size'], 
        shuffle=True
    )

    n_batches = len(CDloader)
    assert n_batches == len(RNAloader), f"Batches for CD ({n_batches}) not same as for RNA ({len(RNAloader)})"


    print(f"San Diego Hospital corresponds to {client_name}")

    # ----------------------------------------------------------------------- #
    # NVFlare Training Loop
    # ----------------------------------------------------------------------- #
    steps = config['epochs'] * n_batches
    for epoch in range(config['epochs']):
        for i, ((clinical_data, progression_labels), rnaseq_data) in enumerate(zip(CDloader, RNAloader)):
            clinical_data = clinical_data.to(device)
            progression_labels = progression_labels.to(device)
            rnaseq_data = rnaseq_data.to(device)

            optimizer.zero_grad()

            predictions = model(clinical_data, rnaseq_data)
            cost = loss(predictions, progression_labels)
            cost.backward()
            optimizer.step()

            print(f"[{epoch + 1}, {i + 1:5d}] loss: {cost.item()}")
            # Optional: Log metrics
            global_step = current_round * steps + epoch * len(CDloader) + i 
            summary_writer.add_scalar(tag="loss", scalar=cost.item(), global_step=global_step)

            print(f"site={client_name}, Epoch: {epoch}/{config['epochs']}, Iteration: {i}, Loss: {cost.item()}")

    print(f"Finished Training for San Diego Hospital, site_name: {client_name}")

    PATH = "./san-diego.pth"
    torch.save(model.state_dict(), PATH)
    return model.cpu().state_dict(), steps