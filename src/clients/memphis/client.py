"""
Adapted from NVFlare "Hello PyTorch" (hello-pt) example code.

Simulates the Memphis Hospital as outlined in the Methods.
"""
import os 

import torch 
from torch import nn 

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

from server.model import FusionNet
from clients.datasets import ClinicalDataset
from clients.evaluate import evaluate, load_eval_data

def main():
    DATASET_PATH="/Users/tyleryang/Developer/CMU-NVIDIA-Hackathon/rna-cd-data/"
    # ----------------------------------------------------------------------- #
    # Client-Specific Constants
    # ----------------------------------------------------------------------- #
    IDS = ['3B_208', '3B_217', '3B_225', '3B_227', '3B_229', '3B_230', '3B_250', '3B_262', '3B_266', '3B_267', '3B_277', '3B_281', '3B_288', '3B_292', '3B_302', '3B_303', '3B_304', '3B_309', '3B_310', '3B_319', '3B_321', '3B_322', '3B_328', '3B_337', '3B_338', '3B_342', '3B_351', '3B_354', '3B_357', '3B_361', '3B_362', '3B_365', '3B_367', '3B_370', '3B_385', '3B_389', '3B_390', '3B_397', '3B_399', '3B_408', '3B_410', '3B_411', '3B_413', '3B_415', '3B_417', '3B_418', '3B_426', '3B_428', '3B_429', '3B_431']
    assert len(IDS) == 50, f"expected 50 IDS in Memphis (CHIMERA Task 3 Cohort B), got {len(IDS)}"

    # ----------------------------------------------------------------------- #
    # Initialize Model
    # ----------------------------------------------------------------------- #

    # Shared across all clients
    model = FusionNet()

    # Free for Client to modify
    config = {
        "batch_size": 16,
        "epochs": 4,
        "learning_rate": 1e-3
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.BCEWithLogitsLoss() # NOTE: model outputs probability, not binary classification
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # ----------------------------------------------------------------------- #
    # Load Client-Specific Data for Federated Training
    # ----------------------------------------------------------------------- #
    clinical_files = [
        os.path.join(DATASET_PATH, f"{id}/{id}_CD.json") for id in IDS
    ]  # Add all clinical file paths here
    clinical_ds = ClinicalDataset(clinical_files)

    CDloader = torch.utils.data.DataLoader(
        clinical_ds, 
        batch_size=config['batch_size'], 
        shuffle=True
    )

    # ----------------------------------------------------------------------- #
    # Load Evaluation Data (only b/c we are simulating FL)
    # ----------------------------------------------------------------------- #
    testCDloader, testRNAloader = load_eval_data(DATASET_PATH)

    # ----------------------------------------------------------------------- #
    # NVFlare Initialization
    # ----------------------------------------------------------------------- #
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    summary_writer = SummaryWriter()
    print(f"Memphis Hospital corresponds to site_name: {client_name}")

    # ----------------------------------------------------------------------- #
    # NVFlare Training Loop
    # ----------------------------------------------------------------------- #
    while flare.is_running():
        global_model = flare.receive()
        print(f"Memphis Hospital site_name: {client_name}")
        print(f"current_round={global_model.current_round}") # type: ignore

        model.load_state_dict(global_model.params) # type: ignore
        model.to(device)

        # evaluate on received model
        accuracy = evaluate(model, testCDloader, testRNAloader, device)

        steps = config['epochs'] * len(CDloader)
        for epoch in range(config['epochs']):
            for i, batch in enumerate(CDloader):
                clinical_data, labels = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()

                predictions = model(clinical_data, None)
                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                print(f"[{epoch + 1}, {i + 1:5d}] loss: {cost.item()}")
                # Optional: Log metrics
                global_step = global_model.current_round * steps + epoch * len(CDloader) + i # type: ignore
                summary_writer.add_scalar(tag="loss", scalar=cost.item(), global_step=global_step)

                print(f"site={client_name}, Epoch: {epoch}/{config['epochs']}, Iteration: {i}, Loss: {cost.item()}")

        print(f"Finished Training for Memphis Hospital, site_name: {client_name}")

        PATH = "./memphis.pth"
        torch.save(model.state_dict(), PATH)

        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        # (8) send model back to NVFlare
        flare.send(output_model)

if __name__ == "__main__":
    main()