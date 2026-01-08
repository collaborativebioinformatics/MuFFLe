# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
client side training scripts
"""

import torch
from server.model import SimpleNetwork
from torch import nn
from torch.optim import SGD
from clients.datasets import ClinicalDataset, RNADataset
import os

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

DATASET_PATH = "/Users/tyleryang/Developer/CMU-NVIDIA-Hackathon/rna-cd-data/"
IDS = ['3A_001', '3A_002', '3A_003', '3A_004', '3A_005', '3A_006', '3A_007', '3A_008', '3A_009', '3A_010', '3A_011', '3A_012', '3A_013', '3A_014', '3A_015', '3A_016', '3A_017', '3A_018', '3A_019', '3A_020', '3A_021', '3A_022', '3A_023', '3A_024', '3A_025', '3A_026', '3A_027', '3A_028', '3A_029', '3A_030', '3A_031', '3A_033', '3A_034', '3A_035', '3A_036', '3A_037', '3A_038', '3A_039', '3A_040', '3A_041', '3A_042', '3A_043', '3A_044', '3A_045', '3A_046', '3A_047', '3A_049', '3A_050', '3A_052', '3A_053', '3A_055', '3A_056', '3A_057', '3A_058', '3A_059', '3A_060', '3A_061', '3A_062', '3A_063', '3A_064', '3A_066', '3A_067', '3A_068', '3A_070', '3A_071', '3A_072', '3A_073', '3A_074', '3A_075', '3A_076', '3A_077', '3A_087', '3A_088', '3A_089', '3A_091', '3A_092', '3A_093', '3A_094', '3A_095', '3A_097', '3A_098', '3A_100', '3A_105', '3A_108', '3A_110', '3A_111', '3A_113', '3A_114', '3A_115', '3A_116', '3A_123', '3A_124', '3A_125', '3A_126', '3A_127', '3A_129', '3A_130', '3A_134', '3A_135', '3A_136', '3A_137', '3A_138', '3A_139', '3A_140', '3A_141', '3A_142', '3A_143', '3A_144', '3A_145', '3A_146', '3A_147', '3A_148', '3A_149', '3A_153', '3A_154', '3A_157', '3A_158', '3A_160', '3A_162', '3A_163', '3A_165', '3A_168', '3A_169', '3A_186', '3A_190', '3A_191', '3B_208', '3B_217', '3B_225', '3B_227', '3B_229', '3B_230', '3B_250', '3B_262', '3B_266', '3B_267', '3B_277', '3B_281', '3B_288', '3B_292', '3B_302', '3B_303', '3B_304', '3B_309', '3B_310', '3B_319', '3B_321', '3B_322', '3B_328', '3B_337', '3B_338', '3B_342', '3B_351', '3B_354', '3B_357', '3B_361', '3B_362', '3B_365', '3B_367', '3B_370', '3B_385', '3B_389', '3B_390', '3B_397', '3B_399', '3B_408', '3B_410', '3B_411', '3B_413', '3B_415', '3B_417', '3B_418', '3B_426', '3B_428', '3B_429', '3B_431']
BATCH_SIZE = 32
EPOCHS = 2


def main():
    # ----------------------------------------------------------------------- #
    # Setup + Model Architecture Init
    # ----------------------------------------------------------------------- #
    lr = 0.01
    model = SimpleNetwork(rna_dim=19359, clinical_dim=13)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

    # ----------------------------------------------------------------------- #
    # Load Datasets
    # ----------------------------------------------------------------------- #
    clinical_files = [
        os.path.join(DATASET_PATH, f"{id}/{id}_CD.json") for id in IDS
    ]  # Add all clinical file paths here
    clinical_ds = ClinicalDataset(clinical_files)

    rna_files = [
        os.path.join(DATASET_PATH, f"{id}/{id}_RNA.json") for id in IDS
    ]  # Add all RNA file paths here
    rna_ds = RNADataset(rna_files)

    # ----------------------------------------------------------------------- #
    # FLARE training loop
    # ----------------------------------------------------------------------- #
    # (3) initializes NVFlare client API
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]

    if client_name == "site-1":
        print(f"--- {client_name}: Training CLINICAL modality ---")
        train_loader = torch.utils.data.DataLoader(
            clinical_ds, batch_size=BATCH_SIZE, shuffle=True
        )

        def predict_risk(model, data):
            return model(x_rna=None, x_clinical=data) 

    elif client_name == "site-2":
        print(f"--- {client_name}: Training RNA modality ---")
        train_loader = torch.utils.data.DataLoader(
            rna_ds, batch_size=BATCH_SIZE, shuffle=True
        )
        def predict_risk(model, data):
            return model(x_rna=data, x_clinical=None)

    else:
        raise ValueError(f"Invalid client name: {client_name}")

    # (optional) metrics tracking
    summary_writer = SummaryWriter()

    while flare.is_running():
        # (4) receives FLModel from NVFlare
        input_model = flare.receive()
        print(f"site = {client_name}, current_round={input_model.current_round}")

        # (5) loads model from NVFlare
        model.load_state_dict(input_model.params)
        model.to(device)
        # (6) evaluate on received model for model selection

        steps = EPOCHS * len(train_loader)
        for epoch in range(EPOCHS):
            running_loss = 0.0
            for i, batch in enumerate(train_loader):
                data = batch.to(device)
                # print(data) # DEBUG STATEMENT
                optimizer.zero_grad()

                predictions = predict_risk(model, data)

                labels = torch.zeros_like(predictions) # dummy labels

                cost = loss(predictions, labels)
                cost.backward()
                optimizer.step()

                running_loss += cost.item()
                
                avg_loss = running_loss / (i + 1)
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}")

                # Optional: Log metrics
                global_step = (
                    input_model.current_round * steps
                    + epoch * len(train_loader)
                    + i
                )
                summary_writer.add_scalar(
                    tag="loss", scalar=avg_loss, global_step=global_step
                )

                print(
                    f"site={client_name}, Epoch: {epoch}/{EPOCHS}, Iteration: {i}, Loss: {running_loss}"
                )
                running_loss = 0.0

        print(f"Finished Training for {client_name}")

        PATH = f"./{client_name}.pth"
        torch.save(model.state_dict(), PATH)

        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            # metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
