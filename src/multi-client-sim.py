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

from server.model import FusionNet
import clients.memphis.client as MEM
import clients.san_diego.client as SAN

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter
from clients.evaluate import evaluate, load_eval_data

DATASET_PATH = "/Users/tyleryang/Developer/CMU-NVIDIA-Hackathon/rna-cd-data/"

def main():
    # ----------------------------------------------------------------------- #
    # Setup Shared Things across All Clients
    # ----------------------------------------------------------------------- #
    model = FusionNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    testCDloader, testRNAloader = load_eval_data(DATASET_PATH)

    # ----------------------------------------------------------------------- #
    # FLARE training loop
    # ----------------------------------------------------------------------- #
    # (3) initializes NVFlare client API
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    
    summary_writer = SummaryWriter() # (optional) metrics tracking

    if client_name == "site-1":
        print(f"--- {client_name}: Memphis ---")
        update_model = MEM.update_model

    elif client_name == "site-2":
        print(f"--- {client_name}: San Diego ---")
        update_model = SAN.update_model

    else:
        raise ValueError(f"Invalid client name: {client_name}")

    while flare.is_running():
        # (4) receives FLModel from NVFlare
        global_model = flare.receive()
        # evaluate on received model
        model.load_state_dict(global_model.params) # type: ignore
        model.to(device)

        accuracy = evaluate(model, testCDloader, testRNAloader, device)
        new_params, steps = update_model(
            client_name=client_name, 
            dataset_path=DATASET_PATH,
            model=model, 
            device=device, 
            summary_writer=summary_writer,  
            current_round=global_model.current_round # type: ignore
        )

        # (7) construct trained FL model
        output_model = flare.FLModel(
            params=new_params,
            metrics={"accuracy": accuracy},
            meta={"NUM_STEPS_CURRENT_ROUND": steps},
        )
        print(f"site: {client_name}, sending model to server.")
        # (8) send model back to NVFlare
        flare.send(output_model)


if __name__ == "__main__":
    main()
