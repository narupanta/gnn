# load model and rollout
import torch
import yaml
import os
from datetime import datetime
from core.meshgraphnet import EncodeProcessDecode
from core.datasetclass import HydrogelDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  
import numpy as np
from core.rollout import rollout

if __name__ == "__main__":
    # find config.yml in model directory
    load_model_dir = "./trained_models/20250919T133846"
    save_rollout_dir = "./rollouts"
    config_path = os.path.join(load_model_dir, 'config.yml')
    if not os.path.exists(config_path):
        print(f"Config file not found in {load_model_dir}")
        exit(1)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        # override model and training parameters if specified in config file
        node_in_dim = config["model"]["node_in_dim"]
        edge_in_dim = config["model"]["edge_in_dim"]
        hidden_size = config["model"]["hidden_size"]
        process_steps = config["model"]["process_steps"]
        node_out_dim = config["model"]["node_out_dim"]
        attention = config["model"]["attention"]
        learning_rate = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"].get("weight_decay", 1e-5))
        num_epochs = config["training"]["num_epochs"]
        percentage_train = config["training"].get("percentage_train", 0.8)
        save_model_dir = config["paths"].get("save_model_dir", './trained_models')
        data_dir = config["paths"]["data_dir"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncodeProcessDecode(node_in_dim=node_in_dim,
                                edge_in_dim=edge_in_dim,
                                mat_param_dim=mat_param_dim,
                                hidden_size=hidden_size,
                                process_steps=process_steps,
                                node_out_dim=node_out_dim,
                                attention = attention,
                                device = device).to(device)
    model_path = os.path.join(load_model_dir, 'best_model')
    model.load_model(model_path)
    model.eval()
    
    dataset = HydrogelDataset(data_dir = data_dir, noise_level=0)
    
    # loop through all samples in the dataset
    for idx in range(len(dataset)):
        sample_name = dataset.get_name(idx)
        print(f"Running rollout for sample {sample_name} ({idx+1}/{len(dataset)})")
        data = dataset[idx].to(device)
        # input trajectory into rollout prediction
        trajectory_rollout = rollout(model, data)
        # save rollout predictions and error
        os.makedirs(os.path.join(save_rollout_dir, sample_name), exist_ok=True)
        np.savez_compressed(os.path.join(save_rollout_dir, sample_name, 'rollout.npz'),
                            time = trajectory_rollout["time"].detach().cpu().numpy(),
                            pred=trajectory_rollout["pred"].detach().cpu().numpy(),
                            gt=trajectory_rollout["gt"].detach().cpu().numpy(),
                            swell_phi = trajectory_rollout["swell_phi"].detach().cpu().numpy(),
                            swell_phi_rate = trajectory_rollout["swell_phi_rate"].detach().cpu().numpy(),
                            node_type = trajectory_rollout["node_type"].detach().cpu().numpy(),
                            cells = trajectory_rollout["cells"].detach().cpu().numpy(),
                            mesh_pos = trajectory_rollout["mesh_pos"].detach().cpu().numpy())
        print(f"Rollout predictions and error saved in {save_rollout_dir}")
        

    