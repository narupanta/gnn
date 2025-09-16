# load model and rollout
import torch
import yaml
import os
from datetime import datetime
from core.model import EncodeProcessDecode
from core.datasetclass import HydrogelDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  
import numpy as np

if __name__ == "__main__":
    # find config.yml in model directory
    load_model_dir = "./trained_models/20250916T191035"
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

        learning_rate = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"].get("weight_decay", 1e-5))
        num_epochs = config["training"]["num_epochs"]
        percentage_train = config["training"].get("percentage_train", 0.8)
        save_model_dir = config["paths"].get("save_model_dir", './trained_models')
        data_dir = config["paths"]["data_dir"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncodeProcessDecode(node_in_dim=node_in_dim,
                                edge_in_dim=edge_in_dim,
                                hidden_size=hidden_size,
                                process_steps=process_steps,
                                node_out_dim=node_out_dim).to(device)
    model_path = os.path.join(load_model_dir, 'best_model', 'model_params.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    dataset = HydrogelDataset(data_dir = data_dir, noise_level=0)
    

    # loop through all samples in the dataset
    for idx in range(len(dataset)):
        sample_name = dataset.get_name(idx)
        print(f"Running rollout for sample {sample_name} ({idx+1}/{len(dataset)})")
        data = dataset[idx]
        # split trajectory into frames
        graphs = []
        for t in range(data.x.shape[0]):
            graph = Data(x=data.x[t], y=data.y[t], edge_index=data.edge_index, edge_attr=data.edge_attr[t], pos=data.pos, time=data.time[t], swelling_phi=data.swelling_phi[t], cells=data.cells)
            graphs.append(graph)
        # run rollout
        rollout_preds = []
        curr_graph = graphs[0].to(device)
        rollout_preds.append(curr_graph.y.cpu().numpy())
        correction_every = None # correct to ground truth every 10 steps to avoid drift 
        with torch.no_grad():
            for t in range(1, len(graphs)):
                # if correction_every :
                #     if t % correction_every == 0:
                #         print(f" Rollout step {t}/{len(graphs)}")
                #         # correct curr_graph to ground truth to avoid drift
                #         curr_graph = graphs[t-1].to(device)
                # predict next state
                pred_delta = model(curr_graph)
                curr_graph.time = graphs[t].time
                curr_graph.swelling_phi = graphs[t].swelling_phi
                pred_next = model.predict(curr_graph)
                curr_graph.x = torch.cat([pred_next, curr_graph.x[:, 3:]], dim=-1) # update current state with predicted next state, keep node_type same
                rollout_preds.append(pred_next.cpu().numpy())

        rollout_preds = np.stack(rollout_preds, axis=0) # T, N, 3
        #compute error with respect to ground truth
        gt = data.y.numpy() # T, N, 3
        error = np.abs(rollout_preds - gt) # T, N, 3
        mae_ux = np.mean(error[:, :, 0])
        mae_uy = np.mean(error[:, :, 1])
        mae_phi = np.mean(error[:, :, 2])
        print(f"MAE Ux: {mae_ux:.6f}, Uy: {mae_uy:.6f}, Phi: {mae_phi:.6f}")
        # save rollout predictions and error
        save_rollout_dir = os.path.join(save_rollout_dir, sample_name)
        os.makedirs(save_rollout_dir, exist_ok=True)
        np.savez_compressed(os.path.join(save_rollout_dir, 'rollout_preds.npz'),
                            rollout_preds=rollout_preds,
                            gt=gt,
                            error=error,
                            mae_ux=mae_ux,
                            mae_uy=mae_uy,
                            mae_phi=mae_phi,
                            cells = data.cells.numpy(),
                            pos = data.pos.numpy())
        print(f"Rollout predictions and error saved in {save_rollout_dir}")
        

    