# load model and rollout
import torch
import yaml
import os
from datetime import datetime
from core.meshgraphnet import EncodeProcessDecodeHistory
from core.datasetclass import HydrogelDatasetHistory
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader  
import numpy as np
from core.rollout import rollout_history
import meshio
if __name__ == "__main__":
    # find config.yml in model directory
    load_model_dir = "./trained_models/20251022T181928"
    data_dir = "./dataset/bending_signal_trainset"
    save_rollout_dir = "./rollouts/test"
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
        with_mat_params = config["training"]["with_mat_params"]
        save_model_dir = config["paths"].get("save_model_dir", './trained_models')
        if data_dir :
            data_dir = data_dir
        else :
            data_dir = config["paths"]["data_dir"]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncodeProcessDecodeHistory(node_in_dim=node_in_dim,
                                edge_in_dim=edge_in_dim,
                                hidden_size=hidden_size,
                                process_steps=process_steps,
                                node_out_dim=node_out_dim,
                                attention = attention,
                                with_mat_params = with_mat_params,
                                device = device).to(device)
    model_path = os.path.join(load_model_dir, 'best_model')
    model.load_model(model_path)
    model.eval()
    
    dataset = HydrogelDatasetHistory(data_dir = data_dir, noise_level=0.0)
    
    # loop through all samples in the dataset
    for idx in range(len(dataset)):
        sample_name = dataset.get_name(idx)
        print(sample_name)
        # if idx != 0 :
        #     continue  
        print(f"Running rollout for sample {sample_name} ({idx+1}/{len(dataset)})")
        data = dataset[idx].to(device)
        # input trajectory into rollout prediction
        trajectory_rollout = rollout_history(model, data)
        # save rollout predictions and error
        os.makedirs(os.path.join(save_rollout_dir, sample_name), exist_ok=True)
        np.savez_compressed(os.path.join(save_rollout_dir, sample_name, 'rollout.npz'),
                            time = trajectory_rollout["time"].detach().cpu().numpy(),
                            pred=trajectory_rollout["pred"].detach().cpu().numpy(),
                            gt=trajectory_rollout["gt"].detach().cpu().numpy(),
                            swell_phi = trajectory_rollout["swell_phi"].detach().cpu().numpy(),
                            swell_phi_rate = trajectory_rollout["swell_phi_rate"].detach().cpu().numpy(),
                            swell_phi_rate_prev = trajectory_rollout["swelling_phi_rate_prev"].detach().cpu().numpy(),
                            node_type = trajectory_rollout["node_type"].detach().cpu().numpy(),
                            cells = trajectory_rollout["cells"].detach().cpu().numpy(),
                            mesh_pos = trajectory_rollout["mesh_pos"].detach().cpu().numpy(),
                            rmse_x = trajectory_rollout["rmse_x"].detach().cpu().numpy(),
                            rmse_y = trajectory_rollout["rmse_y"].detach().cpu().numpy(),
                            rmse_phi = trajectory_rollout["rmse_phi"].detach().cpu().numpy(),)
        print(f"Rollout predictions and error saved in {save_rollout_dir}")
        save_paraview_dir = f"{save_rollout_dir}/{sample_name}/paraview"
        os.makedirs(save_paraview_dir, exist_ok=True)

        # Create pred and gt folders
        pred_dir = os.path.join(save_paraview_dir, "pred")
        gt_dir = os.path.join(save_paraview_dir, "gt")
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        # Load rollout data
        data = dict(time = trajectory_rollout["time"].detach().cpu().numpy(),
                            pred=trajectory_rollout["pred"].detach().cpu().numpy(),
                            gt=trajectory_rollout["gt"].detach().cpu().numpy(),
                            swell_phi = trajectory_rollout["swell_phi"].detach().cpu().numpy(),
                            swell_phi_rate = trajectory_rollout["swell_phi_rate"].detach().cpu().numpy(),
                            swell_phi_rate_prev = trajectory_rollout["swelling_phi_rate_prev"].detach().cpu().numpy(),
                            node_type = trajectory_rollout["node_type"].detach().cpu().numpy(),
                            cells = trajectory_rollout["cells"].detach().cpu().numpy(),
                            mesh_pos = trajectory_rollout["mesh_pos"].detach().cpu().numpy(),
                            rmse_x = trajectory_rollout["rmse_x"].detach().cpu().numpy(),
                            rmse_y = trajectory_rollout["rmse_y"].detach().cpu().numpy(),
                            rmse_phi = trajectory_rollout["rmse_phi"].detach().cpu().numpy(),)

        pred = data["pred"]              # shape: [T, N, 3] or [T, N, 2]
        gt = data["gt"]                  # same shape
        cells = data["cells"]            # [num_cells, nodes_per_cell]
        time = data["time"]
        print(time[1] - time[0])
        # Handle 2D -> 3D conversion
        def ensure_3d(coords):
            if coords.shape[2] == 2:
                return np.concatenate([coords, np.zeros((coords.shape[0], coords.shape[1], 1))], axis=2)
            return coords

        pred_coords = ensure_3d(pred[:, :, :2])
        gt_coords = ensure_3d(gt[:, :, :2])
        swell_phi = data["swell_phi"]
        # Determine mesh cell type
        num_nodes_per_cell = cells.shape[1]
        if num_nodes_per_cell == 3:
            cell_type = "triangle"
        elif num_nodes_per_cell == 4:
            cell_type = "tetra"
        else:
            raise ValueError(f"Unsupported cell with {num_nodes_per_cell} nodes")

        # Function to write VTU files
        def write_vtu_series(coords, scalars, load, folder, prefix):
            pvd_entries = []
            for t_idx in range(coords.shape[0]):
                mesh = meshio.Mesh(
                    points=coords[t_idx],
                    cells=[(cell_type, cells)],
                    point_data={f"{prefix}_phi": scalars[t_idx], f"swelling_phi": load[t_idx]}
                )
                filename = f"{prefix}_{t_idx:04d}.vtu"
                filepath = os.path.join(folder, filename)
                mesh.write(filepath)
                # Add entry to pvd
                pvd_entries.append(f'    <DataSet timestep="{time[t_idx]}" group="" part="0" file="{filename}"/>')
            
            # Write PVD file
            pvd_content = (
                '<?xml version="1.0"?>\n'
                '<VTKFile type="Collection" version="0.1">\n'
                '  <Collection>\n' +
                "\n".join(pvd_entries) +
                '\n  </Collection>\n'
                '</VTKFile>\n'
            )
            with open(os.path.join(folder, f"{prefix}.pvd"), "w") as f:
                f.write(pvd_content)
            print(f"PVD file saved at: {os.path.join(folder, f'{prefix}.pvd')}")

        # Write prediction series
        write_vtu_series(pred_coords, pred[:, :, 1], swell_phi, pred_dir, "pred")

        # Write ground truth series
        write_vtu_series(gt_coords, gt[:, :, 1], swell_phi, gt_dir, "gt")

        print(f"Visualization files saved in {save_paraview_dir}")
        

    