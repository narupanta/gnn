import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from core.datasetclass import HydrogelDataset
from core.temporal_model import TemporalWrapper
from core.meshgraphnet import EncodeProcessDecode
from core.rollout import rollout_temporal
from tqdm import tqdm
import os
import yaml
from datetime import datetime
import logging
def log_gpu_memory(logger=None, prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e6  # MB
        max_allocated = torch.cuda.max_memory_allocated() / 1e6
        reserved = torch.cuda.memory_reserved() / 1e6
        if logger:
            logger.info(f"{prefix} GPU Memory: allocated={allocated:.2f} MB, max_allocated={max_allocated:.2f} MB, reserved={reserved:.2f} MB")
        else:
            print(f"{prefix} GPU Memory: allocated={allocated:.2f} MB, max_allocated={max_allocated:.2f} MB, reserved={reserved:.2f} MB")
def noise_schedule(epoch, total_epochs, initial_noise=0.1, final_noise=0.01):
    """Linear noise schedule from initial_noise to final_noise over total_epochs."""
    if epoch >= total_epochs:
        return final_noise
    return initial_noise + (final_noise - initial_noise) * (epoch / total_epochs)

def log_model_parameters(model):
    total_params = 0
    total_trainable = 0

    logger.info("===== Model Parameters =====")
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            total_trainable += num_params
        logger.info(f"{name}: {param.size()} | params={num_params} | requires_grad={param.requires_grad}")

    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Trainable parameters: {total_trainable}")
    logger.info("============================")

if __name__ == "__main__":
    # read model and training parameters from config yml file if exists
    config_path = "train_temporal_config.yml"
    if os.path.exists(config_path):
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
        start_noise = config["training"].get("start_noise", 0.1)
        end_noise = config["training"].get("end_noise", 0.01)
        save_model_dir = config["paths"].get("save_model_dir", './trained_models')
        data_dir = config["paths"]["data_dir"]

        # save the config file in model_dir for future reference
        now = datetime.now()
        dt_string = now.strftime("%Y%m%dT%H%M%S")
        model_dir = os.path.join(save_model_dir, dt_string)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Model will be saved in {model_dir}")
        with open(os.path.join(model_dir, 'config.yml'), 'w') as f:
            yaml.dump(config, f)

    # --- Setup logging ---
    log_file = os.path.join(model_dir, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("train")
    # --- Dataset ---
    dataset = HydrogelDataset(
        data_dir=data_dir,
        noise_level=noise_schedule(0, num_epochs, initial_noise=start_noise, final_noise=end_noise)
    )

    # For sequences, we assume HydrogelDataset can return full trajectories
    # If not, we need to create sequences manually
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    rollout_dataset = HydrogelDataset(data_dir=data_dir, noise_level=0.0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Model ---
    base_mgn = EncodeProcessDecode(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_size=hidden_size,
        process_steps=process_steps,
        node_out_dim=node_out_dim,
        attention=attention,
        device=device
    ).to(device)

    model = TemporalWrapper(
        base_model=base_mgn,
        latent_dim=hidden_size,
        rnn_hidden=128,        # you can adjust
        rnn_type='RNN'         # 'RNN', 'GRU', or 'LSTM'
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, leave=False)
        for trajectory in loop:   # trajectory is a list of graphs
            optimizer.zero_grad()
            # Move graphs to device
            graph_seq = []
            for t in range(trajectory.world_pos.shape[0]) :
                graph = Data(world_pos=trajectory.world_pos[t], 
                            phi = trajectory.phi[t],
                            node_type = trajectory.node_type, target=trajectory.target[t], edge_index=trajectory.edge_index, mesh_pos=trajectory.mesh_pos, time=trajectory.time[t], 
                            swelling_phi=trajectory.swelling_phi[t], 
                            swelling_phi_rate = trajectory.swelling_phi_rate[t], 
                            cells=trajectory.cells, mat_param=trajectory.mat_param)
                graph_seq.append(graph.to(device))
            # Forward pass: predict Δpos for all timesteps
            pred_seq = model(graph_seq)  # [N, T, node_out_dim]

            # Compute per-timestep loss
            loss = 0.0
            for t, graph in enumerate(graph_seq):
                target = graph.target
                curr = torch.cat([graph.world_pos, graph.phi], dim=-1)
                target_delta = target - curr
                target_delta_normalized = model.base_model.output_normalizer(target_delta)
                pred_delta = pred_seq[:, t, :]
                error = (pred_delta - target_delta_normalized) ** 2

                # Mask DBC nodes
                ux_dbc_nodes = graph.node_type[:, 1] == 1
                uy_dbc_nodes = graph.node_type[:, 2] == 1
                phi_dbc_nodes = graph.node_type[:, 3] == 1
                loss += (
                    torch.mean(error[:, 0].masked_select(~ux_dbc_nodes)) +
                    torch.mean(error[:, 1].masked_select(~uy_dbc_nodes)) +
                    torch.mean(error[:, 2].masked_select(~phi_dbc_nodes))
                )

            loss /= len(graph_seq)  # average over timesteps
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch + 1}")
            loop.set_postfix({"Loss": f"{loss.item():.6f}"})
            log_gpu_memory(logger, prefix=f"Epoch {epoch+1} Batch Memory")
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}, Train Loss: {avg_loss:.6f}")

        scheduler.step()

        # --- Rollout evaluation ---
        model.eval()
        total_rollout_loss = 0.0
        num_rollouts = 0
        with torch.no_grad():
            for trajectory in rollout_dataset:
                # Autoregressive rollout with temporal wrapper
                rollout_result = rollout_temporal(model, trajectory)
                rollout_loss = rollout_result["rmse_x"] + rollout_result["rmse_y"] + rollout_result["rmse_phi"]
                total_rollout_loss += rollout_loss
                num_rollouts += 1

        avg_rollout_loss = total_rollout_loss / max(1, num_rollouts)
        logging.info(f"Epoch {epoch+1}, Avg Rollout Loss: {avg_rollout_loss:.6f}")

        # Save best model
        if avg_rollout_loss < best_val_loss:
            best_val_loss = avg_rollout_loss
            best_model_dir = os.path.join(model_dir, "best_temporal_model")
            os.makedirs(best_model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(best_model_dir, "model_weights.pth"))
            logging.info("Saved best temporal model")
