# training routine for the GNN model
import torch
import torch.optim as optim

from torch_geometric.data import Data, DataLoader
# from torch.loader import DataLoader
# from torch.utils.data import DataLoader
from core.datasetclass import HydrogelDataset
from core.meshgraphnet import EncodeProcessDecodeMultiScale
from core.rollout import rollout
from tqdm import tqdm
import os
import yaml
from datetime import datetime
import logging


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
    config_path = "train_multiscale_config.yml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # override model and training parameters if specified in config file
        node_in_dim = config["model"]["node_in_dim"]
        edge_in_dim = config["model"]["edge_in_dim"]
        hidden_size = config["model"]["hidden_size"]
        process_steps = config["model"]["process_steps"]
        coarse_process_steps = config["model"]["coarse_process_steps"]
        node_out_dim = config["model"]["node_out_dim"]
        attention = config["model"]["attention"]
        sample_ratio = config["model"]["sample_ratio"]
        time_dim = config["model"]["time_dim"]

        learning_rate = float(config["training"]["learning_rate"])
        weight_decay = float(config["training"].get("weight_decay", 1e-5))
        num_epochs = config["training"]["num_epochs"]
        start_noise = config["training"]["start_noise_level"]
        end_noise = config["training"]["end_noise_level"]
        with_mat_params= config["training"].get("with_mat_params", False)
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

    dataset = HydrogelDataset(
        data_dir=data_dir,
        noise_level=noise_schedule(0, num_epochs, initial_noise=start_noise, final_noise=end_noise),
        add_targets=True,
        split_to_frames=True,
        time_dim = time_dim
    )
    
    rollout_dataset = HydrogelDataset(data_dir=data_dir, noise_level=0.0, time_dim = time_dim)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncodeProcessDecodeMultiScale(
        node_in_dim=node_in_dim,
        edge_in_dim=edge_in_dim,
        hidden_size=hidden_size,
        process_steps=process_steps,
        coarse_process_steps=coarse_process_steps,
        node_out_dim=node_out_dim,
        attention=attention,
        sample_ratio=sample_ratio,
        with_mat_params=with_mat_params,
        time_dim=time_dim,
        device=device
    ).to(device)
    # --- Example usage ---
    # Suppose `model` is your PyTorch model
    log_model_parameters(model)
    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.get("weight_decay", 1e-5)
    )

    # Scheduler (optional: cosine annealing works well for GNNs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6
    )

    best_val_loss = float('inf')
    for train_epoch in range(num_epochs):
        model.train()
        total_loss, total_loss_ux, total_loss_uy, total_loss_phi = 0.0, 0.0, 0.0, 0.0
        dataset.noise_level = noise_schedule(train_epoch, num_epochs, initial_noise=start_noise, final_noise=end_noise)
        for traj_idx, data in enumerate(dataset):
            traj_total_loss, traj_disp_x_loss, traj_disp_y_loss, traj_pvf_loss = 0, 0, 0, 0
            train_loader = DataLoader(data, batch_size=1, shuffle=True)
            loop = tqdm(train_loader, leave=False)
            for batch in loop:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss, loss_ux, loss_uy, loss_phi = model.loss(batch)
                loss.backward()
                optimizer.step()
                traj_total_loss += loss.item()
                traj_disp_x_loss += loss_ux.item()
                traj_disp_y_loss += loss_uy.item()
                traj_pvf_loss += loss_phi.item()
                loop.set_description(f"Epoch {train_epoch + 1}, Traj {traj_idx + 1}")
                loop.set_postfix({
                        "Loss": f"{loss.item():.4f}",
                        "Disp X": f"{loss_ux.item():.4f}",
                        "Disp Y": f"{loss_uy.item():.4f}",
                        "PVF": f"{loss_phi.item():.4f}"
                    })
            total_loss += traj_total_loss
            total_loss_ux += traj_disp_x_loss
            total_loss_uy += traj_disp_y_loss
            total_loss_phi += traj_pvf_loss
            logger.info(
                f"Epoch {train_epoch + 1}, Trajectory {traj_idx + 1}: Train Loss: {traj_total_loss:.4f}, Ux Loss: {traj_disp_x_loss:.4f}, Uy Loss: {traj_disp_y_loss:.4f}, Phi Loss: {traj_pvf_loss:.4f}"
            )


        avg_loss = total_loss / len(dataset)
        avg_loss_ux = total_loss_ux / len(dataset)
        avg_loss_uy = total_loss_uy / len(dataset)
        avg_loss_phi = total_loss_phi / len(dataset)

        # Log training loss
        logger.info(
            f"Epoch {train_epoch + 1}, "
            f"Train Loss: {avg_loss:.6f}, "
            f"Ux Loss: {avg_loss_ux:.6f}, "
            f"Uy Loss: {avg_loss_uy:.6f}, "
            f"Phi Loss: {avg_loss_phi:.6f}"
        )

        scheduler.step()
        model.eval()
        total_rollout_loss = 0.0
        total_rmse_x, total_rmse_y, total_rmse_phi = 0.0, 0.0, 0.0
        num_rollouts = 0
        with torch.no_grad():
            for trajectory in rollout_dataset:
                rollout_result = rollout(model, trajectory)
                rmse_x, rmse_y, rmse_phi = (
                    rollout_result["rmse_x"],
                    rollout_result["rmse_y"],
                    rollout_result["rmse_phi"],
                )
                rollout_loss = rmse_x + rmse_y + rmse_phi
                total_rollout_loss += rollout_loss
                total_rmse_x += rmse_x
                total_rmse_y += rmse_y
                total_rmse_phi += rmse_phi
                num_rollouts += 1
                logger.info(
                f"Rollout Nr.{num_rollouts} Loss: {rollout_loss:.6f}, "
                f"RMSE_x: {rmse_x:.6f}, "
                f"RMSE_y: {rmse_y:.6f}, "
                f"RMSE_phi: {rmse_phi:.6f}"
            )
        avg_rollout_loss = total_rollout_loss / max(1, num_rollouts)
        avg_rmse_x = total_rmse_x / max(1, num_rollouts)
        avg_rmse_y = total_rmse_y / max(1, num_rollouts)
        avg_rmse_phi = total_rmse_phi / max(1, num_rollouts)

        logger.info(
            f"Rollout Loss: {avg_rollout_loss:.6f}, "
            f"RMSE_x: {avg_rmse_x:.6f}, "
            f"RMSE_y: {avg_rmse_y:.6f}, "
            f"RMSE_phi: {avg_rmse_phi:.6f}"
        )

        # Save best model
        if avg_rollout_loss < best_val_loss:
            best_val_loss = avg_rollout_loss
            best_model_dir = os.path.join(model_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            model.save_model(best_model_dir)
            logger.info("best rollout model saved")

        # Save checkpoint every 20 epochs
        if (train_epoch + 1) % 20 == 0:
            epoch_model_dir = os.path.join(model_dir, f"epoch_{train_epoch+1}")
            os.makedirs(epoch_model_dir, exist_ok=True)
            model.save_model(epoch_model_dir)
            logger.info("epoch model saved")
