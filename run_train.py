# training routine for the GNN model
import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from core.datasetclass import HydrogelDataset
from core.model import EncodeProcessDecode   
def noise_schedule(epoch, total_epochs, initial_noise=0.1, final_noise=0.01):
    """Linear noise schedule from initial_noise to final_noise over total_epochs."""
    if epoch >= total_epochs:
        return final_noise
    return initial_noise + (final_noise - initial_noise) * (epoch / total_epochs)

if __name__ == "__main__":
    import os
    from datetime import datetime
    import argparse, yaml

    # read model and training parameters from config yml file if exists
    config_path = "train_config.yml"
    if os.path.exists(config_path):
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


    dataset = HydrogelDataset(data_dir = data_dir, noise_level=noise_schedule(0, num_epochs, initial_noise=start_noise, final_noise=end_noise))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # split trajectory into frames
    all_graphs = []
    for data in dataloader:
        for t in range(data.x.shape[0]):
            graph = Data(x=data.x[t], y=data.y[t], edge_index=data.edge_index, edge_attr=data.edge_attr[t], pos=data.pos, time=data.time[t], swelling_phi=data.swelling_phi[t], cells=data.cells)
            all_graphs.append(graph)
    # split into train and val
    percentage_train = 0.8
    train_loader = DataLoader(all_graphs[:int(percentage_train*len(all_graphs))], batch_size=1, shuffle=True)
    val_loader = DataLoader(all_graphs[int(percentage_train*len(all_graphs)):], batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncodeProcessDecode(node_in_dim=node_in_dim,
                                edge_in_dim=edge_in_dim,
                                hidden_size=hidden_size,
                                process_steps=process_steps,
                                node_out_dim=node_out_dim).to(device)
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
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss, loss_ux, loss_uy, loss_phi = model.loss(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss_ux += loss_ux.item()
            total_loss_uy += loss_uy.item()
            total_loss_phi += loss_phi.item()
        avg_loss = total_loss / len(train_loader)
        avg_loss_ux = total_loss_ux / len(train_loader)
        avg_loss_uy = total_loss_uy / len(train_loader)
        avg_loss_phi = total_loss_phi / len(train_loader)

        #print train loss and component loss
        print(f"Epoch {train_epoch}, Train Loss: {avg_loss:.6f}, Ux Loss: {avg_loss_ux:.6f}, Uy Loss: {avg_loss_uy:.6f}, Phi Loss: {avg_loss_phi:.6f}")
        scheduler.step()
        model.eval()
        total_val_loss, total_val_loss_ux, total_val_loss_uy, total_val_loss_phi = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss, loss_ux, loss_uy, loss_phi = model.loss(batch)
                total_val_loss += loss.item()
                total_val_loss_ux += loss_ux.item()
                total_val_loss_uy += loss_uy.item()
                total_val_loss_phi += loss_phi.item()
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_loss_ux = total_val_loss_ux / len(val_loader)
        avg_val_loss_uy = total_val_loss_uy / len(val_loader)
        avg_val_loss_phi = total_val_loss_phi / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # save model in subdirectory of model_dir as best_model
            best_model_dir = os.path.join(model_dir, "best_model")
            if not os.path.exists(best_model_dir):
                os.makedirs(best_model_dir)
            torch.save(model.state_dict(), os.path.join(best_model_dir, "model_params.pth"))
            torch.save(optimizer.state_dict(), os.path.join(best_model_dir, "optimizer_params.pth"))
            torch.save(scheduler.state_dict(), os.path.join(best_model_dir, "scheduler_params.pth"))
        #save model every 20 epochs in model_dir
        if train_epoch % 20 == 0:
            # save model in subdirectory of model_dir as epoch_{train_epoch}
            epoch_model_dir = os.path.join(model_dir, "epoch_model")
            if not os.path.exists(epoch_model_dir):
                os.makedirs(epoch_model_dir)
            torch.save(model.state_dict(), os.path.join(epoch_model_dir, f"model_epoch_{train_epoch}.pth"))
            torch.save(optimizer.state_dict(), os.path.join(epoch_model_dir, f"optimizer_epoch_{train_epoch}.pth"))
            torch.save(scheduler.state_dict(), os.path.join(epoch_model_dir, f"scheduler_epoch_{train_epoch}.pth"))
        
        #print val loss and component loss
        print(f"Epoch {train_epoch}, Val Loss: {avg_val_loss:.6f}, Ux Loss: {avg_val_loss_ux:.6f}, Uy Loss: {avg_val_loss_uy:.6f}, Phi Loss: {avg_val_loss_phi:.6f}")

        #log training and val loss to a file in model_dir
        with open(os.path.join(model_dir, "training_log.txt"), "a") as f:
            f.write(f"Epoch {train_epoch}, Train Loss: {avg_loss:.6f}, Ux Loss: {avg_loss_ux:.6f}, Uy Loss: {avg_loss_uy:.6f}, Phi Loss: {avg_loss_phi:.6f}, Val Loss: {avg_val_loss:.6f}, Ux Val Loss: {avg_val_loss_ux:.6f}, Uy Val Loss: {avg_val_loss_uy:.6f}, Phi Val Loss: {avg_val_loss_phi:.6f}\n")