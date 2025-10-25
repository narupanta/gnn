from torch.utils.data import Dataset
import torch
import numpy as np
import os
from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
def cells_to_edge_index(cells: torch.Tensor) -> torch.Tensor:
    """
    Convert triangular cells (N,3) into undirected edge_index (2,M) for PyG.
    """
    # edges from triangles
    edges = torch.cat([
        torch.stack([cells[:, 0], cells[:, 1]], dim=0),
        torch.stack([cells[:, 1], cells[:, 2]], dim=0),
        torch.stack([cells[:, 2], cells[:, 0]], dim=0),
    ], dim=1)

    # make undirected (adds reverse edges + removes duplicates)
    edge_index = to_undirected(edges)
    return edge_index

class HydrogelDataset(Dataset):
    def __init__(self, data_dir, noise_level=None, add_targets = None, split_to_frames = None, time_dim=5):
        """
        Args:
            data_dir (str): Directory containing .npz hydrogel samples
            noise_level (float, optional): Noise factor for augmentation
            coarse_factor (int): Factor to downsample time dimension. 1 = full resolution
        """
        self.data_dir = data_dir
        self.noise_level = noise_level
        self.add_targets = add_targets
        self.split_to_frames = split_to_frames
        self.time_dim = time_dim
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]

    def __len__(self):
        return len(self.data_files)

    def get_name(self, idx):
        return self.data_files[idx].rstrip('.npz')

    def __getitem__(self, idx):
        # Load data
        sample = np.load(os.path.join(self.data_dir, self.data_files[idx]))
        mesh_pos = torch.tensor(sample['mesh_coords'], dtype=torch.float32)
        cells = torch.tensor(sample['cells'], dtype=torch.long)
        edge_index = cells_to_edge_index(cells)
        node_type = torch.tensor(sample["node_type"], dtype=torch.float32)
        u = torch.tensor(sample['u_time_series'], dtype=torch.float32)
        world_pos = mesh_pos + u
        phi = torch.tensor(sample["φ_time_series"], dtype=torch.float32).unsqueeze(-1)
        swell_phi = torch.tensor(sample["swell_time_series"], dtype=torch.float32).unsqueeze(-1)
        time = torch.tensor(sample["t"], dtype=torch.float32)
        mat_param = torch.tensor([sample['chi'].item(), sample['diffusivity'].item()], dtype=torch.float32)
        swell_nodes = node_type[:, 4] == 1

        # Create swell_phi_tensor
        swell_phi_tensor = torch.zeros((phi.shape[0], phi.shape[1]), device=phi.device)
        swell_phi_tensor[:, swell_nodes] = swell_phi.expand(phi.shape[0], sum(swell_nodes))


        # Compute target as next time step (delta = coarse step)
        row, col = edge_index
        if self.add_targets :
            time_curr = time[:-self.time_dim]

            world_pos_curr = world_pos[:-self.time_dim]
            target_world_pos = torch.stack([world_pos[i + 1: i + 1 + self.time_dim] for i in range(world_pos_curr.shape[0])])
            phi_curr = phi[:-self.time_dim]
            target_phi = torch.stack([phi[i + 1: i + 1 + self.time_dim] for i in range(phi_curr.shape[0])])
            swelling_phi = torch.stack([swell_phi_tensor[i: i + self.time_dim + 1].T for i in range(phi_curr.shape[0])])
            frames = []
            for t in range(world_pos_curr.shape[0]) :
                if self.noise_level > 0.0 :
                    avg_conn_length = torch.max(torch.norm(mesh_pos[row] - mesh_pos[col], dim=-1))
                    world_pos_noise = torch.randn_like(world_pos_curr[t]) * self.noise_level * avg_conn_length
                    ux_dbc = node_type[:, 1] == 1
                    uy_dbc = node_type[:, 2] == 1
                    world_pos_noise[ux_dbc, 0] = 0.0
                    world_pos_noise[uy_dbc, 1] = 0.0

                    phi_range = torch.max(phi) - torch.min(phi)
                    phi_noise = torch.randn_like(phi_curr[t]) * self.noise_level * phi_range
                    phi_dbc = node_type[:, 3] == 1
                    phi_noise[phi_dbc] = 0.0

                    world_pos_t = world_pos_curr[t] + world_pos_noise
                    phi_t = phi_curr[t] + phi_noise
                else :
                    world_pos_t = world_pos_curr[t]
                    phi_t = phi_curr[t]

                swelling_phi_t = swelling_phi[t]
                target_world_pos_t = target_world_pos[t]
                target_phi_t = target_phi[t]
                frame = Data(mesh_pos = mesh_pos,
                            node_type = node_type,
                            mat_param = mat_param,
                            cells = cells,
                            edge_index=edge_index,
                            time = time_curr[t],
                            world_pos = world_pos_t,
                            phi = phi_t,
                            swelling_phi = swelling_phi_t,
                            target = torch.cat([target_world_pos_t, target_phi_t], dim=-1))
                frames.append(frame)
            return frames
        else :
            frames = [Data(mesh_pos = mesh_pos,
                           node_type = node_type,
                           mat_param = mat_param,
                           cells = cells,
                           edge_index=edge_index,
                           time = time[t],
                           world_pos = world_pos[t],
                           phi = phi[t],
                           swelling_phi = swell_phi_tensor[t]) for t in range(world_pos.shape[0])]
            return frames