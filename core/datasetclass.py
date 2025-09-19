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
    def __init__(self, data_dir, noise_level=None):
        """
        Args:
            data (array-like): Array of hydrogel data samples.
            labels (array-like): Array of labels corresponding to the data samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.noise_level = noise_level
        #list all .npz files in the directory ans save name in a list
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    def __len__(self):
        return len(self.data_files)
    def get_name(self, idx):
        return self.data_files[idx].rstrip('.npz')
    def __getitem__(self, idx):
        # Load data from .npz file saved in data_files
        sample = np.load(os.path.join(self.data_dir, self.data_files[idx]))
        #transform these to torch tensors
        mesh_pos = torch.tensor(sample['mesh_coords'], dtype=torch.float32)
        cells = torch.tensor(sample['cells'], dtype=torch.long)
        edge_index = cells_to_edge_index(cells)
        node_type = torch.tensor(sample["node_type"], dtype=torch.float32)
        u = torch.tensor(sample['u_time_series'], dtype=torch.float32)
        world_pos = mesh_pos + u
        phi = torch.tensor(sample["φ_time_series"], dtype=torch.float32).unsqueeze(-1)
        swell_phi = torch.tensor(sample["swell_time_series"], dtype=torch.float32).unsqueeze(-1)  # (T, 1)
        time = torch.tensor(sample["t"], dtype=torch.float32)
        mat_param = torch.tensor([sample['chi'].item(), sample['diffusivity'].item()], dtype=torch.float32)
        swell_nodes = node_type[:, 4] == 1
        swell_phi_tensor = torch.zeros_like(phi)
        swell_phi_tensor[:, swell_nodes, :] = swell_phi.unsqueeze(-1).expand(phi.shape[0], sum(swell_nodes), phi.shape[2])
    
        #create target as [target_u, target_phi] where target is next time step
        row, col = edge_index
        time_curr = time[:-1]
        time_next = time[1:]
        delta_time = torch.mean(time_next - time_curr)
        world_pos_curr = world_pos[:-1, :, :]
        target_world_pos = world_pos[1:, :, :]
        phi_curr = phi[:-1, :]
        target_phi = phi[1:, :]
        swelling_phi_curr = swell_phi_tensor[:-1, :, :]
        swelling_phi_next = swell_phi_tensor[1:, :, :]
        swelling_phi_rate = (swelling_phi_next - swelling_phi_curr)/delta_time
        if self.noise_level is not None:
            # u_curr noise define by average connection length of mesh
            # average connection length of mesh
            avg_conn_length = torch.mean(torch.norm(mesh_pos[row] - mesh_pos[col], dim=-1))
            world_pos_noise = torch.randn_like(world_pos_curr) * self.noise_level * avg_conn_length
            ux_dbc = node_type[:, 1] == 1
            uy_dbc = node_type[:, 2] == 1
            world_pos_noise[:, ux_dbc, 0] = 0.0 # no noise on fixed nodes
            world_pos_noise[:, uy_dbc, 1] = 0.0 # no noise on fixed nodes
            world_pos_curr = world_pos_curr + world_pos_noise
            # phi_curr noise defined by range of phi
            phi_range = torch.max(phi) - torch.min(phi)
            phi_noise = torch.randn_like(phi_curr) * self.noise_level * phi_range
            phi_dbc = node_type[:, 3] == 1
            phi_noise[:, phi_dbc] = 0.0 # no noise on fixed nodes
            phi_curr = phi_curr + phi_noise

        target = torch.cat([target_world_pos, target_phi], dim=-1)
        data = Data(world_pos = world_pos_curr, phi = phi_curr, swelling_phi = swelling_phi_curr, swelling_phi_rate = swelling_phi_rate, node_type = node_type, target = target, edge_index = edge_index, mesh_pos = mesh_pos, time = time_curr, cells = cells, mat_param = mat_param)

        return data
    
class HydrogelDatasetHistory(Dataset):
    def __init__(self, data_dir, noise_level=None):
        """
        Args:
            data (array-like): Array of hydrogel data samples.
            labels (array-like): Array of labels corresponding to the data samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.noise_level = noise_level
        #list all .npz files in the directory ans save name in a list
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
    def __len__(self):
        return len(self.data_files)
    def get_name(self, idx):
        return self.data_files[idx].rstrip('.npz')
    def __getitem__(self, idx):
        # Load data from .npz file saved in data_files
        sample = np.load(os.path.join(self.data_dir, self.data_files[idx]))
        #transform these to torch tensors
        mesh_pos = torch.tensor(sample['mesh_coords'], dtype=torch.float32)
        cells = torch.tensor(sample['cells'], dtype=torch.long)
        edge_index = cells_to_edge_index(cells)
        node_type = torch.tensor(sample["node_type"], dtype=torch.float32)
        u = torch.tensor(sample['u_time_series'], dtype=torch.float32)
        world_pos = mesh_pos + u
        phi = torch.tensor(sample["φ_time_series"], dtype=torch.float32).unsqueeze(-1)
        swell_phi = torch.tensor(sample["swell_time_series"], dtype=torch.float32).unsqueeze(-1)  # (T, 1)
        time = torch.tensor(sample["t"], dtype=torch.float32)
        mat_param = torch.cat([sample['chi'], sample["diffusivity"]], dtype=torch.float32)
        swell_nodes = node_type[:, 4] == 1
        swell_phi_tensor = torch.zeros_like(phi)
        swell_phi_tensor[:, swell_nodes, :] = swell_phi.unsqueeze(-1).expand(phi.shape[0], sum(swell_nodes), phi.shape[2])
    
        #create target as [target_u, target_phi] where target is next time step
        row, col = edge_index
        time_prev = time[:-2]
        time_curr = time[1:-1]
        time_next = time[2:]
        delta_time_prev = time_curr - time_prev
        delta_time_curr = time_next - time_curr
        world_pos_prev = world_pos[:-2, :, :]
        world_pos_curr = world_pos[1:-1, :, :]
        target_world_pos = world_pos[2:, :, :]
        phi_prev = phi[:-2, :]
        phi_curr = phi[1:-1, :]
        target_phi = phi[2:, :]
        swelling_phi_prev = swell_phi_tensor[:-2, :, :]
        swelling_phi_curr = swell_phi_tensor[1:-1, :, :]
        swelling_phi_next = swell_phi_tensor[2:, :, :]
        swelling_phi_rate_curr = (swelling_phi_next - swelling_phi_curr)/(delta_time_curr.unsqueeze(-1).unsqueeze(-1))
        swelling_phi_rate_prev = (swelling_phi_curr - swelling_phi_prev)/(delta_time_prev.unsqueeze(-1).unsqueeze(-1))
        if self.noise_level is not None:
            # u_curr noise define by average connection length of mesh
            # average connection length of mesh
            avg_conn_length = torch.mean(torch.norm(mesh_pos[row] - mesh_pos[col], dim=-1))
            world_pos_noise = torch.randn_like(world_pos_curr) * self.noise_level * avg_conn_length
            world_pos_prev_noise = torch.randn_like(world_pos_prev) * self.noise_level * avg_conn_length
            ux_dbc = node_type[:, 1] == 1
            uy_dbc = node_type[:, 2] == 1

            world_pos_noise[:, ux_dbc, 0] = 0.0 # no noise on fixed nodes
            world_pos_noise[:, uy_dbc, 1] = 0.0 # no noise on fixed nodes
            world_pos_prev_noise[:, ux_dbc, 0] = 0.0 # no noise on fixed nodes
            world_pos_prev_noise[:, uy_dbc, 1] = 0.0 # no noise on fixed nodes

            world_pos_curr = world_pos_curr + world_pos_noise
            world_pos_prev = world_pos_prev + world_pos_prev_noise
            # phi_curr noise defined by range of phi
            phi_range = torch.max(phi) - torch.min(phi)
            phi_noise = torch.randn_like(phi_curr) * self.noise_level * phi_range
            phi_prev_noise = torch.randn_like(phi_prev) * self.noise_level * phi_range
            phi_dbc = node_type[:, 3] == 1
            phi_noise[:, phi_dbc] = 0.0 # no noise on fixed nodes
            phi_prev_noise[:, phi_dbc] = 0.0 # no noise on fixed nodes
            phi_curr = phi_curr + phi_noise
            phi_prev = phi_prev + phi_prev_noise

        target = torch.cat([target_world_pos, target_phi], dim=-1)
        data = Data(world_pos = world_pos_curr, prev_world_pos = world_pos_prev, 
                    phi = phi_curr, prev_phi = phi_prev, 
                    swelling_phi = swelling_phi_curr, 
                    swelling_phi_rate = swelling_phi_rate_curr, swelling_phi_rate_prev = swelling_phi_rate_prev, 
                    node_type = node_type, target = target, 
                    edge_index = edge_index, mesh_pos = mesh_pos, time = time_curr, cells = cells,
                    mat_param = mat_param)

        return data
if __name__ == "__main__":
    dataset = HydrogelDataset(data_dir = "/mnt/c/Users/narun/Desktop/Project/hydrogel/gnn/dataset/free_swelling", noise_level=0.01)
    data = dataset[0]
    print(data.mat_param)
    print(len(dataset))