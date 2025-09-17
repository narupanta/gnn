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
        # world_pos = mesh_pos + u
        phi = torch.tensor(sample["φ_time_series"], dtype=torch.float32).unsqueeze(-1)
        swell_phi = torch.tensor(sample["swell_time_series"], dtype=torch.float32)
        # swell_phi_tensor = torch.zeros_like(phi)
        # swell_nodes = node_type[:, :, 4] == 1
        # swell_phi_tensor[:, swell_nodes , :] = swell_phi
        time = torch.tensor(sample["t"], dtype=torch.float32)
        #create input features as [u, phi, node_type]
        #create target as [target_u, target_phi] where target is next time step
        row, col = edge_index
        time_curr = time[:-1]
        u_curr = u[:-1, :, :]
        target_u = u[1:, :, :]
        phi_curr = phi[:-1, :]
        target_phi = phi[1:, :]
        swelling_phi_curr = swell_phi[:-1]
        if self.noise_level is not None:
            # u_curr noise define by average connection length of mesh
            # average connection length of mesh
            avg_conn_length = torch.mean(torch.norm(mesh_pos[row] - mesh_pos[col], dim=-1))
            u_noise = torch.randn_like(u_curr) * self.noise_level * avg_conn_length
            ux_dbc = node_type[:, 1] == 1
            uy_dbc = node_type[:, 2] == 1
            u_noise[:, ux_dbc, 0] = 0.0 # no noise on fixed nodes
            u_noise[:, uy_dbc, 1] = 0.0 # no noise on fixed nodes
            u_curr = u_curr + u_noise
            # phi_curr noise defined by range of phi
            phi_range = torch.max(phi) - torch.min(phi)
            phi_noise = torch.randn_like(phi_curr) * self.noise_level * phi_range
            phi_dbc = node_type[:, 3] == 1
            phi_noise[:, phi_dbc] = 0.0 # no noise on fixed nodes
            phi_curr = phi_curr + phi_noise


        x = torch.cat([u_curr, phi_curr, node_type.unsqueeze(0).expand(phi_curr.shape[0], phi_curr.shape[1], -1)], dim=-1)
        # edge attr defined as relative position and distance of connected nodes
        row, col = edge_index
        #relative position and distance as edge features    
        rel_position = mesh_pos[col] - mesh_pos[row, :]
        distance = torch.norm(rel_position, dim=-1, keepdim=True)

        rel_u = u_curr[:, col, :] - u_curr[:, row, :]
        distance_u  = torch.norm(rel_u, dim=-1, keepdim=True)
        rel_phi = phi_curr[:, col] - phi_curr[:, row]
        edge_attr = torch.cat([rel_position.unsqueeze(0).expand(phi_curr.shape[0], -1, rel_position.shape[1]), distance.unsqueeze(0).expand(phi_curr.shape[0], -1, distance.shape[1]), rel_u, distance_u, rel_phi], dim=-1)
        y = torch.cat([target_u, target_phi], dim=-1)

        data = Data(x = x, y = y, edge_index = edge_index, edge_attr = edge_attr, pos = mesh_pos, time = time_curr, swelling_phi = swelling_phi_curr, cells = cells)

        return data
if __name__ == "__main__":
    dataset = HydrogelDataset(data_dir = "/mnt/c/Users/narun/Desktop/Project/hydrogel/gnn/dataset/uniaxial_signal", noise_level=0.01)
    data = dataset[0]
    print(data)
    print(len(dataset))