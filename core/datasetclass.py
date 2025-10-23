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
def triangles_to_edges(faces):
    """Computes mesh edges from triangles."""
    # collect edges from triangles
    edges = torch.cat((faces[:, 0:2],
                       faces[:, 1:3],
                       torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers, _ = torch.min(edges, dim=1)
    senders, _ = torch.max(edges, dim=1)

    packed_edges = torch.stack((senders, receivers), dim=1)
    unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
    senders, receivers = torch.unbind(unique_edges, dim=1)
    senders = senders.to(torch.int64)
    receivers = receivers.to(torch.int64)

    two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
    return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
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
        # deltat_ = 1e-2
        # num_timesteps = int(5 * 2/deltat_)
        # t_new = np.linspace(sample["t"][1], sample["t"][-1], num_timesteps)
        t_new = sample["t"]
        time = torch.tensor(t_new, dtype=torch.float32)
        
        mat_param = torch.tensor([sample['chi'].item(), sample['diffusivity'].item()], dtype=torch.float32)
        swell_nodes = node_type[:, 4] == 1
        swell_phi_tensor = torch.zeros_like(phi)
        swell_phi_tensor[:, swell_nodes, ] = swell_phi.unsqueeze(-1).expand(phi.shape[0], sum(swell_nodes), phi.shape[2])
    
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
        swelling_phi_rate_curr = (swelling_phi_next - swelling_phi_curr)
        swelling_phi_rate_prev = (swelling_phi_curr - swelling_phi_prev)
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
    

class HydrogelImageDataset(Dataset):
    def __init__(self, data_dir, H=32, W=16, noise_level=None):
        self.data_dir = data_dir
        self.H = H
        self.W = W
        self.noise_level = noise_level
        self.data_files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        sample = np.load(os.path.join(self.data_dir, self.data_files[idx]))
        mesh_pos = torch.tensor(sample['mesh_coords'], dtype=torch.float32)  # (N,2)
        u = torch.tensor(sample['u_time_series'], dtype=torch.float32)      # (T,N,2)
        phi = torch.tensor(sample['φ_time_series'], dtype=torch.float32)    # (T,N)
        swelling_phi = torch.tensor(sample['swell_time_series'], dtype=torch.float32)  # (T,N_swell)
        node_type = torch.tensor(sample['node_type'], dtype=torch.float32)  # (N, d)
        time = torch.tensor(sample['t'], dtype=torch.float32)
        mat_param = torch.tensor([sample['chi'].item(), sample['diffusivity'].item()], dtype=torch.float32)
        
        T, N, _ = u.shape
        # world_pos = mesh_pos.unsqueeze(0) + u  # (T,N,2)
        
        # Channels: x_disp, y_disp, phi, load
        C = 4
        img_tensor = torch.zeros(T, C, self.H, self.W, dtype=torch.float32)
        
        # Map nodes to grid
        x_norm = (mesh_pos[:,0] - mesh_pos[:,0].min()) / (mesh_pos[:,0].ptp() + 1e-8)
        y_norm = (mesh_pos[:,1] - mesh_pos[:,1].min()) / (mesh_pos[:,1].ptp() + 1e-8)
        x_idx = (x_norm * (self.W-1)).long()
        y_idx = (y_norm * (self.H-1)).long()
        
        swell_nodes = node_type[:,4] == 1
        
        for t in range(T):
            for i in range(N):
                img_tensor[t, 0, y_idx[i], x_idx[i]] = u[t,i,0]  # x disp
                img_tensor[t, 1, y_idx[i], x_idx[i]] = u[t,i,1]  # y disp
                img_tensor[t, 2, y_idx[i], x_idx[i]] = phi[t,i]           # phi
                if swell_nodes[i]:
                    img_tensor[t, 3, y_idx[i], x_idx[i]] = swelling_phi[t,0]  # load
    
        # Target: delta to next timestep
        delta_tensor = img_tensor[1:] - img_tensor[:-1]
        input_tensor = img_tensor[:-1]
        time_curr = time[:-1]
        
        return {
            'input': input_tensor,  # (T-1, C, H, W)
            'delta': delta_tensor,  # (T-1, C, H, W)
            'time': time_curr,
            'mat_param': mat_param
        }
def generate_history_sequences(n, max_history):
    """
    Generate all possible integer sequences from 0..n-1
    with up to `max_history` previous steps for each element,
    without duplicates. Output as list of torch tensors.

    Example:
        n = 5, max_history = 2
        Output:
            [tensor([0]), tensor([0,1]), tensor([1,2]), 
             tensor([0,1,2]), tensor([1,2,3]), tensor([2,3,4]), 
             tensor([3,4]), tensor([4])]
    """
    sequences = set()
    for i in range(n):              # End index from 0..n-1
        for h in range(max_history + 1):
            start = max(0, i - h)
            seq = tuple(range(start, i + 1))
            if seq:
                sequences.add(seq)
                
    # Sort by starting index then length, and convert to tensors
    sorted_seqs = sorted(sequences, key=lambda x: (x[0], len(x)))
    return sorted_seqs

class HydrogelDatasetFlexibleHistory(Dataset):
    def __init__(self, data_dir, max_history = 0, noise_level=None):
        """
        Args:
            data (array-like): Array of hydrogel data samples.
            labels (array-like): Array of labels corresponding to the data samples.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.noise_level = noise_level
        self.max_history = max_history
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

        target_world_pos = world_pos[1:, :, :]
        target_phi= phi[1:, :]

        world_pos_curr = world_pos[:-1, :, :]
        phi_curr = phi[:-1, :]
        swelling_phi_curr = swell_phi_tensor[:-1, :]
        swelling_phi_next = swell_phi_tensor[1:, :]
        swelling_phi_rate = swelling_phi_next - swelling_phi_curr
        time_curr = time[:-1]
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

        # T, N, D = world_pos_curr.shape
        # mesh_pos = mesh_pos.unsqueeze(0).expand(T, -1, -1)  # (T, N, D)
        # edge_index = edge_index.unsqueeze(0).expand(T, -1, -1)  # (T, 2, E)
        # node_type = node_type.unsqueeze(0).expand(T, -1, -1)  # (T, N, 5)
        # mat_param = mat_param.unsqueeze(0).unsqueeze(0).expand(T, N, -1)  # (T, N, 2)


        data = Data(world_pos = world_pos_curr, phi = phi_curr, swelling_phi = swelling_phi_curr, next_swelling_phi = swelling_phi_next,
                    rate_swelling_phi = swelling_phi_rate, node_type = node_type, target = target, 
                    edge_index = edge_index, mesh_pos = mesh_pos, time = time_curr, cells = cells,
                    mat_param = mat_param)
        graphs = []
        for t in range(data.world_pos.shape[0]) :
            graph = Data(
                    world_pos=data.world_pos[t],
                    phi=data.phi[t],
                    node_type=data.node_type,
                    target=data.target[t],
                    edge_index=data.edge_index,
                    mesh_pos=data.mesh_pos,
                    time=data.time[t],
                    swelling_phi=data.swelling_phi[t],
                    next_swelling_phi = data.next_swelling_phi[t],
                    rate_swelling_phi = data.rate_swelling_phi[t],
                    cells=data.cells,
                    mat_param=data.mat_param
                )
            graphs.append(graph)
        history_indices = generate_history_sequences(world_pos_curr.shape[0], max_history=self.max_history)

        return graphs, history_indices
    

if __name__ == "__main__":
    dataset = HydrogelDataset(data_dir = "/mnt/c/Users/narun/Desktop/Project/hydrogel/gnn/dataset/free_swelling", noise_level=0.01,
                              add_targets=None,
                              split_to_frames=True)
    data = dataset[0]
    print(data[0].mat_param)
    print(len(dataset))

