# encode_process_decode (meshgraphnet style)


"""
Encode-Process-Decode GNN (PyTorch + PyTorch Geometric)

- Encoder: MLPs for node and edge features
- Process: message-passing steps (custom MessagePassing) that update edges then nodes
- Decode: MLPs that produce final node/edge/global outputs

Example usage and training loop included.

Requirements:
- torch
- torch_geometric (PyTorch Geometric)

Run: pip install torch torch_geometric (follow PyG instructions for your CUDA)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .normalization import Normalizer
from scipy.spatial import Delaunay
from .normalization import Normalizer
import torch_scatter
from torch_geometric.nn import knn_graph
from torch_geometric.utils import softmax
import os

def sinusoidal_time_embedding(t, dim=16):
    """
    Create a sinusoidal time embedding.
    t: scalar tensor [1] or float
    dim: embedding dimension (even number)
    """
    assert dim % 2 == 0, "Embedding dimension must be even"
    device = t.device if isinstance(t, torch.Tensor) else 'cpu'

    freqs = 2 ** torch.arange(dim // 2, dtype=torch.float32, device=device)
    t = t.float()
    emb = torch.cat([torch.sin(freqs * t), torch.cos(freqs * t)], dim=-1)
    return emb  # shape [dim]
    
def MLP(in_dim, out_dim, hidden_dims=(128, 128), activate_final=False, layer_norm=False):
    layers = []
    last = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    if activate_final:
        layers.append(nn.ReLU())
    if layer_norm:
        layers.append(nn.LayerNorm(out_dim))
    return nn.Sequential(*layers)

class EdgeNodeMessagePassing(MessagePassing):
    """Custom MP layer that:
      1) updates edge attributes using (x_i, x_j, e_ij)
      2) computes messages using updated edge attributes and x_j
      3) aggregates messages and updates node attributes

    This follows the Encode-Process-Decode pattern where the processing layer is a message-passing step.
    """

    def __init__(self, hidden_dim, attention):
        super().__init__(aggr='add')  # 'add' aggregation
        self.attention = attention
        # Node attention: compute attention scores for neighbors
        if self.attention :
            self.attn_lin = torch.nn.Linear(hidden_dim, hidden_dim)
        # edge update: (x_i, x_j, e_ij) -> e'_ij
        self.edge_mlp = MLP(hidden_dim * 2 + hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), layer_norm=True)
        # node update: (x_i, aggregated_message) -> x'_i
        self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), layer_norm=True)
    def forward(self, node_feat, edge_index, edge_feat):
        
        # Node update
        new_node_features = self.propagate(edge_index, x= node_feat, edge_attr = edge_feat)        
        
        # Edge update
        row, col = edge_index
        new_edge_features = self.edge_mlp(torch.cat([node_feat[row], node_feat[col], edge_feat], dim=-1))
        
        # Add residuals
        new_node_features = new_node_features + node_feat
        new_edge_features = new_edge_features + edge_feat     
                
        return new_node_features, new_edge_features
    
    def message(self, x_i, x_j, edge_attr, index):
        # Compute attention score
        if self.attention :
            alpha = (self.attn_lin(x_i) * self.attn_lin(x_j)).sum(dim=-1)  # [E]
            alpha = F.leaky_relu(alpha)
            alpha = softmax(alpha, index=index)  # normalize per target node
            alpha = alpha.unsqueeze(-1)  # [E, 1]

            # Message is neighbor feature weighted by attention
            msg = alpha * self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
        else :
            features = torch.cat([x_i, x_j, edge_attr], dim=-1)        
            msg = self.edge_mlp(features)
        return msg
    def update(self, aggr_out, x):
        # aggr_out has shape [num_nodes, out_channels]        
        tmp = torch.cat([aggr_out, x], dim=-1)                
       
        # Step 5: Return new node embeddings.        
        return self.node_mlp(tmp)
# class EdgeNodeMessagePassing(torch.nn.Module):
#     """Graph Network block with residual connections."""
#     def __init__(self, hidden_dim, attention):
#         super().__init__()  # 'add' aggregation
#         self.attention = attention
#         # Node attention: compute attention scores for neighbors
#         if self.attention :
#             self.attn_lin = torch.nn.Linear(hidden_dim, hidden_dim)
#         # edge update: (x_i, x_j, e_ij) -> e'_ij
#         self.edge_mlp = MLP(hidden_dim * 2 + hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), layer_norm=True)
#         # node update: (x_i, aggregated_message) -> x'_i
#         self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), layer_norm=True)
#     def forward(self, node_feat, edge_index, edge_feat):
#         senders, receivers = edge_index
#         node_latents, mesh_edge_latents = node_feat, edge_feat

#         edge_input = torch.cat([node_latents[senders, :], node_latents[receivers, :], mesh_edge_latents], dim=-1)
#         new_mesh_edge_latents = self.edge_mlp(edge_input)
#         if self.attention :
#             attention_input = self.attn_lin(new_mesh_edge_latents)
#             attention = F.softmax(attention_input, dim=0)
#             new_mesh_edge_latents = attention * new_mesh_edge_latents
#             # torch.einsum('ij,ikj->ikj', attention, new_mesh_edge_latents)

#         aggr = torch_scatter.scatter_add(new_mesh_edge_latents.float(), receivers, dim=0)
#         node_input = torch.cat([node_latents, aggr], dim=-1)
#         new_node_features = self.node_mlp(node_input) + node_feat
#         new_edge_features = new_mesh_edge_latents + edge_feat

#         return new_node_features, new_edge_features
class EncodeProcessDecode(nn.Module):
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hidden_size=128,
                 process_steps=3,
                 node_out_dim=0, 
                 attention=False,
                 with_mat_params = True,
                 device = "cpu"):
        super().__init__()
        # Encoders
        self.node_encoder = MLP(node_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.attention = attention
        self.with_mat_params = with_mat_params
        # Processor: stack of message passing steps
        self.process_steps = process_steps
        self.processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(process_steps)])

        # Decoder MLPs
        self.node_decoder = MLP(hidden_size, node_out_dim, hidden_dims=(hidden_size,), layer_norm=False) if node_out_dim > 0 else None

        self.device = device
        self.node_features_normalizer = Normalizer(node_in_dim, device)
        self.edge_features_normalizer = Normalizer(edge_in_dim, device)
        self.output_normalizer = Normalizer(node_out_dim, device)
    def forward(self, batch):
        # x: [N, node_in_dim]
        # edge_attr: [E, edge_in_dim]
        # batch: [N] batch index if using batching, required for global readout

        # Encode
        # Normalize x and edge features before input to encoder
        x = self._build_node_features(batch)
        e = self._build_edge_features(batch)
        x_h = self.node_encoder(self.node_features_normalizer(x))
        e_h = self.edge_encoder(self.edge_features_normalizer(e))

        # Process (multiple message-passing steps)
        for i, proc in enumerate(self.processors):
            x_h, e_h = proc(x_h, batch.edge_index, e_h)

        # Decode
        output = self.node_decoder(x_h)

        return output
    def _build_node_features(self, graph) :
        time_emb = sinusoidal_time_embedding(graph.time, dim = 4)
        # time_emb = graph.time
        u = graph.world_pos - graph.mesh_pos
        phi = graph.phi
        swell_phi = graph.swelling_phi
        swelling_phi_rate = graph.swelling_phi_rate
        node_type = graph.node_type
        time_emb = time_emb.unsqueeze(0).repeat(u.shape[0], 1)
        mat_param = graph.mat_param.unsqueeze(0).repeat(u.shape[0], 1)
        # mat_param = _apply_film
        if self.with_mat_params :
            x = torch.cat([u, phi, swell_phi, swelling_phi_rate, node_type, mat_param], dim = -1)
        else :
            x = torch.cat([u, phi, swell_phi, swelling_phi_rate, node_type], dim = -1)
        return x
    def _build_edge_features(self, graph) :
        senders, receivers = graph.edge_index[0], graph.edge_index[1]
        rel_position = graph.mesh_pos[senders, :] - graph.mesh_pos[receivers, :]
        distance = torch.norm(rel_position, dim=-1, keepdim=True)

        rel_world_pos = graph.world_pos[senders, :] - graph.world_pos[receivers, :]
        distance_world_pos  = torch.norm(rel_world_pos, dim=-1, keepdim=True)
        rel_phi = graph.phi[senders] - graph.phi[receivers]
        edge_features = torch.cat([rel_position, distance, rel_world_pos, distance_world_pos, rel_phi], dim=-1)
        return edge_features
    def loss(self, graph):
        target = graph.target
        curr = torch.cat([graph.world_pos, graph.phi], dim = -1)

        target_delta = target - curr
        target_delta_normalize = self.output_normalizer(target_delta)
        pred_delta = self.forward(graph)

        error = (pred_delta - target_delta_normalize)**2
        # MSE loss on the change (delta) prediction
        # exclude nodes with dbc (node_type 1, 2, 3 in last feature)
        ux_dbc_nodes = graph.node_type[:, 1] == 1
        uy_dbc_nodes = graph.node_type[:, 2] == 1
        phi_dbc_nodes = graph.node_type[:, 3] == 1
        graph_error_ux = torch.mean(error[:, 0].masked_select(~ux_dbc_nodes))
        graph_error_uy = torch.mean(error[:, 1].masked_select(~uy_dbc_nodes))
        graph_error_phi = torch.mean(error[:, 2].masked_select(~phi_dbc_nodes))
        return graph_error_ux + graph_error_uy + graph_error_phi, graph_error_ux, graph_error_uy, graph_error_phi
    def predict(self, graph):
        with torch.no_grad():
            pred_delta_normalized = self.forward(graph)
            pred_delta = self.output_normalizer.inverse(pred_delta_normalized)
            curr = torch.cat([graph.world_pos, graph.phi], dim = -1)
            ux_dbc_nodes = graph.node_type[:, 1] == 1
            uy_dbc_nodes = graph.node_type[:, 2] == 1
            phi_dbc_nodes = graph.node_type[:, 3] == 1
            pred_delta[ux_dbc_nodes, 0] = 0.0 # no change on fixed nodes
            pred_delta[uy_dbc_nodes, 1] = 0.0 # no change on fixed nodes
            pred_delta[phi_dbc_nodes, 2] = 0.0 # no change on fixed nodes
            pred = curr + pred_delta
        return pred
    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self.output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self.node_features_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self.edge_features_normalizer, os.path.join(path, "edge_features_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self.output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self.node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self.edge_features_normalizer = torch.load(os.path.join(path, "edge_features_normalizer.pth"))



# give me code for encode process decode 


class EncodeProcessDecodeHistory(nn.Module):
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hidden_size=128,
                 process_steps=3,
                 node_out_dim=0, 
                 attention=False,
                 with_mat_params = False,
                 device = "cpu"):
        super().__init__()
        # Encoders
        self.node_encoder = MLP(node_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.attention = attention
        self.with_mat_params = with_mat_params
        # Processor: stack of message passing steps
        self.process_steps = process_steps
        self.processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(process_steps)])

        # Decoder MLPs
        self.world_pos_decoder = MLP(hidden_size, 2, hidden_dims=(hidden_size,), layer_norm=False) if node_out_dim > 0 else None
        self.phi_decoder = MLP(hidden_size, 1, hidden_dims=(hidden_size,), layer_norm=False) if node_out_dim > 0 else None
        self.device = device
        self.node_features_normalizer = Normalizer(node_in_dim, device)
        self.edge_features_normalizer = Normalizer(edge_in_dim, device)
        self.output_normalizer = Normalizer(node_out_dim, device)
    def forward(self, batch):
        # x: [N, node_in_dim]
        # edge_attr: [E, edge_in_dim]
        # batch: [N] batch index if using batching, required for global readout

        # Encode
        # Normalize x and edge features before input to encoder
        x = self._build_node_features(batch)
        e = self._build_edge_features(batch)
        x_h = self.node_encoder(self.node_features_normalizer(x))
        e_h = self.edge_encoder(self.edge_features_normalizer(e))

        # Process (multiple message-passing steps)
        for i, proc in enumerate(self.processors):
            x_h, e_h = proc(x_h, batch.edge_index, e_h)

        # Decode
        acc_world_pos = self.world_pos_decoder(x_h)
        acc_phi = self.phi_decoder(x_h)
        output = torch.cat([acc_world_pos, acc_phi], dim = -1)

        return output
    def _build_node_features(self, graph) :
        u = graph.world_pos - graph.mesh_pos
        u_prev = graph.prev_world_pos - graph.mesh_pos
        u_delta = u - u_prev
        phi_delta = graph.phi - graph.prev_phi
        prev_phi = graph.prev_phi
        swell_phi = graph.swelling_phi
        swelling_phi_rate = graph.swelling_phi_rate
        swelling_phi_rate_prev = graph.swelling_phi_rate_prev
        node_type = graph.node_type
        mat_param = graph.mat_param.unsqueeze(0).repeat(u.shape[0], 1)
        if self.with_mat_params :
            x = torch.cat([u_delta, phi_delta, swell_phi, swelling_phi_rate, swelling_phi_rate_prev, node_type, mat_param], dim = -1)
        else :
            x = torch.cat([u_delta, phi_delta, swell_phi, swelling_phi_rate, swelling_phi_rate_prev, node_type], dim = -1)
        return x
    def _build_edge_features(self, graph) :
        senders, receivers = graph.edge_index[0], graph.edge_index[1]
        rel_position = graph.mesh_pos[senders, :] - graph.mesh_pos[receivers, :]
        distance = torch.norm(rel_position, dim=-1, keepdim=True)

        rel_world_pos = graph.world_pos[senders, :] - graph.world_pos[receivers, :]
        distance_world_pos  = torch.norm(rel_world_pos, dim=-1, keepdim=True)
        rel_phi = graph.phi[senders] - graph.phi[receivers]
        edge_features = torch.cat([rel_position, distance, rel_world_pos, distance_world_pos, rel_phi], dim=-1)
        return edge_features
    def loss(self, graph):
        target = graph.target
        # prev = torch.cat([graph.prev_world_pos, graph.prev_phi], dim = -1)
        curr = torch.cat([graph.world_pos, graph.phi], dim = -1)
        # prev = torch.cat([graph.prev_world_pos, graph.prev_phi], dim = -1)
        target_delta = target - curr
        target_delta_normalize = self.output_normalizer(target_delta)

        pred_delta = self.forward(graph)

        error = (pred_delta - target_delta_normalize)**2
        # MSE loss on the change (delta) prediction
        # exclude nodes with dbc (node_type 1, 2, 3 in last feature)
        ux_dbc_nodes = graph.node_type[:, 1] == 1
        uy_dbc_nodes = graph.node_type[:, 2] == 1
        phi_dbc_nodes = graph.node_type[:, 3] == 1
        graph_error_ux = torch.mean(error[:, 0].masked_select(~ux_dbc_nodes))
        graph_error_uy = torch.mean(error[:, 1].masked_select(~uy_dbc_nodes))
        graph_error_phi = torch.mean(error[:, 2].masked_select(~phi_dbc_nodes))
        return graph_error_ux + graph_error_uy + 10 * graph_error_phi, graph_error_ux, graph_error_uy, graph_error_phi
    def predict(self, graph):
        with torch.no_grad():
            pred_delta_normalized = self.forward(graph)
            pred_delta = self.output_normalizer.inverse(pred_delta_normalized)
            curr = torch.cat([graph.world_pos, graph.phi], dim = -1)
            # prev = torch.cat([graph.prev_world_pos, graph.prev_phi], dim = -1)

            ux_dbc_nodes = graph.node_type[:, 1] == 1
            uy_dbc_nodes = graph.node_type[:, 2] == 1
            phi_dbc_nodes = graph.node_type[:, 3] == 1
            # pred_delta = curr + pred_acc 
            pred_delta[ux_dbc_nodes, 0] = 0.0 # no change on fixed nodes
            pred_delta[uy_dbc_nodes, 1] = 0.0 # no change on fixed nodes
            pred_delta[phi_dbc_nodes, 2] = 0.0 # no change on fixed nodes
            pred = curr + pred_delta
        return pred
    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self.output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self.node_features_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self.edge_features_normalizer, os.path.join(path, "edge_features_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self.output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self.node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self.edge_features_normalizer = torch.load(os.path.join(path, "edge_features_normalizer.pth"))

class Swish(torch.nn.Module):
    """Swish activation function."""
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class EncodeProcessDecodeMultiScale(nn.Module):
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hidden_size=128,
                 process_steps=3,
                 coarse_process_steps = 3,
                 node_out_dim=0, 
                 attention=False,
                 with_mat_params = True,
                 sample_ratio = 0.5,
                 time_dim = 1,
                 device = "cpu"):
        super().__init__()
        # Encoders
        self.node_encoder = MLP(node_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.attention = attention
        self.with_mat_params = with_mat_params
        self.sample_ratio = sample_ratio
        self.time_dim = time_dim
        self.node_out_dim = node_out_dim
        # Processor: stack of message passing steps
        self.process_steps = process_steps
        self.processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(process_steps)])
        if sample_ratio > 0:
            self.coarse_edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
            self.coarse_edge_features_normalizer = Normalizer(edge_in_dim, device)
            self.coarse_processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(coarse_process_steps)])

        # Decoder MLPs
        # self.node_decoder = MLP(2*hidden_size, node_out_dim, hidden_dims=(hidden_size,), layer_norm=False) if sample_ratio > 0 else MLP(hidden_size, node_out_dim, hidden_dims=(hidden_size,), layer_norm=False)
        self.node_decoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size//2, 1),
            Swish(),
            nn.Conv1d(hidden_size//2, node_out_dim * time_dim, 1)
        )
        self.device = device
        self.node_features_normalizer = Normalizer(node_in_dim, device)
        self.edge_features_normalizer = Normalizer(edge_in_dim, device)
        self.output_normalizer = Normalizer(node_out_dim, device)
    def forward(self, batch):
        # x: [N, node_in_dim]
        # edge_attr: [E, edge_in_dim]
        # batch: [N] batch index if using batching, required for global readout

        # Encode
        # Normalize x and edge features before input to encoder
        x = self._build_node_features(batch)
        e = self._build_edge_features(batch)
        x_h = self.node_encoder(self.node_features_normalizer(x))
        e_h = self.edge_encoder(self.edge_features_normalizer(e))
        if self.sample_ratio > 0 :
            ce, c_edge_index, first_indices = self._build_coarse_edge_features(batch)
            ce_h = self.coarse_edge_encoder(self.coarse_edge_features_normalizer(ce))
            cx_h = x_h[first_indices]
        # Process (multiple message-passing steps)
        for i, proc in enumerate(self.processors):
            x_h, e_h = proc(x_h, batch["edge_index"], e_h)

        if self.sample_ratio > 0 :
        
            for i, proc in enumerate(self.coarse_processors) :
                cx_h, ce_h = proc(cx_h, c_edge_index, ce_h)

            map_back = torch.zeros_like(x_h)
            map_back[first_indices] = cx_h
            x_h = torch.cat([x_h, map_back], dim = -1)
        
        # Decode
        x_h = x_h.unsqueeze(-1)  
        decoded = self.node_decoder(x_h).squeeze(-1)
        # dt = torch.arange(1, self.time_dim + 1).to(device=self.device)
        delta = decoded.reshape(self.time_dim, -1, self.node_out_dim)
        # delta = dt.unsqueeze(-1).unsqueeze(-1) * delta 

        return delta
    def _build_node_features(self, graph) :
        # time_emb = sinusoidal_time_embedding(graph["time"], dim = self.time_dim)
        u = graph["world_pos"] - graph["mesh_pos"]
        phi = graph["phi"]
        swell_phi = graph["swelling_phi"]
        swelling_phi_rate = graph["swelling_phi_rate"]
        node_type = graph["node_type"]
        # time_emb = time_emb.unsqueeze(0).repeat(u.shape[0], 1)
        mat_param = graph["mat_param"].unsqueeze(0).repeat(u.shape[0], 1)
        # mat_param = _apply_film
        if self.with_mat_params :
            x = torch.cat([phi, swell_phi, swelling_phi_rate, node_type, mat_param], dim = -1)
        else :
            x = torch.cat([phi, swell_phi, swelling_phi_rate, node_type], dim = -1)
        return x
    def _build_edge_features(self, graph) :
        senders, receivers = graph["edge_index"][0], graph["edge_index"][1]
        rel_position = graph["mesh_pos"][senders, :] - graph["mesh_pos"][receivers, :]
        distance = torch.norm(rel_position, dim=-1, keepdim=True)

        rel_world_pos = graph["world_pos"][senders, :] - graph["world_pos"][receivers, :]
        distance_world_pos  = torch.norm(rel_world_pos, dim=-1, keepdim=True)
        rel_phi = graph["phi"][senders] - graph["phi"][receivers]
        edge_features = torch.cat([rel_position, distance, rel_world_pos, distance_world_pos, rel_phi], dim=-1)
        return edge_features
    
    def _build_coarse_edge_features(self, graph):
        """
        Build coarse edges by sampling a subset of nodes with FPS (or random) instead of voxel sampling.
        sample_ratio: fraction of nodes to keep in coarse graph (0 < sample_ratio <= 1)
        """

        def farthest_point_sampling(points, num_samples):
            N = points.shape[0]
            device = points.device
            sampled_idx = torch.zeros(num_samples, dtype=torch.long, device=device)
            # pick first point randomly
            sampled_idx[0] = torch.randint(0, N, (1,), device=device)
            dist = torch.full((N,), float('inf'), device=device)
            for i in range(1, num_samples):
                last = points[sampled_idx[i-1]].unsqueeze(0)
                dist = torch.minimum(dist, torch.norm(points - last, dim=-1))
                sampled_idx[i] = torch.argmax(dist)
            return sampled_idx

        mesh_pos = graph["mesh_pos"]
        world_pos = graph["world_pos"]
        phi = graph["phi"]
        k = 4 if mesh_pos.shape[-1] == 3 else 3

        N_coarse = max(1, int(mesh_pos.shape[0] * self.sample_ratio))
        first_indices = farthest_point_sampling(mesh_pos, N_coarse)  # indices of coarse nodes in original

        # Sampled points in coarse graph
        sampled_points = mesh_pos[first_indices]

        # Build coarse KNN graph
        sampled_edge_index = knn_graph(sampled_points, k=k, loop=False)
        # Map edges back to original nodes
        sampled_edge_index_by_original = first_indices[sampled_edge_index]

        # Senders / receivers
        sampled_senders, sampled_receivers = sampled_edge_index_by_original[0], sampled_edge_index_by_original[1]

        # Edge features
        rel_position = mesh_pos[sampled_senders, :] - mesh_pos[sampled_receivers, :]
        distance = torch.norm(rel_position, dim=-1, keepdim=True)
        rel_world_pos = world_pos[sampled_senders, :] - world_pos[sampled_receivers, :]
        distance_world_pos  = torch.norm(rel_world_pos, dim=-1, keepdim=True)
        rel_phi = phi[sampled_senders] - phi[sampled_receivers]
        edge_features = torch.cat([rel_position, distance, rel_world_pos, distance_world_pos, rel_phi], dim=-1)

        return edge_features, sampled_edge_index, first_indices

    def loss(self, graph):
        target = graph["target"]
        curr = torch.cat([graph["world_pos"], graph["phi"]], dim = -1)

        delta_time = graph["delta_time"]
        check = torch.ones_like((target - curr))/delta_time.unsqueeze(-1).unsqueeze(-1)
        target_delta = (target - curr)/delta_time.unsqueeze(-1).unsqueeze(-1)
        target_delta_normalize = self.output_normalizer(target_delta)
        pred_delta = self.forward(graph)

        error = (pred_delta - target_delta_normalize)**2
        # MSE loss on the change (delta) prediction
        # exclude nodes with dbc (node_type 1, 2, 3 in last feature)
        ux_dbc_nodes = graph["node_type"][:, 1] == 1
        uy_dbc_nodes = graph["node_type"][:, 2] == 1
        phi_dbc_nodes = graph["node_type"][:, 3] == 1
        graph_error_ux = torch.mean(error[:, ~ux_dbc_nodes, 0])
        graph_error_uy = torch.mean(error[:, ~uy_dbc_nodes, 1])
        graph_error_phi = torch.mean(error[:, ~phi_dbc_nodes, 2])
        return graph_error_ux + graph_error_uy + graph_error_phi, graph_error_ux, graph_error_uy, graph_error_phi
    def predict(self, graph):
        with torch.no_grad():
            pred_dot_normalized = self.forward(graph)
            pred_dot = self.output_normalizer.inverse(pred_dot_normalized)
            pred_delta = pred_dot * graph["delta_time"].unsqueeze(-1).unsqueeze(-1) 
            curr = torch.cat([graph["world_pos"], graph["phi"]], dim = -1)
            ux_dbc_nodes = graph["node_type"][:, 1] == 1
            uy_dbc_nodes = graph["node_type"][:, 2] == 1
            phi_dbc_nodes = graph["node_type"][:, 3] == 1
            pred_delta[:, ux_dbc_nodes, 0] = 0.0 # no change on fixed nodes
            pred_delta[:, uy_dbc_nodes, 1] = 0.0 # no change on fixed nodes
            pred_delta[:, phi_dbc_nodes, 2] = 0.0 # no change on fixed nodes
            pred = curr + pred_delta
        return pred
    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self.output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self.node_features_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self.edge_features_normalizer, os.path.join(path, "edge_features_normalizer.pth"))
        if self.sample_ratio > 0 :
            torch.save(self.coarse_edge_features_normalizer, os.path.join(path, "coarse_edge_features_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self.output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self.node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self.edge_features_normalizer = torch.load(os.path.join(path, "edge_features_normalizer.pth"))
        if self.sample_ratio > 0 :
            self.coarse_edge_features_normalizer = torch.load(os.path.join(path, "coarse_edge_features_normalizer.pth"))
import math
class SinusoidalTimeEncoding(nn.Module):
    """
    Sinusoidal positional encoding like in Transformers,
    but applied to timestep indices.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: [B, H] integers (0..H-1)
        Returns:
            enc: [B, H, dim]
        """
        device = timesteps.device
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=device).float() / half_dim)
        angles = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)  # [B,H,half_dim]

        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [B,H,dim]
        return emb
class TemporalAttentionPooling(nn.Module):
    """
    Pools variable-length graph history [B, H, N, D] → [B, N, D]
    using temporal attention with sinusoidal time encoding.
    """
    def __init__(self, in_dim, out_dim, time_dim=16):
        super().__init__()
        self.time_encoder = SinusoidalTimeEncoding(time_dim)
        self.attn_proj = nn.Linear(in_dim + time_dim, 1)  # attention score
        self.out_proj = nn.Linear(in_dim, out_dim)

    def forward(self, h_seq, mask=None):
        """
        Args:
            h_seq: [B, H, N, D] graph embeddings over time
            mask:  [B, H] binary mask (1=valid, 0=pad), optional
        """
        h_seq = h_seq.unsqueeze(0)
        B, H, N, D = h_seq.shape

        # timestep indices 0..H-1
        timesteps = torch.arange(H, device=h_seq.device).unsqueeze(0).expand(B, H)  # [B,H]

        # add time embeddings
        t_emb = self.time_encoder(timesteps)              # [B,H,time_dim]
        t_emb = t_emb.unsqueeze(2).expand(B, H, N, -1)    # [B,H,N,time_dim]

        h_seq_time = torch.cat([h_seq, t_emb], dim=-1)    # [B,H,N,D+time_dim]

        # attention scores
        scores = self.attn_proj(h_seq_time).squeeze(-1)   # [B,H,N]

        if mask is not None:
            scores = scores.masked_fill(mask[:, :, None] == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=1)           # [B,H,N]

        # weighted sum across history
        h_pooled = torch.sum(attn_weights.unsqueeze(-1) * h_seq, dim=1)  # [B,N,D]
        out = self.out_proj(h_pooled)
        return out.squeeze(0), attn_weights

class TemporalTransformer(nn.Module):
    """
    Applies Transformer-style temporal self-attention over a history of node embeddings.
    Processes each node's time sequence independently.

    Input:  h_seq [H, N, D]
    Output: h_out [N, D]
    """
    def __init__(self, dim, n_heads=4, n_layers=2, time_dim=16, dropout=0.1):
        super().__init__()
        self.time_encoder = SinusoidalTimeEncoding(time_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim + time_dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True  # easier handling: (batch, seq, feature)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.proj = nn.Linear(dim + time_dim, dim)

    def forward(self, h_seq):
        """
        Args:
            h_seq: [H, N, D] (history length, nodes, feature)
        Returns:
            h_out: [N, D]
        """
        H, N, D = h_seq.shape
        device = h_seq.device
        timesteps = torch.arange(H, device=device)
        t_emb = self.time_encoder(timesteps).to(device)  # [H, time_dim]
        t_emb = t_emb.unsqueeze(1).expand(H, N, -1)      # [H, N, time_dim]

        # concat time embedding
        x = torch.cat([h_seq, t_emb], dim=-1)            # [H, N, D+time_dim]

        # transformer expects [batch, seq, dim]
        x = x.permute(1, 0, 2)                           # [N, H, D+time_dim]
        y = self.transformer(x)                          # [N, H, D+time_dim]

        # take last timestep (autoregressive target)
        h_out = self.proj(y[:, -1, :])                   # [N, D]
        return h_out
class EncodeProcessDecodeTemporalAttention(nn.Module):
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hidden_size=128,
                 process_steps=3,
                 time_dim = 16,
                 coarse_process_steps = 3,
                 node_out_dim=0, 
                 attention=False,
                 with_mat_params = True,
                 sample_ratio = 0.5,
                 max_history = 0,
                 device = "cpu"):
        super().__init__()

        # Temporal attention
        self.max_history = max_history
        # self.time_dim = time_dim
        # if self.max_history > 0 :
        #     self.node_temp_att_pooling = TemporalTransformer(dim=node_in_dim, n_heads=1, n_layers=2)
        #     self.edge_temp_att_pooling = TemporalTransformer(dim=edge_in_dim, n_heads=1, n_layers=1)
        # Encoders
        self.node_encoder = MLP(node_in_dim * (1 + max_history), hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim * (1 + max_history), hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        # self.coarse_edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.attention = attention
        self.with_mat_params = with_mat_params
        self.sample_ratio = sample_ratio
        # Processor: stack of message passing steps
        self.process_steps = process_steps
        self.processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(process_steps)])
        # self.coarse_processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
        #                                  for _ in range(coarse_process_steps)])

        # Decoder MLPs
        self.node_decoder = MLP(hidden_size, node_out_dim, hidden_dims=(hidden_size,), layer_norm=False) if node_out_dim > 0 else None

        self.device = device
        self.node_features_normalizer = Normalizer(node_in_dim, device)
        self.edge_features_normalizer = Normalizer(edge_in_dim, device)
        # self.coarse_edge_features_normalizer = Normalizer(edge_in_dim, device)
        self.output_normalizer = Normalizer(node_out_dim, device)
    def forward(self, graph_seq):
        # batch: [N] batch index if using batching, required for global readout
        g = graph_seq[-1]
        # Encode
        # Normalize x and edge features before input to encoder
        if self.max_history > 0:
            # 1️⃣ Normalize and collect feature history
            x_hist = [self.node_features_normalizer(self._build_node_features(g)) for g in graph_seq]
            e_hist = [self.edge_features_normalizer(self._build_edge_features(g)) for g in graph_seq]

            # 2️⃣ Pad with zeros if not enough history
            missing = (self.max_history + 1) - len(graph_seq)
            if missing > 0:
                x_pad = torch.zeros_like(x_hist[0])
                e_pad = torch.zeros_like(e_hist[0])
                for _ in range(missing):
                    x_hist.insert(0, x_pad)  # prepend oldest frames as zeros
                    e_hist.insert(0, e_pad)

            # 3️⃣ If more than max_history+1, truncate (keep most recent)
            x_hist = x_hist[-(self.max_history + 1):]
            e_hist = e_hist[-(self.max_history + 1):]

            # 4️⃣ Concatenate along feature dimension
            x = torch.cat(x_hist, dim=-1)
            e = torch.cat(e_hist, dim=-1)
        else :
            x = self.node_features_normalizer(self._build_node_features(g))
            e = self.edge_features_normalizer(self._build_edge_features(g)) 

        x_h = self.node_encoder(x)
        e_h = self.edge_encoder(e)

        for i, proc in enumerate(self.processors):
            x_h, e_h = proc(x_h, g.edge_index, e_h)

        # check = cx_h - check_x_h
        x_h = torch.cat([x_h], dim = -1)
        # Decode
        output = self.node_decoder(x_h)

        return output
    def _build_node_features(self, graph) :
        u = graph.world_pos - graph.mesh_pos
        phi = graph.phi
        swell_phi = graph.swelling_phi
        next_swell_phi = graph.next_swelling_phi
        rate_swell_phi = graph.rate_swelling_phi
        node_type = graph.node_type
        mat_param = graph.mat_param.unsqueeze(0).repeat(u.shape[0], 1)
        # mat_param = _apply_film
        if self.with_mat_params :
            x = torch.cat([u, phi, swell_phi, next_swell_phi, rate_swell_phi, node_type, mat_param], dim = -1)
        else :
            x = torch.cat([u, phi, swell_phi, next_swell_phi, rate_swell_phi, node_type], dim = -1)
        return x
    def _build_edge_features(self, graph) :
        senders, receivers = graph.edge_index[0], graph.edge_index[1]
        rel_position = graph.mesh_pos[senders, :] - graph.mesh_pos[receivers, :]
        distance = torch.norm(rel_position, dim=-1, keepdim=True)

        rel_world_pos = graph.world_pos[senders, :] - graph.world_pos[receivers, :]
        distance_world_pos  = torch.norm(rel_world_pos, dim=-1, keepdim=True)
        rel_phi = graph.phi[senders] - graph.phi[receivers]
        edge_features = torch.cat([rel_position, distance, rel_world_pos, distance_world_pos, rel_phi], dim=-1)
        return edge_features
    
    def _build_coarse_edge_features(self, graph):
        """
        Build coarse edges by sampling a subset of nodes with FPS (or random) instead of voxel sampling.
        sample_ratio: fraction of nodes to keep in coarse graph (0 < sample_ratio <= 1)
        """

        def farthest_point_sampling(points, num_samples):
            N = points.shape[0]
            device = points.device
            sampled_idx = torch.zeros(num_samples, dtype=torch.long, device=device)
            # pick first point randomly
            sampled_idx[0] = torch.randint(0, N, (1,), device=device)
            dist = torch.full((N,), float('inf'), device=device)
            for i in range(1, num_samples):
                last = points[sampled_idx[i-1]].unsqueeze(0)
                dist = torch.minimum(dist, torch.norm(points - last, dim=-1))
                sampled_idx[i] = torch.argmax(dist)
            return sampled_idx

        mesh_pos = graph.mesh_pos
        world_pos = graph.world_pos
        phi = graph.phi
        k = 4 if mesh_pos.shape[-1] == 3 else 3

        N_coarse = max(1, int(mesh_pos.shape[0] * self.sample_ratio))
        first_indices = farthest_point_sampling(mesh_pos, N_coarse)  # indices of coarse nodes in original

        # Sampled points in coarse graph
        sampled_points = mesh_pos[first_indices]

        # Build coarse KNN graph
        sampled_edge_index = knn_graph(sampled_points, k=k, loop=False)
        # Map edges back to original nodes
        sampled_edge_index_by_original = first_indices[sampled_edge_index]

        # Senders / receivers
        sampled_senders, sampled_receivers = sampled_edge_index_by_original[0], sampled_edge_index_by_original[1]

        # Edge features
        rel_position = mesh_pos[sampled_senders, :] - mesh_pos[sampled_receivers, :]
        distance = torch.norm(rel_position, dim=-1, keepdim=True)
        rel_world_pos = world_pos[sampled_senders, :] - world_pos[sampled_receivers, :]
        distance_world_pos  = torch.norm(rel_world_pos, dim=-1, keepdim=True)
        rel_phi = phi[sampled_senders] - phi[sampled_receivers]
        edge_features = torch.cat([rel_position, distance, rel_world_pos, distance_world_pos, rel_phi], dim=-1)

        return edge_features, sampled_edge_index, first_indices

    def loss(self, graph_seq):
        graph = graph_seq[-1]
        curr = torch.cat([graph.world_pos, graph.phi], dim = -1)
        check = graph.target
        target_delta = graph.target - curr
        target_delta_normalize = self.output_normalizer(target_delta)
        pred_delta = self.forward(graph_seq)
        check = torch.mean(pred_delta, dim = 0)
        error = (pred_delta - target_delta_normalize)**2
        # MSE loss on the change (delta) prediction
        # exclude nodes with dbc (node_type 1, 2, 3 in last feature)
        ux_dbc_nodes = graph.node_type[:, 1] == 1
        uy_dbc_nodes = graph.node_type[:, 2] == 1
        phi_dbc_nodes = graph.node_type[:, 3] == 1
        graph_error_ux = torch.mean(error[:, 0].masked_select(~ux_dbc_nodes))
        graph_error_uy = torch.mean(error[:, 1].masked_select(~uy_dbc_nodes))
        graph_error_phi = torch.mean(error[:, 2].masked_select(~phi_dbc_nodes))
        return graph_error_ux + graph_error_uy + graph_error_phi, graph_error_ux, graph_error_uy, graph_error_phi
    def predict(self, graph_seq):
        graph = graph_seq[-1]
        with torch.no_grad():
            pred_delta_normalized = self.forward(graph_seq)
            pred_delta = self.output_normalizer.inverse(pred_delta_normalized)
            curr = torch.cat([graph.world_pos, graph.phi], dim = -1)
            ux_dbc_nodes = graph.node_type[:, 1] == 1
            uy_dbc_nodes = graph.node_type[:, 2] == 1
            phi_dbc_nodes = graph.node_type[:, 3] == 1
            pred_delta[ux_dbc_nodes, 0] = 0.0 # no change on fixed nodes
            pred_delta[uy_dbc_nodes, 1] = 0.0 # no change on fixed nodes
            pred_delta[phi_dbc_nodes, 2] = 0.0 # no change on fixed nodes
            pred = curr + pred_delta
        output_graph = graph.clone()
        output_graph.world_pos = pred[:, :2].clone()
        output_graph.phi = pred[:, 2:].clone()
        return output_graph
    def save_model(self, path):
        torch.save(self.state_dict(), os.path.join(path, "model_weights.pth"))
        torch.save(self.output_normalizer, os.path.join(path, "output_normalizer.pth"))
        torch.save(self.node_features_normalizer, os.path.join(path, "node_features_normalizer.pth"))
        torch.save(self.edge_features_normalizer, os.path.join(path, "edge_features_normalizer.pth"))
        # torch.save(self.coarse_edge_features_normalizer, os.path.join(path, "coarse_edge_features_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self.output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self.node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self.edge_features_normalizer = torch.load(os.path.join(path, "edge_features_normalizer.pth"))
        # self.coarse_edge_features_normalizer = torch.load(os.path.join(path, "coarse_edge_features_normalizer.pth"))