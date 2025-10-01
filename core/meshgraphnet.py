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
from torch_geometric.nn import knn_graph
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
        new_node_features = new_node_features +node_feat
        new_edge_features = new_edge_features + edge_feat     
                
        return new_node_features, new_edge_features
    
    def message(self, x_i, x_j, edge_attr):
        # Compute attention score
        if self.attention :
            alpha = (self.attn_lin(x_i) * self.attn_lin(x_j)).sum(dim=-1, keepdim=True)
            alpha = F.leaky_relu(alpha)
            alpha = torch.softmax(alpha, dim=0)  # softmax over neighbors

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
            x = torch.cat([u, phi, swell_phi, swelling_phi_rate, node_type, time_emb, mat_param], dim = -1)
        else :
            x = torch.cat([u, phi, swell_phi, swelling_phi_rate, node_type, time_emb], dim = -1)
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
        u = graph.world_pos - graph.mesh_pos
        u_prev = graph.prev_world_pos - graph.mesh_pos
        phi = graph.phi
        prev_phi = graph.prev_phi
        swell_phi = graph.swelling_phi
        swelling_phi_rate = graph.swelling_phi_rate
        swelling_phi_rate_prev = graph.swelling_phi_rate_prev
        node_type = graph.node_type
        time_emb = time_emb.unsqueeze(0).repeat(u.shape[0], 1)
        mat_param = graph.mat_param.unsqueeze(0).repeat(u.shape[0], 1)
        if self.with_mat_params :
            x = torch.cat([u, u_prev, phi, prev_phi, swell_phi, swelling_phi_rate, swelling_phi_rate_prev, node_type, mat_param, time_emb], dim = -1)
        else :
            x = torch.cat([u, u_prev, phi, prev_phi, swell_phi, swelling_phi_rate, swelling_phi_rate_prev, node_type, time_emb], dim = -1)
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
        return graph_error_ux + graph_error_uy + graph_error_phi, graph_error_ux, graph_error_uy, graph_error_phi
    def predict(self, graph):
        with torch.no_grad():
            pred_delta_normalized = self.forward(graph)
            pred_delta = self.output_normalizer.inverse(pred_delta_normalized)
            curr = torch.cat([graph.world_pos, graph.phi], dim = -1)
            # prev = torch.cat([graph.prev_world_pos, graph.prev_phi], dim = -1)
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
                 time_dim = 16,
                 device = "cpu"):
        super().__init__()
        # Encoders
        self.node_encoder = MLP(node_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.coarse_edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.attention = attention
        self.with_mat_params = with_mat_params
        self.sample_ratio = sample_ratio
        self.time_dim = time_dim
        # Processor: stack of message passing steps
        self.process_steps = process_steps
        self.processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(process_steps)])
        self.coarse_processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(coarse_process_steps)])

        # Decoder MLPs
        self.node_decoder = MLP(2*hidden_size, node_out_dim, hidden_dims=(hidden_size,), layer_norm=False) if node_out_dim > 0 else None

        self.device = device
        self.node_features_normalizer = Normalizer(node_in_dim, device)
        self.edge_features_normalizer = Normalizer(edge_in_dim, device)
        self.coarse_edge_features_normalizer = Normalizer(edge_in_dim, device)
        self.output_normalizer = Normalizer(node_out_dim, device)
    def forward(self, batch):
        # x: [N, node_in_dim]
        # edge_attr: [E, edge_in_dim]
        # batch: [N] batch index if using batching, required for global readout

        # Encode
        # Normalize x and edge features before input to encoder
        x = self._build_node_features(batch)
        e = self._build_edge_features(batch)
        ce, c_edge_index = self._build_coarse_edge_features(batch)
        x_h = self.node_encoder(self.node_features_normalizer(x))
        e_h = self.edge_encoder(self.edge_features_normalizer(e))
        ce_h = self.coarse_edge_encoder(self.coarse_edge_features_normalizer(ce))

        # Process (multiple message-passing steps)
        cx_h = x_h.clone()
        for i, proc in enumerate(self.processors):
            x_h, e_h = proc(x_h, batch.edge_index, e_h)
        for i, proc in enumerate(self.coarse_processors) :
            cx_h, ce_h = proc(cx_h, c_edge_index, ce_h)

        # check = cx_h - check_x_h
        x_h = torch.cat([x_h, cx_h], dim = -1)
        # Decode
        output = self.node_decoder(x_h)

        return output
    def _build_node_features(self, graph) :
        time_emb = sinusoidal_time_embedding(graph.time, dim = self.time_dim)
        # time_emb = graph.time
        u = graph.world_pos - graph.mesh_pos
        phi = graph.phi
        swell_phi = graph.swelling_phi
        swelling_phi_rate = graph.swelling_phi_rate
        node_type = graph.node_type
        time_emb = time_emb.unsqueeze(0).repeat(u.shape[0], 1)
        physical_time = graph.time.unsqueeze(0).repeat(u.shape[0], 1)
        mat_param = graph.mat_param.unsqueeze(0).repeat(u.shape[0], 1)
        # mat_param = _apply_film
        if self.with_mat_params :
            x = torch.cat([u, phi, swell_phi, swelling_phi_rate, node_type, time_emb, physical_time, mat_param], dim = -1)
        else :
            x = torch.cat([u, phi, swell_phi, swelling_phi_rate, node_type, time_emb, physical_time], dim = -1)
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

        return edge_features, sampled_edge_index_by_original

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
        torch.save(self.coarse_edge_features_normalizer, os.path.join(path, "coarse_edge_features_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self.output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self.node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self.edge_features_normalizer = torch.load(os.path.join(path, "edge_features_normalizer.pth"))
        self.coarse_edge_features_normalizer = torch.load(os.path.join(path, "coarse_edge_features_normalizer.pth"))

class EncodeProcessDecodeMultiScaleHistory(nn.Module):
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
                 device = "cpu"):
        super().__init__()
        # Encoders
        self.node_encoder = MLP(node_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.coarse_edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.attention = attention
        self.with_mat_params = with_mat_params
        self.sample_ratio = sample_ratio
        # Processor: stack of message passing steps
        self.process_steps = process_steps
        self.processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(process_steps)])
        self.coarse_processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size, attention)
                                         for _ in range(coarse_process_steps)])

        # Decoder MLPs
        self.node_decoder = MLP(2*hidden_size, node_out_dim, hidden_dims=(hidden_size,), layer_norm=False) if node_out_dim > 0 else None

        self.device = device
        self.node_features_normalizer = Normalizer(node_in_dim, device)
        self.edge_features_normalizer = Normalizer(edge_in_dim, device)
        self.coarse_edge_features_normalizer = Normalizer(edge_in_dim, device)
        self.output_normalizer = Normalizer(node_out_dim, device)
    def forward(self, batch):
        # x: [N, node_in_dim]
        # edge_attr: [E, edge_in_dim]
        # batch: [N] batch index if using batching, required for global readout

        # Encode
        # Normalize x and edge features before input to encoder
        x = self._build_node_features(batch)
        e = self._build_edge_features(batch)
        ce, c_edge_index = self._build_coarse_edge_features(batch)
        x_h = self.node_encoder(self.node_features_normalizer(x))
        e_h = self.edge_encoder(self.edge_features_normalizer(e))
        ce_h = self.coarse_edge_encoder(self.coarse_edge_features_normalizer(ce))

        # Process (multiple message-passing steps)
        cx_h = x_h.clone()
        for i, proc in enumerate(self.processors):
            x_h, e_h = proc(x_h, batch.edge_index, e_h)
        for i, proc in enumerate(self.coarse_processors) :
            cx_h, ce_h = proc(cx_h, c_edge_index, ce_h)

        # check = cx_h - check_x_h
        x_h = torch.cat([x_h, cx_h], dim = -1)
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
            x = torch.cat([u, phi, swell_phi, swelling_phi_rate, node_type, time_emb, mat_param], dim = -1)
        else :
            x = torch.cat([u, phi, swell_phi, swelling_phi_rate, node_type, time_emb], dim = -1)
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

        return edge_features, sampled_edge_index_by_original

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
        torch.save(self.coarse_edge_features_normalizer, os.path.join(path, "coarse_edge_features_normalizer.pth"))

    def load_model(self, path):
        self.load_state_dict(torch.load(os.path.join(path, "model_weights.pth"), weights_only=True))
        self.output_normalizer = torch.load(os.path.join(path, "output_normalizer.pth"))
        self.node_features_normalizer = torch.load(os.path.join(path, "node_features_normalizer.pth"))
        self.edge_features_normalizer = torch.load(os.path.join(path, "edge_features_normalizer.pth"))
        self.coarse_edge_features_normalizer = torch.load(os.path.join(path, "coarse_edge_features_normalizer.pth"))