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
from torch_geometric.utils import softmax
import os
from torch_scatter import scatter_add, scatter_mean, scatter_max


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

class EdgeNodeMessagePassing(nn.Module):
    """
    Custom Message Passing layer using torch_scatter:

    1) Update edge attributes using (x_i, x_j, e_ij)
    2) Compute messages for neighbors
    3) Aggregate messages and update node attributes
    """

    def __init__(self, hidden_dim, attention=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attention = attention

        if self.attention:
            self.attn_lin = nn.Linear(hidden_dim, hidden_dim)


        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        # Edge update MLP: (x_i, x_j, e_ij) -> e'_ij
        # self.edge_mlp = MLP(hidden_dim * 2 + hidden_dim, hidden_dim,
        #                     hidden_dims=(hidden_dim,), layer_norm=True)
        # # Node update MLP: (x_i, aggregated_message) -> x'_i
        # self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim,
        #                     hidden_dims=(hidden_dim,), layer_norm=True)

    def forward(self, node_feat, edge_index, edge_feat):
        """
        node_feat: [N, hidden_dim]
        edge_index: [2, E]
        edge_feat: [E, hidden_dim]
        """
        row, col = edge_index  # row: source, col: target

        # --- Edge Update ---
        edge_input = torch.cat([node_feat[row], node_feat[col], edge_feat], dim=-1)
        new_edge_feat = self.edge_mlp(edge_input)
        # new_edge_feat = new_edge_feat + edge_feat  # residual

        # --- Message computation ---
        if self.attention:
            alpha = (self.attn_lin(node_feat[row]) * self.attn_lin(node_feat[col])).sum(dim=-1)
            alpha = F.leaky_relu(alpha)
            # normalize per target node
            alpha = scatter_add(alpha, col, dim=0, dim_size=node_feat.size(0))
            # avoid divide by zero
            alpha[col] = alpha[col].clamp(min=1e-6)
            alpha = (F.leaky_relu((self.attn_lin(node_feat[row]) * self.attn_lin(node_feat[col])).sum(dim=-1)) / alpha[col]).unsqueeze(-1)
            msg = new_edge_feat * alpha
        else:
            msg = new_edge_feat  # [E, hidden_dim]

        # --- Aggregate messages to nodes ---
        aggr_out = scatter_add(msg, col, dim=0, dim_size=node_feat.size(0))

        # --- Node update ---
        node_input = torch.cat([node_feat, aggr_out], dim=-1)
        new_node_feat = self.node_mlp(node_input)
        new_node_feat = new_node_feat + node_feat  # residual

        return new_node_feat, new_edge_feat + edge_feat


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
        phi = graph.phi
        phi_delta = graph.phi - graph.prev_phi
        prev_phi = graph.prev_phi
        swell_phi = graph.swelling_phi
        swelling_phi_rate = graph.swelling_phi_rate
        swelling_phi_rate_prev = graph.swelling_phi_rate_prev
        node_type = graph.node_type
        mat_param = graph.mat_param.unsqueeze(0).repeat(u.shape[0], 1)
        if self.with_mat_params :
            x = torch.cat([phi, phi_delta, swell_phi, swelling_phi_rate, swelling_phi_rate_prev, node_type, mat_param], dim = -1)
        else :
            x = torch.cat([phi, phi_delta, swell_phi, swelling_phi_rate, swelling_phi_rate_prev, node_type], dim = -1)
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
        # curr = graph.world_pos
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
            # curr = graph.world_pos
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
        self.node_decoder = MLP(hidden_size, node_out_dim * time_dim, hidden_dims=(hidden_size,), layer_norm=False)
        # self.node_decoder = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size//2, 1),
        #     Swish(),
        #     nn.Conv1d(hidden_size//2, node_out_dim * time_dim, 1)
        # )
        self.device = device
        self.node_features_normalizer = Normalizer(1, node_in_dim, "node_features_normalizer", device)
        self.edge_features_normalizer = Normalizer(1, edge_in_dim, "edge_features_normalizer", device)
        self.output_normalizer = Normalizer(time_dim, node_out_dim, "output_normalizer", device)
    def forward(self, batch):
        # x: [N, node_in_dim]
        # edge_attr: [E, edge_in_dim]
        # batch: [N] batch index if using batching, required for global readout

        # Encode
        # Normalize x and edge features before input to encoder
        x = self._build_node_features(batch).unsqueeze(0)
        e = self._build_edge_features(batch).unsqueeze(0)
        x_h = self.node_encoder(self.node_features_normalizer(x)).squeeze(0)
        e_h = self.edge_encoder(self.edge_features_normalizer(e)).squeeze(0)
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
        # x_h = x_h.unsqueeze(0).permute(0, 2, 1)  
        # decoded = self.node_decoder(x_h)
        # dt = torch.arange(1, self.time_dim + 1).to(device=self.device).unsqueeze(-1).unsqueeze(-1)
        # decoded = decoded.squeeze(0).transpose(0, 1)  # (num_nodes, time_dim * node_out_dim)
        # delta = decoded.view(-1, self.time_dim, self.node_out_dim).permute(1, 0, 2)


        decoded = self.node_decoder(x_h)  # [num_nodes, node_out_dim * time_dim]

        # Reshape to [time_dim, num_nodes, node_out_dim]
        delta = decoded.view(-1, self.time_dim, self.node_out_dim).permute(1, 0, 2)

        # Optional: multiply by time step index if needed
        dt = torch.arange(1, self.time_dim + 1, device=self.device).view(self.time_dim, 1, 1)
        return delta * dt
    def _build_node_features(self, graph) :
        # time_emb = sinusoidal_time_embedding(graph["time"], dim = self.time_dim)
        u = graph["world_pos"] - graph["mesh_pos"]
        phi = graph["phi"]
        swell_phi = graph["swelling_phi"]
        node_type = graph["node_type"]
        # time_emb = time_emb.unsqueeze(0).repeat(u.shape[0], 1)
        mat_param = graph["mat_param"].unsqueeze(0).repeat(u.shape[0], 1)
        # mat_param = _apply_film
        if self.with_mat_params :
            x = torch.cat([u, phi, swell_phi, node_type, mat_param], dim = -1)
        else :
            x = torch.cat([u, phi, swell_phi, node_type], dim = -1)
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

        target_delta = (target - curr)
        target_delta_normalize = self.output_normalizer(target_delta)
        pred_delta = self.forward(graph)

        error = (pred_delta - target_delta_normalize)**2
        # MSE loss on the change (delta) prediction
        # exclude nodes with dbc (node_type 1, 2, 3 in last feature)
        ux_dbc_nodes = graph["node_type"][:, 1] == 1
        uy_dbc_nodes = graph["node_type"][:, 2] == 1
        phi_dbc_nodes = graph["node_type"][:, 3] == 1

        graph_error_ux = torch.mean(torch.sum(torch.sum(error[:, ~ux_dbc_nodes, 0:1], dim=2), dim=1))
        graph_error_uy = torch.mean(torch.sum(torch.sum(error[:, ~uy_dbc_nodes, 1:2], dim=2), dim=1))
        graph_error_phi = torch.mean(torch.sum(torch.sum(error[:, ~phi_dbc_nodes, 2:], dim=2), dim=1))

        return graph_error_ux + graph_error_uy + graph_error_phi, graph_error_ux, graph_error_uy, graph_error_phi
    def predict(self, graph):
        with torch.no_grad():
            pred_delta_normalized = self.forward(graph)
            pred_delta = self.output_normalizer.inverse(pred_delta_normalized)
            curr = torch.cat([graph["world_pos"], graph["phi"]], dim = -1)
            ux_dbc_nodes = graph["node_type"][:, 1] == 1
            uy_dbc_nodes = graph["node_type"][:, 2] == 1
            phi_dbc_nodes = graph["node_type"][:, 3] == 1
            pred_delta[:, ux_dbc_nodes, 0] = 0.0 # no change on fixed nodes
            pred_delta[:, uy_dbc_nodes, 1] = 0.0 # no change on fixed nodes
            pred_delta[:, phi_dbc_nodes, 2] = 0.0 # no change on fixed nodes
            pred = curr.unsqueeze(0) + pred_delta
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