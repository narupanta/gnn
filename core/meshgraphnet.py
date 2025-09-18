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
                 device = "cpu"):
        super().__init__()
        # Encoders
        self.node_encoder = MLP(node_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.attention = attention
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
        phi = graph.phi
        swell_phi = graph.swelling_phi
        swelling_phi_rate = graph.swelling_phi_rate
        node_type = graph.node_type
        time_emb = time_emb.unsqueeze(0).repeat(u.shape[0], 1)
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