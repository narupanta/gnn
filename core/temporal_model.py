import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .normalization import Normalizer
import os

class TemporalWrapper(nn.Module):
    def __init__(self, base_model, latent_dim=128, rnn_hidden=128, rnn_type='RNN'):
        """
        base_model: your EncodeProcessDecode instance
        latent_dim: dimension of node embeddings from base_model
        rnn_hidden: hidden size for temporal module
        rnn_type: 'RNN', 'GRU', or 'LSTM'
        """
        super().__init__()
        self.base_model = base_model

        if rnn_type == 'RNN':
            self.temporal_rnn = nn.RNN(latent_dim, rnn_hidden, batch_first=True)
        elif rnn_type == 'GRU':
            self.temporal_rnn = nn.GRU(latent_dim, rnn_hidden, batch_first=True)
        elif rnn_type == 'LSTM':
            self.temporal_rnn = nn.LSTM(latent_dim, rnn_hidden, batch_first=True)
        else:
            raise ValueError("Invalid rnn_type")

        # Node decoder: output Δpos/Δphi
        self.decoder = nn.Linear(rnn_hidden, base_model.node_decoder[-1].out_features)

    def forward(self, graph_seq):
        """
        graph_seq: list of graphs of length T (batch=1)
        """
        node_embeddings_seq = []
        for graph in graph_seq:
            # Run your base MGN model up to node embeddings (before decoding)
            x_h = self.base_model.node_encoder(self.base_model.node_features_normalizer(self.base_model._build_node_features(graph)))
            e_h = self.base_model.edge_encoder(self.base_model.edge_features_normalizer(self.base_model._build_edge_features(graph)))
            for proc in self.base_model.processors:
                x_h, e_h = proc(x_h, graph.edge_index, e_h)
            node_embeddings_seq.append(x_h)  # [N, latent]

        # Stack along time: [N, T, latent]
        node_embeddings_seq = torch.stack(node_embeddings_seq, dim=1)

        # Per-node RNN: treat nodes as batch dimension
        out, _ = self.temporal_rnn(node_embeddings_seq)  # [N, T, rnn_hidden]

        # Decode Δpos per node
        delta_pred = self.decoder(out)  # [N, T, node_out_dim]
        return delta_pred
