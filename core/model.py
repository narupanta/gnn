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
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data, Batch


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

    def __init__(self, hidden_dim):
        super().__init__(aggr='add')  # 'add' aggregation
        # edge update: (x_i, x_j, e_ij) -> e'_ij
        self.edge_mlp = MLP(hidden_dim * 2 + hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), layer_norm=True)
        # message: (x_j, e'_ij) -> message
        self.msg_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), layer_norm=True)
        # node update: (x_i, aggregated_message) -> x'_i
        self.node_mlp = MLP(hidden_dim + hidden_dim, hidden_dim, hidden_dims=(hidden_dim,), layer_norm=True)

    def forward(self, x, edge_index, edge_attr):
        # x: [N, node_dim]
        # edge_attr: [E, edge_dim]
        # First, compute updated edge attributes
        row, col = edge_index  # sender (col), receiver (row)
        x_i = x[row]
        x_j = x[col]
        edge_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        edge_attr_updated = self.edge_mlp(edge_input)

        # Then run message passing where message uses x_j and updated edge attr
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr_updated)
        edge_attr_updated += edge_attr  # residual connection
        # out is aggregated message for each node
        node_input = torch.cat([x, out], dim=-1)
        x_updated = self.node_mlp(node_input) + x

        return x_updated, edge_attr_updated

    def message(self, x_j, edge_attr):
        # x_j: sender node features
        # edge_attr: updated edge features for that edge
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        msg = self.msg_mlp(msg_input)
        return msg


class EncodeProcessDecode(nn.Module):
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 hidden_size=128,
                 process_steps=3,
                 node_out_dim=0):
        super().__init__()
        # Encoders
        self.node_encoder = MLP(node_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)
        self.edge_encoder = MLP(edge_in_dim, hidden_size, hidden_dims=(hidden_size,), layer_norm=True)

        # Processor: stack of message passing steps
        self.process_steps = process_steps
        self.processors = nn.ModuleList([EdgeNodeMessagePassing(hidden_size)
                                         for _ in range(process_steps)])

        # Decoder MLPs
        self.node_decoder = MLP(hidden_size, node_out_dim, hidden_dims=(hidden_size,), layer_norm=False) if node_out_dim > 0 else None

    def forward(self, batch):
        # x: [N, node_in_dim]
        # edge_attr: [E, edge_in_dim]
        # batch: [N] batch index if using batching, required for global readout

        # Encode
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        swelling_phi = batch.swelling_phi
        load_input = torch.zeros(x.shape[0], 1, device=x.device)

        # apply load only to designated nodes (node_type == 4)
        load_nodes = x[:, 7] == 1
        load_input[load_nodes, 0] = swelling_phi
        x = torch.cat([x, load_input], dim=-1)
        x_h = self.node_encoder(x)
        e_h = self.edge_encoder(edge_attr)

        # Process (multiple message-passing steps)
        for i, proc in enumerate(self.processors):
            x_h, e_h = proc(x_h, edge_index, e_h)

        # Decode
        output = self.node_decoder(x_h)

        return output
    def loss(self, graph):
        target = graph.y
        curr = graph.x[:, :3]
        target_delta = target - curr
        pred_delta = self.forward(graph)

        error = torch.abs(pred_delta - target_delta)
        # MSE loss on the change (delta) prediction
        # exclude nodes with dbc (node_type 1, 2, 3 in last feature)
        ux_dbc_nodes = graph.x[:, 4] == 1
        uy_dbc_nodes = graph.x[:, 5] == 1
        phi_dbc_nodes = graph.x[:, 6] == 1
        graph_error_ux = torch.mean(error[:, 0].masked_select(~ux_dbc_nodes))
        graph_error_uy = torch.mean(error[:, 1].masked_select(~uy_dbc_nodes))
        graph_error_phi = torch.mean(error[:, 2].masked_select(~phi_dbc_nodes))
        return graph_error_ux + graph_error_uy + graph_error_phi, graph_error_ux, graph_error_uy, graph_error_phi
    def predict(self, graph):
        with torch.no_grad():
            pred_delta = self.forward(graph)
            curr = graph.x[:, :3]
            ux_dbc_nodes = graph.x[:, 4] == 1
            uy_dbc_nodes = graph.x[:, 5] == 1
            phi_dbc_nodes = graph.x[:, 6] == 1
            pred_delta[:, 0][ux_dbc_nodes] = 0.0 # no change on fixed nodes
            pred_delta[:, 1][uy_dbc_nodes] = 0.0 # no change on fixed nodes
            pred_delta[:, 2][phi_dbc_nodes] = 0.0 # no change on fixed nodes
            pred = curr + pred_delta
        return pred

# ---------------------- Example usage ----------------------
if __name__ == '__main__':

    from datasetclass import HydrogelDataset
    from torch_geometric.loader import DataLoader

    dataset = HydrogelDataset(data_dir = "/mnt/c/Users/narun/Desktop/Project/hydrogel/gnn/dataset/bending_signal_/")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # split trajectory into frames
    all_graphs = []
    for data in dataloader:
        for t in range(data.x.shape[0]):
            graph = Data(x=data.x[t], y=data.y[t], edge_index=data.edge_index, edge_attr=data.edge_attr[t], pos=data.pos, time=data.time[t], swelling_phi=data.swelling_phi[t], cells=data.cells)
            all_graphs.append(graph)
    percentage_train = 0.8
    train_loader = DataLoader(all_graphs[:int(percentage_train*len(all_graphs))], batch_size=1, shuffle=True)
    val_loader = DataLoader(all_graphs[int(percentage_train*len(all_graphs)):], batch_size=1, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncodeProcessDecode(node_in_dim=9,
                                edge_in_dim=7,
                                hidden_size=64,
                                process_steps=10,
                                node_out_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for train_epoch in range(10):
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
        #print val loss and component loss
        print(f"Epoch {train_epoch}, Val Loss: {avg_val_loss:.6f}, Ux Loss: {avg_val_loss_ux:.6f}, Uy Loss: {avg_val_loss_uy:.6f}, Phi Loss: {avg_val_loss_phi:.6f}")

