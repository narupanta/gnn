import torch
from copy import deepcopy
# from torch_geometric.data import Data
from tqdm import tqdm
# from copy import deepcopy
from torch_geometric.data import Data
def rollout(model, data):
    device = model.device
    rollout_preds = []
    time_dim = model.time_dim
    curr_graph = deepcopy(data[0]) # deep copy ensures clean start
    rollout_preds = [torch.cat([data[0].world_pos, data[0].phi], dim = -1).unsqueeze(0).to(model.device)]
    with torch.no_grad():
        loop = tqdm(range(0, len(data), time_dim), desc="Rollout")

        for t in loop:
            # Replace only time-dependent fields for this step
            # curr_graph.swelling_phi = data[t].swelling_phi
            curr_graph.swelling_phi = torch.full_like(data[t].swelling_phi, 0.65)
            swelling_phi = []
            for w in range(time_dim + 1):
                idx = t + w
                if idx < len(data):
                    # swelling_phi.append(data[idx].swelling_phi)
                    swelling_phi.append(torch.full_like(data[t].swelling_phi, 0.65))
                else:
                    # pad with zeros of the same shape
                    swelling_phi.append(torch.zeros_like(data[0].swelling_phi))

            curr_graph.swelling_phi = torch.stack(swelling_phi).T
            # Predict next state
            pred = model.predict(curr_graph.to(device))
            last_step = pred[-1].clone()
            # Update for next step
            curr_graph.world_pos = last_step[:, :2]
            curr_graph.phi = last_step[:, 2:]

            rollout_preds.append(pred)

        rollout_preds = torch.cat(rollout_preds, dim = 0)[:len(data)]
        rollout_gts = torch.stack([torch.cat([data[t].world_pos, data[t].phi], dim = -1) for t in range(0, len(data))])
        swelling_phi = torch.stack([data[t].swelling_phi for t in range(0, len(data))])
        # Compute error
        error = (rollout_preds.to(device) - rollout_gts.to(device)) ** 2
        rmse_x = torch.sqrt(torch.mean(error[:, :, 0]))
        rmse_y = torch.sqrt(torch.mean(error[:, :, 1]))
        rmse_phi = torch.sqrt(torch.mean(error[:, :, 2]))

        print(f"RMSE Ux: {rmse_x:.6f}, Uy: {rmse_y:.6f}, Phi: {rmse_phi:.6f}")

    return {
        "time": torch.tensor([data[t]["time"] for t in range(len(data))]),
        "pred": rollout_preds,
        "gt": rollout_gts,
        "swelling_phi": swelling_phi,
        "mat_param": data[0]["mat_param"],
        "mesh_pos": data[0]["mesh_pos"],
        "cells": data[0]["cells"],
        "node_type": data[0]["node_type"],
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_phi": rmse_phi,
    }

# def rollout(model, graphs):
#     device = model.device

#     # Split trajectory into frames

#     rollout_preds = []
#     load = []
#     load_rate = []
#     curr_graph = deepcopy(graphs[0]) # deep copy ensures clean start

#     with torch.no_grad():
#         loop = tqdm(range(len(graphs)), desc="Rollout")

#         for t in loop:
#             # Replace only time-dependent fields for this step
#             curr_graph["time"] = graphs[t]["time"].clone()
#             curr_graph["swelling_phi"] = graphs[t]["swelling_phi"].clone()
#             # if t % 20 == 0 :
#             #     curr_graph = graphs[t]
#             load.append(curr_graph["swelling_phi"])
#             # curr_graph = graphs[0]
#             # Predict next state
#             pred_next = model.predict(curr_graph)

#             # Update for next step
#             curr_graph["world_pos"] = pred_next[0, :, :2].clone()
#             curr_graph["phi"] = pred_next[0, :, 2:].clone()

#             rollout_preds.append(pred_next[0].detach().cpu())

#         rollout_preds = torch.stack(rollout_preds, axis=0)  # [T, N, 3]
#         load = torch.stack(load, axis = 0)
#         load_rate = torch.stack(load_rate, axis = 0)
#         # Compute error
#         gt = torch.stack([target[0].to(device) for target in data["target"]])
#         error = (rollout_preds.to(device) - gt) ** 2
#         rmse_x = torch.sqrt(torch.mean(error[:, :, 0]))
#         rmse_y = torch.sqrt(torch.mean(error[:, :, 1]))
#         rmse_phi = torch.sqrt(torch.mean(error[:, :, 2]))

#         print(f"RMSE Ux: {rmse_x:.6f}, Uy: {rmse_y:.6f}, Phi: {rmse_phi:.6f}")

#     return {
#         "time": data["time"],
#         "pred": rollout_preds,
#         "gt": gt,
#         "mesh_pos": data["mesh_pos"],
#         "cells": data["cells"],
#         "mat_param": data["mat_param"],
#         "node_type": data["node_type"],
#         "swell_phi": load,
#         "swell_phi_rate": load_rate,
#         "rmse_x": rmse_x,
#         "rmse_y": rmse_y,
#         "rmse_phi": rmse_phi,
#     }

@torch.no_grad()
def rollout_history(model, data) :
    device = model.device
    # split trajectory into frames
    graphs = []
    for t in range(data.world_pos.shape[0]):
        graph = Data(world_pos=data.world_pos[t], prev_world_pos=data.prev_world_pos[t], 
                     phi = data.phi[t], prev_phi = data.prev_phi[t], 
                     node_type = data.node_type, target=data.target[t], edge_index=data.edge_index, mesh_pos=data.mesh_pos, time=data.time[t], 
                     swelling_phi=data.swelling_phi[t], 
                     swelling_phi_rate = data.swelling_phi_rate[t], swelling_phi_rate_prev = data.swelling_phi_rate_prev[t], 
                     cells=data.cells, mat_param = data.mat_param)
        graphs.append(graph.to(device))
        # run rollout
    rollout_preds = []
    curr_graph = graphs[0].to(device)
    with torch.no_grad():
        loop = tqdm(range(len(graphs)), desc ="Rollout")
        for t in loop:
            # if t % 50 == 0:
            #     print(f" Rollout step {t}/{len(graphs)}")
            #     # correct curr_graph to ground truth to avoid drift
            #     curr_graph.prev_world_pos = graphs[t].prev_world_pos.to(device)
            #     curr_graph.world_pos = graphs[t].world_pos.to(device)
            #     curr_graph.prev_phi = graphs[t].prev_phi.to(device)
            #     curr_graph.phi = graphs[t].phi.to(device)
                # predict next state
            curr_graph.time = graphs[t].time
            curr_graph.swelling_phi = graphs[t].swelling_phi.clone()
            curr_graph.swelling_phi_rate = graphs[t].swelling_phi_rate.clone()
            curr_graph.swelling_phi_rate_prev = graphs[t].swelling_phi_rate_prev.clone()
            # curr_graph.swelling_phi = torch.full_like(graphs[0].swelling_phi, 0.0)
            # curr_graph.swelling_phi_rate = torch.full_like(graphs[0].swelling_phi, 0.0)
            # curr_graph.swelling_phi_rate_prev = torch.full_like(graphs[0].swelling_phi, 0.0)
            pred_next = model.predict(curr_graph)
            # curr_graph.prev_world_pos = curr_graph.world_pos.clone()
            # curr_graph.world_pos = pred_next[:, :2].clone()
            curr_graph.prev_world_pos, curr_graph.prev_phi = curr_graph.world_pos.clone(), curr_graph.phi.clone()
            curr_graph.world_pos, curr_graph.phi = pred_next[:, :2].clone(), pred_next[:, 2:].clone()
            rollout_preds.append(pred_next)

        rollout_preds = torch.stack(rollout_preds, axis=0) # T, N, 3
        #compute error with respect to ground truth
        gt = data.target.to(device) # T, N, 3
        error = (rollout_preds - gt)**2 # T, N, 3
        rmse_x = torch.sqrt(torch.mean(error[:, :, 0]))
        rmse_y = torch.sqrt(torch.mean(error[:, :, 1]))
        rmse_phi = torch.sqrt(torch.mean(error[:, :, 2]))
        print(f"RMSE Ux: {rmse_x:.6f}, Uy: {rmse_y:.6f}, Phi: {rmse_phi:.6f}")
    return {"time" : data.time,
            "pred" : rollout_preds,
            "gt" : gt,
            "mesh_pos" : data.mesh_pos,
            "cells": data.cells,
            "node_type": data.node_type,
            "swell_phi": data.swelling_phi,
            "swell_phi_rate": data.swelling_phi_rate,
            "swelling_phi_rate_prev": data.swelling_phi_rate_prev,
            "rmse_x": rmse_x,
            "rmse_y": rmse_y,
            "rmse_phi": rmse_phi}  

import torch
from tqdm import tqdm

def rollout_temporal_att(model, data):
    """
    Autoregressive rollout for temporal GNN model.
    The model already handles padding for missing history steps.
    Uses up to model.max_history + 1 past graphs as input.
    """
    device = model.device
    graphs = data[0]  # list of ground-truth graphs [G0, G1, ..., GT]
    T = len(graphs)

    gt_graph_seq = [g.to(device).clone() for g in graphs]
    rollout_preds = [gt_graph_seq[0].clone()]  # start from initial GT graph

    with torch.no_grad():
        loop = tqdm(range(1, T), desc="Rollout", total=T - 1)
        for t in loop:
            # Build sequence of past predictions (model handles padding)
            start_idx = max(0, t - model.max_history)
            input_seq = rollout_preds[start_idx:t]

            # Always use the next step’s swelling_phi as control input
            swelling_phi = gt_graph_seq[t - 1].swelling_phi.clone()
            next_swelling_phi = gt_graph_seq[t - 1].next_swelling_phi.clone()
            rate_swelling_phi = gt_graph_seq[t - 1].rate_swelling_phi.clone()
            input_seq[-1].swelling_phi = swelling_phi
            input_seq[-1].next_swelling_phi = next_swelling_phi
            input_seq[-1].rate_swelling_phi = rate_swelling_phi

            # Predict next graph
            pred_graph = model.predict(input_seq)
            pred_graph.time = gt_graph_seq[t].time
            pred_graph.swelling_phi = swelling_phi
            pred_graph.next_swelling_phi = next_swelling_phi
            pred_graph.rate_swelling_phi = rate_swelling_phi

            rollout_preds.append(pred_graph.clone())

        # Compute rollout error vs GT
        pred = torch.stack(
            [torch.cat([g.world_pos, g.phi], dim=-1) for g in rollout_preds],
            dim=0,
        ).to(device)
        gt = torch.stack([g.target for g in gt_graph_seq], dim=0).to(device)

        error = (pred - gt) ** 2
        rmse_x = torch.sqrt(torch.mean(error[:, :, 0]))
        rmse_y = torch.sqrt(torch.mean(error[:, :, 1]))
        rmse_phi = torch.sqrt(torch.mean(error[:, :, 2]))

        print(f"RMSE → Ux: {rmse_x:.6f}, Uy: {rmse_y:.6f}, Phi: {rmse_phi:.6f}")

    return {
        "time": torch.tensor([g.time for g in gt_graph_seq], device=device),
        "pred": pred,
        "gt": gt,
        "mesh_pos": gt_graph_seq[0].mesh_pos,
        "cells": gt_graph_seq[0].cells,
        "mat_param": gt_graph_seq[0].mat_param,
        "node_type": gt_graph_seq[0].node_type,
        "swell_phi": torch.stack([g.swelling_phi for g in gt_graph_seq], dim=0),
        "rmse_x": rmse_x,
        "rmse_y": rmse_y,
        "rmse_phi": rmse_phi,
    }


@torch.no_grad()
def rollout_with_multistep_corrector(one_step_model, corrector, x0, edge_index, edge_attr, steps, history_T=4, device="cpu"):
    """
    x0: initial node states [N, state_dim]
    history_T: number of previous deltas used for correction
    """
    N, state_dim = x0.shape
    x_curr = x0.clone().to(device)
    predicted_states = [x_curr.cpu()]
    
    # Keep a rolling buffer of previous predicted deltas
    delta_buffer = []

    for t in range(steps):
        # 1️⃣ One-step delta prediction
        delta_pred = one_step_model(x_curr, edge_index, edge_attr)  # [N, state_dim]
        delta_buffer.append(delta_pred)

        # 2️⃣ Keep only last history_T deltas
        if len(delta_buffer) > history_T:
            delta_buffer.pop(0)

        # 3️⃣ Prepare input sequence for corrector
        input_seq = torch.stack(delta_buffer)  # [T, N, state_dim]

        # 4️⃣ Correct residual
        residual = corrector(input_seq)         # [T, N, state_dim]
        delta_corrected = delta_pred + residual[-1]  # use last timestep's correction

        # 5️⃣ Update state
        x_next = x_curr + delta_corrected
        predicted_states.append(x_next.cpu())
        x_curr = x_next

    return predicted_states  # list of [N, state_dim]
