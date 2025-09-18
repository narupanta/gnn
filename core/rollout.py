import torch

from torch_geometric.data import Data
from tqdm import tqdm
def rollout(model, data) :
    device = model.device
    # split trajectory into frames
    graphs = []
    for t in range(data.world_pos.shape[0]):
        graph = Data(world_pos=data.world_pos[t], phi = data.phi[t], node_type = data.node_type, target=data.target[t], edge_index=data.edge_index, mesh_pos=data.mesh_pos, time=data.time[t], swelling_phi=data.swelling_phi[t], swelling_phi_rate = data.swelling_phi_rate[t], cells=data.cells)
        graphs.append(graph.to(device))
        # run rollout
    rollout_preds = []
    curr_graph = graphs[0].to(device)
    with torch.no_grad():
        loop = tqdm(range(len(graphs)), desc ="Rollout")
        for t in loop:
            # if t % 1 == 0:
            #     print(f" Rollout step {t}/{len(graphs)}")
            #     # correct curr_graph to ground truth to avoid drift
            #     curr_graph = graphs[t-1].to(device)
            #     # predict next state
            curr_graph.time = graphs[t].time
            curr_graph.swelling_phi = graphs[t].swelling_phi
            curr_graph.swelling_phi_rate = graphs[t].swelling_phi_rate
            pred_next = model.predict(curr_graph)
            curr_graph.world_pos, curr_graph.phi = pred_next[:, :2], pred_next[:, 2:]
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
            "rmse_x": rmse_x,
            "rmse_y": rmse_y,
            "rmse_phi": rmse_phi}  