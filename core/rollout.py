import torch
from copy import deepcopy
from tqdm import tqdm

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
            curr_graph.swelling_phi = data[t].swelling_phi
            swelling_phi = []
            for w in range(time_dim + 1):
                idx = t + w
                if idx < len(data):
                    swelling_phi.append(data[idx].swelling_phi)
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