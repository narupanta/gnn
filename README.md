# Surrogate Modeling of Hydrogel Diffusion-Deformation Using MeshGraphNets

This repository outlines a procedure for developing a data-driven surrogate model for the time-dependent, nonlinear, and coupled diffusion-deformation problem in hydrogels. We bridge the gap between traditional physics-based modeling and modern machine learning by explaining how to apply a Graph Neural Network (GNN) architecture, specifically the *MeshGraphNets* approach, to a hydrogel model.

The generated groundtruth dataset is based on the analysis of **Constitutive Model IV** as detailed in the review by Urrea-Quintero et al. (2024), providing a step-by-step guide for its implementation.

## Key Features
* MeshGraphNets implementation on mesh data regarding Coupled diffusion-deformation problem.

## Repository Structure

* `dataset/` – Example meshes and simulation datasets
* `selected_model/` – trained model on uniaxial and bending cases
* `core/` – The folder contains model architecture, rollout, data preprocessing and utilities codes
* `notebooks/` – Tutorials and step-by-step implementation

## Example of Results

![til](https://github.com/narupanta/hydrogel_gnn/blob/main/hydrogel_bend4cycles.gif)


## Citation

* J.-H. Urrea-Quintero, M. Marino, T. Wick, and U. Nackenhorst, "A Comparative Analysis of Transient Finite-Strain Coupled Diffusion-Deformation Theories for Hydrogels," *Archives of Computational Methods in Engineering*, vol. 31, pp. 3767--3800, 2024.
* T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, and P. W. Battaglia, "Learning Mesh-Based Simulation with Graph Networks," in *International Conference on Learning Representations (ICLR)*, 2021.
