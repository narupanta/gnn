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


## Example of training setups
Below picture is the example of domains and their boundary conditions. Case 1 is the case in which the hydrogel is freely deformed with the constant fluid environment $\phi_{env}$. Then, Case 2 is when hydrogel is under actuation of controlled fluid environment $\phi_{env}$. Finallt, Case 3 is when hydrogel gets exposed at the top boundary with controlled fluid environment $\phi_{env}$ which causes hydrogel to bend.
![alt text](https://github.com/narupanta/hydrogel_gnn/blob/main/figures/domains.png)

## Example of Results

The result of testing the model trained using dataset 1 cycle of the periodic input signal, which its environment swelling function is defined as:

$$\phi_{env}(t) = \frac{\phi_{max} + \phi_{min}}{2}+ \frac{\phi_{max} - \phi_{min}}{2}\tanh\left( A \cos\left( \frac{2\pi t}{T} \right) \right)$$

against dataset of 4 cycle of the periodic input signal.

![til](https://github.com/narupanta/hydrogel_gnn/blob/main/figures/hydrogel_bend4cycles.gif)
![alt text](https://github.com/narupanta/hydrogel_gnn/blob/main/figures/overall_error.png)
![alt text](https://github.com/narupanta/hydrogel_gnn/blob/main/figures/xy_topright_corner_timeseries.png)

## Limitation
Eventhough the model is able to capture the deformation and diffusion of the hydrogel. It is still not able to capture high speed deformation (high sharpness of the periodic function) and need high amount of data for bending cases to capture possible deformation. Apart from this, the model deteriorates after rolling out which can cause problems in a long simulation generation.
## Citation

* J.-H. Urrea-Quintero, M. Marino, T. Wick, and U. Nackenhorst, "A Comparative Analysis of Transient Finite-Strain Coupled Diffusion-Deformation Theories for Hydrogels," *Archives of Computational Methods in Engineering*, vol. 31, pp. 3767--3800, 2024.
* T. Pfaff, M. Fortunato, A. Sanchez-Gonzalez, and P. W. Battaglia, "Learning Mesh-Based Simulation with Graph Networks," in *International Conference on Learning Representations (ICLR)*, 2021.
