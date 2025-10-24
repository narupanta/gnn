from dolfin import *
from mshr import *
import numpy as np
import matplotlib.pyplot as plt
from dolfin import XDMFFile

def build_node_type_from_config(mesh, boundaries, bc_config):
    """
    Build one-hot node_type array from bc_config and boundaries MeshFunction.

    Columns:
        0 = interior
        1 = ux = 0
        2 = uy = 0
        3 = phi = 0
        4 = swell
    """
    num_nodes = mesh.num_vertices()
    node_type = np.zeros((num_nodes, 5), dtype=int)

    # Map boundary names to marker numbers
    boundary_map = {
        "left": 1,
        "right": 2,
        "top": 3,
        "bottom": 4
    }

    # Track which nodes are on any BC
    bc_nodes = set()

    # Assign columns for ux, uy, phi
    for comp, key in enumerate(["ux", "uy", "phi"], start=1):  # start=1 because column 0 is interior
        for side in bc_config.get(key, []):
            marker = boundary_map[side]
            for facet in facets(mesh):
                if boundaries[facet.index()] == marker:
                    for vertex in vertices(facet):
                        node_type[vertex.index(), comp] = 1
                        bc_nodes.add(vertex.index())

    # Assign column for swell
    swell_sides = bc_config.get("swell", [])
    swell_markers = [boundary_map[side] for side in swell_sides]
    for facet in facets(mesh):
        if boundaries[facet.index()] in swell_markers:
            for vertex in vertices(facet):
                node_type[vertex.index(), 4] = 1
                bc_nodes.add(vertex.index())

    # Column 0 = interior nodes (not part of any BC or swell)
    all_nodes = set(range(num_nodes))
    interior_nodes = all_nodes - bc_nodes
    for idx in interior_nodes:
        node_type[idx, 0] = 1

    return node_type



def solve_bending_swell(d_, chi, p, swell_function, bc_config):
    # Rectangle dimensions
    W, H, nx, ny = 0.08, 0.01, 32, 16

    mesh = RectangleMesh(Point(0.0, 0.0), Point(W, H), nx, ny)
    coordinates = mesh.coordinates()

    # Mark boundaries
    class Left(SubDomain):  
        def inside(self,x,on_boundary): return on_boundary and near(x[0], 0)
    class Right(SubDomain): 
        def inside(self,x,on_boundary): return on_boundary and near(x[0], W)
    class Top(SubDomain):   
        def inside(self,x,on_boundary): return on_boundary and near(x[1], H)
    class Bottom(SubDomain):
        def inside(self,x,on_boundary): return on_boundary and near(x[1], 0)

    left, right, top, bottom = Left(), Right(), Top(), Bottom()
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)
    top.mark(boundaries, 3)
    bottom.mark(boundaries, 4)

    node_type = build_node_type_from_config(mesh, boundaries, bc_config)
    # Function spaces
    element_u = VectorElement('CG', mesh.ufl_cell(), 2)
    element_phi = FiniteElement('CG', mesh.ufl_cell(), 1)
    V, Q = FunctionSpace(mesh, element_u), FunctionSpace(mesh, element_phi)

    # --- Build BCs dynamically from bc_config ---
    side_map = {"left": left, "right": right, "top": top, "bottom": bottom}
    bc_u, bc_phi, swell_markers = [], [], []

    if "ux" in bc_config:
        for side in bc_config["ux"]:
            bc_u.append(DirichletBC(V.sub(0), Constant(0.0), side_map[side]))
    if "uy" in bc_config:
        for side in bc_config["uy"]:
            bc_u.append(DirichletBC(V.sub(1), Constant(0.0), side_map[side]))
    if "phi" in bc_config:
        for side in bc_config["phi"]:
            bc_phi.append(DirichletBC(Q, Constant(0.45), side_map[side]))
    if "swell" in bc_config:
        # collect swell boundary markers (IDs)
        marker_map = {"left": 1, "right": 2, "top": 3, "bottom": 4}
        swell_markers = [marker_map[s] for s in bc_config["swell"]]

    (du, dφ) = TrialFunction(V), TrialFunction(Q)
    (v, q) = TestFunction(V), TestFunction(Q)

    (u, φ) = Function(V), Function(Q)
    (uold, φold) = Function(V), Function(Q)
    # V1 = VectorFunctionSpace(mesh, 'CG', 1)
    # Q1 = FunctionSpace(mesh, 'CG', 1)

    # # Assign u_last and phi_last to CG1 functions
    # u1 = Function(V1)
    # phi1 = Function(Q1)
    # u1.vector()[:] = np.array(u_last).flatten()
    # phi1.vector()[:] = np.array(phi_last).flatten()

    # # Interpolate to CG2 spaces
    # u_init = interpolate(u1, V)
    # phi_init = interpolate(phi1, Q)

    # # Assign to initial conditions
    # uold.assign(u_init)
    # φold.assign(phi_init)
    # u.assign(uold)
    # φ.assign(φold)
    # Create intial conditions and interpolate
    # u_init = InitialConditions()
    φ_init = Expression("0.75", degree=1)
    φold.interpolate(φ_init)
    φ.assign(φold)

    # Model parameters
    # Elasticity parameters
    G0, χ = Constant(10.0e6), Constant(chi)#Constant(0.1)
    K = Constant(100.0e6)#Constant(3.93e6) # Bulk modulus
    ΩK = Constant(100.0e6)
    NΩ = Constant(1e-3)
    Ω = Constant(1.7e-28)#Constant(1e-4)                          # the volume per solvent molecule
    kBT = Constant(1.3806488e-23*298)              # T = 298 K
    RT = Constant(8.31446261815324*298)
    D = Constant(d_)#Constant(0.1)#Constant(7.4e-11)                          # species diffusivity m^2 s^-1
    mc = Constant(0.1)
    gamma_s = Constant(10.)

    µ_o = Constant(0.0)

    alpha_r = 1.0 # Robin Boundary constant

    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)                         # Identity tensor

    def F(u):
        return variable(I + grad(u)) 

    # F = variable(I + grad(u))               # Deformation gradient
    # C = F.T*F                             # Right Cauchy-Green tensor

    # Left Cauchy-Green tensor
    def B(u):
        return variable(F(u)*F(u).T)

    # Invariants of deformation tensors
    def Ic(u):
        return variable(tr(F(u).T*F(u)))
    
    # Ic = tr(F(u).T*F(u))

    def J(u):
        return variable(det(F(u)))

    # J  = (det(F)) #1 + Ω*C

    # Cauchy stress
    def T_(u,φ):
        return variable(inv(J(u))*(G0*(B(u) - (φ**(-2))*I)))# + (1/φ)*K*ln(J(u)*φ)*I))

    # Piola stress - P = JTF^-T
    def PK1(u,φ):
        # return variable(G0*(B(u) - (φ**(-2))*I)*inv(F(u).T)) # <= as in Chester et al. 2010
        return variable((G0*(B(u) - I) + K*(ln(J(u)*0.999*φ))*I)*inv(F(u).T))         # <= as in Chester et al. 2011
        # return variable((G0*(B(u) - I) + (1/φ)*K*ln(J(u)*φ)*I)*inv(F(u).T))         # <= as in Chester et al. 2015

    # Chemical potential - µ
    def µ_(u,φ):
        # return variable(µ_o + kBT*(ln(1 - 0.999*φ) + φ + χ*(φ**2)) + (Ω*G0)*(0.999/φ - 0.999*φ)) # <= as in Chester et al. 2010
        return variable((µ_o) + RT*(ln(1 - 0.999*φ) + φ + χ*(φ**2)) - (Ω*K)*ln(J(u)*φ)*φ) # <= as in Chester et al. 2011
        # return variable((µ_o) + RT*(ln(1 - 0.999*φ) + φ + χ*(φ**2)) - (Ω*K)*ln(J(u)) + (1/2)*(Ω*K)*ln(J(u)*0.999*φ)**2)  # <= as in Chester et al. 2015



    ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
    dt = Constant(0.)

    # Forces balance
    Pi_mech_n = inner(PK1(u,φ), grad(v))*dx
    Jac_n = derivative(Pi_mech_n, u, du) 

    # Mass balance
    # let φ_swell be a function of time from swell_function 
    φ_swell = Constant(0.35)
    D = d_*kBT # <= this values works as well for Chester et al. 2011 and matches Chester et al. 2015
    alpha_r = 10 # <= this works for Chester et al. 2015 
    phi_form = (((φ - φold)/dt*q
                - (D/(kBT))*(dot(grad(φ),grad(µ_(u,φ))))*q 
                - (D/(kBT))*(dot(((φ**2)*((1 - 0.999*φ)/φ))*grad(µ_(u,φ)),grad(q))))*dx)

    # Add Robin term for all swell boundaries
    for marker in swell_markers:
        phi_form += alpha_r*(φ - φ_swell)*q*ds(marker)
    Jac_phi = derivative(phi_form, φ, dφ)



    # Time-stepping
    Nincr = 3000
    # t = np.linspace(0, 50, Nincr+1)
    #first 10 seconds log spacing from 1e-5 to 10, then linear spacing to 50
    t = np.concatenate((np.logspace(-5, 1, Nincr + 1), np.linspace(10, 60, 1250 + 1)[1:], np.linspace(60, 100, 1250 + 1)[1:]))
    tolerance = 1e-6  # Set your own tolerance level
    max_iterations = 10  # Set your own max number of iterations
    u_time_series = []
    φ_time_series = []
    for (i, dti) in enumerate(np.diff(t)):
        print("Increment " + str(i+1))
        t_current = t[i+1]
        print("t =", t_current) 
        dt.assign(dti)
        #update φ_swell in phi_form based on the swell function at the current time
        print("swell_function(t_current) =", swell_function(p, t_current))
        φ_swell.assign(swell_function(p, t_current))

        iteration = 0
        error = 1.0  # Initial error value
        
        while error > tolerance and iteration < max_iterations:
            iteration += 1
            
            # Save previous solutions
            u_prev = u.copy(deepcopy=True)
            φ_prev = φ.copy(deepcopy=True)
            
            # Compute tentative displacement step
            begin("Computing tentative displacement")
            solve(Pi_mech_n == 0, u, bc_u, J=Jac_n, solver_parameters={"newton_solver":
                                            {"relative_tolerance": 1e-6, "convergence_criterion": "incremental"}})
            end()
        
            # Chemical potential
            begin("Computing phi")
            solve(phi_form == 0, φ, J=Jac_phi)#bc_phi
            end()
            
            # Compute error as the maximum difference in the solutions
            error_u = np.max(np.abs(u.vector().get_local() - u_prev.vector().get_local()))
            error_φ = np.max(np.abs(φ.vector().get_local() - φ_prev.vector().get_local()))
            error = max(error_u, error_φ)
            
            print(f'Iteration: {iteration}, Error: {error}')
        
        # Store results for post-processing
        uold.assign(u)
        φold.assign(φ)
        
        u_time_series.append([u(x[0], x[1]) for x in coordinates])
        φ_time_series.append([φ(x[0], x[1]) for x in coordinates])
        if i == 4250:
            with XDMFFile(mesh.mpi_comm(), "checkpoints/u_restart.xdmf") as u_file, \
                XDMFFile(mesh.mpi_comm(), "checkpoints/phi_restart.xdmf") as phi_file:
                u_file.write_checkpoint(u, "u", i, XDMFFile.Encoding.HDF5, append=False)
                phi_file.write_checkpoint(φ, "phi", i, XDMFFile.Encoding.HDF5, append=False)
            print("Checkpoint saved at t =", t_current)
    #convert below list to array
    u_time_series = np.array(u_time_series)
    φ_time_series = np.array(φ_time_series)
    mesh_coords = np.array(mesh.coordinates())
    cells = np.array(mesh.cells())

    #reinterpolate time_series by cutting the logspace timestep off
    # interpolate to 200 timesteps
    # from scipy.interpolate import interp1d
    # num_timesteps = 400
    # t_new = np.linspace(t[1], t[-1], num_timesteps)
    # u_interp_func = interp1d(t[1:], u_time_series, axis=0)
    # φ_interp_func = interp1d(t[1:], φ_time_series, axis=0)
    # u_time_series = u_interp_func(t_new)
    # φ_time_series = φ_interp_func(t_new)
    swell_time_series = np.array([swell_function(p, ti) for ti in t])
    # t = t_new
    # save the data to a .npz file
    np.savez(f'/home/fenics/shared/dataset/bending_signal/bending_{d_}_{chi}_{p}.npz', 
             mesh_coords=mesh_coords, 
             cells=cells, 
             node_type = node_type, 
             diffusivity=d_, 
             chi=chi,
             u_time_series=u_time_series, 
             φ_time_series=φ_time_series,
             swell_time_series=swell_time_series[1:],
             t=t[1:])


if __name__ == "__main__":
    import time 
    import os
    from itertools import product 
    # constant swelling on the boundaries
    d = [7.5e-7, 7.5e-6] #, 7.5e-5]
    chi = [0.2, 0.3] #, 0.5]
    period = [5, 20, 40]
    bc_config = {
        'ux': ["left"],
        'uy': ["left"],
        'phi': [],
        'swell': ["top"]
    }
    #separate everthing swell function linear in time from 0.35 for first 10 seconds, then ramp to 0.75
    def swell_function(t):
        # Linear ramp expression
        return 0.35 + (0.65 - 0.35) * np.clip((t - 10) / 40, 0, 1)
    def swell_function(period, t):
        # Linear ramp contribution (active for 10 <= t <= 60)
        linear = 0.35 + (0.65 - 0.35) * np.clip((t - 10) / 40, 0, 1)
        
        # Cosine contribution (starts at t >= 60)
        cosine = 0.5 + 0.15 * np.cos(2 * np.pi * (t - 60) / period) * (t >= 60)
        
        # Combine: before 60 s use linear, after 60 s use linear + cosine - linear offset
        return linear * (t < 60) + cosine * (t >= 60)
    def swell_function(period, t, sharpness=10):
        # Linear ramp contribution (active for 10 <= t <= 60)
        linear = 0.35 + (0.65 - 0.35) * np.clip((t - 10) / 40, 0, 1)
        
        # Smooth square wave (starts at t >= 60)
        smooth_square = 0.5 + 0.15 * np.tanh(sharpness * np.cos(2 * np.pi * (t - 60) / period)) * (t >= 60)
        
        # Combine: before 60 s use linear, after 60 s use smooth square
        return linear * (t < 60) + smooth_square * (t >= 60)
    #create log file name uniaxial_log.txt if it doesn't exist
    log_file = "bending_log.txt"
    #check if log file exists, if not create it and write header
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Uniaxial swelling simulation log\n")
            f.write("d, chi, period, runtime (seconds)\n") 
    for d_, chi_, p in product(d, chi, period):
        print("Running for d =", d_, "chi =", chi_, "period = ", p)
        # save runtime in log file if it exists, else create it
        with open("bending_log.txt", "a") as f:
            start_time = time.time()
            solve_bending_swell(d_, chi_, p, swell_function, bc_config)
            end_time = time.time()
            f.write(f"d: {d_}, chi: {chi_}, period: {p}, runtime: {end_time - start_time} seconds\n") 
    
