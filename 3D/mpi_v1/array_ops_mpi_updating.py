import numpy as np
import torch as tc
import os
import dxchange



def initialize_guess_3d(dev, ini_kind, grid_path, f_grid, recon_path, f_recon_grid, f_initial_guess, init_const=0.5):
    if ini_kind == "rand":
        # The shape of X(3d) is initally (5,5,5,5) = (n_element, n_z, n_x, n_y)
        X = np.load(os.path.join(grid_path, f_grid)).astype(np.float32)  
        X = tc.from_numpy(X).float().to(dev)
        X = X + 0.1 * tc.rand(X.shape[0], X.shape[1], X.shape[2], X.shape[3], device=dev)
        X = tc.clamp(X, 0, 10)

    elif ini_kind == "randn":
        # The shape of X(3d) is initally (5,5,5,5) = (n_element, n_z, n_x, n_y)
        X = np.load(os.path.join(grid_path, f_grid)).astype(np.float32)  
        X = tc.from_numpy(X).float().to(dev)
        X = X + 0.1 * tc.randn(X.shape[0], X.shape[1], X.shape[2], X.shape[3], device=dev)
        X = tc.clamp(X, 0, 10)

    elif ini_kind == "const":
        # X is loaded just to get the shape of the model X
        X = np.load(os.path.join(grid_path, f_grid)).astype(np.float32) 
        X = tc.from_numpy(X).float().to(dev)
        X = tc.zeros(X.shape[0], X.shape[1], X.shape[2], X.shape[3], device=dev) + init_const

    else:
        print("Please specify the correct kind of the initialization condition.")


    return X