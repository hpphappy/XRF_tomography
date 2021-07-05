import numpy as np
import torch as tc
import os
import dxchange
import torch.nn.functional as F


def rotation_grid(arr, theta, dev):
    """
    This function calcualtes the coordinates of the current grid points before rotation (at theta = 0).
    
    Parameters
    ----------
    arr: torch tensor
        grid concentration with the dimension in [C, N, H, W]
    
    theta: float
        rotation angle in radians 
        
    dev: string
        specify "cpu" or the cuda device
        
    Returns
    -------
    g: torch tensor
        the coordinates of the current grid points before rotation (at thetat = 0)
    """
    m0 = tc.tensor([tc.cos(theta), -tc.sin(theta), 0.0], device=dev)
    m1 = tc.tensor([tc.sin(theta), tc.cos(theta), 0.0], device=dev)
    m = tc.stack([m0, m1]).view(1, 2, 3)
    m = m.repeat([arr.shape[0], 1, 1])
    
    g = F.affine_grid(m, arr.shape)
    
    return g

def rotate(arr, theta, dev):
    """
    This function rotates the grid concentration with dimension: (n_element, sample_height_n, sample_size_n, sample_size_n)
    The rotational axis is along dim 1 of the grid
    
    Parameters
    ----------
    arr : torch tensor
         grid concentration with the dimension in [C, N, H, W]
        
    theta : float
        rotation angle in radians (clockwise)
    
    dev : string
        specify "cpu" or the cuda device (ex: cuda:0) 


    Returns
    -------
    q : torch tensor
        the rotated grid concentration

    """
    
    g = rotation_grid(arr, theta, dev)
    q = F.grid_sample(arr, g, padding_mode='border')
    
    return q

def get_cooridnates_stack_for_rotation(array_size, axis=0):
    """
    Get the coordinates of the grid points
    
    Parameters
    ----------     
    array_size: list
        object dimension in [N, H, W]
        
    Returns
    -------
    coord_new: numpy array
        The coordinates of the grid points with the dimensions [2, # of grid points]. 
        The first dimension is 2 because there're x and y coodinates for each grid point.
    
    """
    image_center = [(x - 1) / 2 for x in array_size]
    coords_ls = []
    for this_axis, s in enumerate(array_size):
        if this_axis != axis:
            coord = np.arange(s)
            for i in range(len(array_size)):
                if i != axis and i != this_axis:
                    other_axis = i
                    break
            if other_axis < this_axis:
                coord = np.tile(coord, array_size[other_axis])
            else:
                coord = np.repeat(coord, array_size[other_axis])
            coords_ls.append(coord - image_center[i])
    coord_new = np.stack(coords_ls)
    return coord_new

def calculate_original_coordinates_for_rotation(array_size, coord_new, theta, dev=None):
    """
    Calculate the old coordinates (coodinates without rotation) of the current grid points before rotation by rotating the current grid points back.
    
    Parameters
    ----------
    array_size: list
        object dimension in [N, H, W]

    cood_new: numpy array
        Use the return output from the get_cooridnates_stack_for_rotation function.

    theta: float
    
    Returns
    -------
    coord_old: torch tensor
        the old coordinates (before rotation) of the current grid points
        the dimension = [# of grid points, 2]
    """
    image_center = [(x - 1) / 2 for x in array_size]
    m0 = tc.tensor([tc.cos(theta), -tc.sin(theta)], device=dev)
    m1 = tc.tensor([tc.sin(theta), tc.cos(theta)], device=dev)
    m_rot = tc.stack([m0, m1])

    coord_old = tc.matmul(m_rot, coord_new)
    coord1_old = coord_old[0, :] + image_center[1]
    coord2_old = coord_old[1, :] + image_center[2]
    coord_old = np.stack([coord1_old, coord2_old], axis=1)
    return coord_old


def save_rotation_lookup(array_size, theta_ls, dest_folder=None):
    """
    Save the old coordinates (before rotating theta) of the current grid points for all object angles in theta_ls.
    Save the old coordinates (before rotating -theta) of the current grid points for all object angles in theta_ls.
    
    Parameters
    ----------
    array_size: list
        object dimension in [N, H, W]
        
    theta_ls: torch tensor
        a torch tensor of all obeject angles
        
    dest_folder: string
        the path of storing the old coordinates
    
    Returns
    -------
    None
        
    """

    # create matrix of coordinates
    coord_new = tc.from_numpy(get_cooridnates_stack_for_rotation(array_size, axis=0).astype(np.float32))

    n_theta = len(theta_ls)
    if dest_folder is None:
        dest_folder = 'arrsize_{}_{}_{}_ntheta_{}'.format(array_size[0], array_size[1], array_size[2], n_theta)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    for i, theta in enumerate(theta_ls):  #changed from theta_ls[rank:n_theta:n_ranks]
        coord_old = calculate_original_coordinates_for_rotation(array_size, coord_new, theta)
        coord_inv = calculate_original_coordinates_for_rotation(array_size, coord_new, -theta)
        # coord_old_ls are the coordinates in original (0-deg) object frame at each angle, corresponding to each
        # voxel in the object at that angle.
        np.save(os.path.join(dest_folder, '{:.5f}'.format(theta)), coord_old.astype('float16'))
        np.save(os.path.join(dest_folder, '_{:.5f}'.format(theta)), coord_inv.astype('float16'))
    return None

def read_origin_coords(src_folder, theta, reverse=False):
    """
    Read the old coordinates from the file generated from the save_rotation_lookup function
    
    Parameters
    ----------
    src_folder: string
        saving path of the old coordinates
    
    theta: float
        object angle
    
    reverse: Boolean
        If the rotating angle is theta, set reverse to True.
    """
    if not reverse:
        coords = np.load(os.path.join(src_folder, '{:.5f}.npy'.format(theta)), allow_pickle=True)
    else:
        coords = np.load(os.path.join(src_folder, '_{:.5f}.npy'.format(theta)), allow_pickle=True)
    return tc.from_numpy(coords).type(tc.float)


def initialize_guess_3d(dev, ini_kind, grid_path, f_grid, recon_path, f_recon_grid, f_initial_guess, init_const=0.5):
    if ini_kind == "rand":
        X = np.load(os.path.join(grid_path, f_grid)).astype(np.float32)  # The shape of X(3d) is initally (5,5,5,5) = (n_element, n_z, n_x, n_y)
        X = tc.from_numpy(X).float().to(dev)
        X = X + 0.1 * tc.rand(X.shape[0], X.shape[1], X.shape[2], X.shape[3], device=dev)
        X = tc.clamp(X, 0, 10)

    elif ini_kind == "randn":
        X = np.load(os.path.join(grid_path, f_grid)).astype \
            (np.float32)  # The shape of X(3d) is initally (5,5,5,5) = (n_element, n_z, n_x, n_y)
        X = tc.from_numpy(X).float().to(dev)
        X = X + 0.1 * tc.randn(X.shape[0], X.shape[1], X.shape[2], X.shape[3], device=dev)
        X = tc.clamp(X, 0, 10)

    elif ini_kind == "const":
        X = np.load(os.path.join(grid_path, f_grid)).astype \
            (np.float32)  # X is loaded just to get the shape of the model X
        X = tc.from_numpy(X).float().to(dev)
        X = tc.zeros(X.shape[0], X.shape[1], X.shape[2], X.shape[3], device=dev) + init_const

    else:
        print("Please specify the correct kind of the initialization condition.")
        
    ## Save the initial guess for future reference
    np.save(os.path.join(recon_path, f_initial_guess ) +'.npy', X.cpu())
    dxchange.write_tiff(X.cpu(), os.path.join(recon_path, f_initial_guess), dtype='float32', overwrite=True)

    ## Save the initial guess which will be used in reconstruction and will be updated to the current reconstructing result
    np.save(os.path.join(recon_path, f_recon_grid ) +'.npy', X.cpu())
    dxchange.write_tiff(X.cpu(), os.path.join(recon_path, f_recon_grid), dtype='float32', overwrite=True)

    return X



def init_xp(recon_path, theta, dev, cmap_rot, n_element, sample_height_n, sample_size_n, minibatch_size, p):
    """
    Slicing the current updated target out from the rotated whole object. (n_element, n_z in this minibatch, n_x, n_y)
    
    Parameters
    ----------
    cmap_rot: torch tensor
        the rotated whole object with the dimension [n_element, sample_height_n, sample_size_n, sample_size_n] ([C, N, H, W])
        
    n_element: integer
        the number of elements in this object
    
    sample_size_n: integer
        the number of voxel along the x or y direction
    
    minibatch_size: integer
        the number of strips along the x direction in this minibatch
    
    p: integer
        the index of the current minibatch
    
    Returns
    -------
    The 
    
    """
    
    def apply_rotation_transpose_to_grad(obj, interpolation='bilinear', axis=0, device=None):
        """
        Apply the transpose of the rotation matrix to the obj matrix.

        Parameters
        ----------
        obj: torch tensor
            obj is the tensor which the rotation transpose matrix is going to act on.
            The dimension = [C, N, H, W]

        Returns
        -------
        coord_old: torch tensor
            the old coordinates (before rotation) of the current grid points
            the dimension = [# of grid points, 2]

        interpolation: string
            the interpolation method to get the values in the new grid

        axis: integer
            the axis which the obj rotates about

        Returns
        -------
        obj_rot: torch tensor
            the torch tensor after applying the rotation transpose
            the dimension = [C, N, H, W]

        """
        coord_old = read_origin_coords(os.path.join(recon_path, "rotation_look_up"), theta).to(dev)
        # The dimension of the input obj is [C, N, H, W]
        # This piece of code is borowed from Ming. He arranged the object in [N, H, W, C]
        obj = obj.permute(1,2,3,0)
        s = obj.shape
        axes_rot = []
        for i in range(len(obj.shape)):
            if i != axis and i <= 2:
                axes_rot.append(i)

        coord_old_1 = coord_old[:, 0]
        coord_old_2 = coord_old[:, 1]

        # Clip coords, so that edge values are used for out-of-array indices
        coord_old_1 = tc.clamp(coord_old_1, 0, s[axes_rot[0]] - 1)
        coord_old_2 = tc.clamp(coord_old_2, 0, s[axes_rot[1]] - 1)

        coord_old_floor_1 = tc.floor(coord_old_1).type(tc.int64)
        coord_old_ceil_1 = coord_old_floor_1 + 1
        coord_old_floor_2 = tc.floor(coord_old_2).type(tc.int64)
        coord_old_ceil_2 = coord_old_floor_2 + 1
        # create an empty object to store the results after applying the rotation transpose
        obj_rot = tc.zeros_like(obj, requires_grad=False)

        # calculate the area fraction
        # the dimension of fac_XX = [H * W]
        fac_ff = (coord_old_ceil_1 - coord_old_1) * (coord_old_ceil_2 - coord_old_2)
        fac_fc = (coord_old_ceil_1 - coord_old_1) * (coord_old_2 - coord_old_floor_2)
        fac_cf = (coord_old_1 - coord_old_floor_1) * (coord_old_ceil_2 - coord_old_2)
        fac_cc = (coord_old_1 - coord_old_floor_1) * (coord_old_2 - coord_old_floor_2)

        # expand fac_XX along the dimension C, the dimension = [H * W, C]
        fac_ff = tc.stack([fac_ff] * s[-1], axis=1)
        fac_fc = tc.stack([fac_fc] * s[-1], axis=1)
        fac_cf = tc.stack([fac_cf] * s[-1], axis=1)
        fac_cc = tc.stack([fac_cc] * s[-1], axis=1)

        # apply the rotation transpose to each layer (along dimension N)
        for i_slice in range(s[axis]):

            # slicer_XX represents the indices of the corner points around each points of coord_old.
            # XX = ff, fc, cf, cc represent 4 corners respectively
            # slicer_XX -> [i_slice, tc.tensor([x1, x2, x3, ...]), tc.tensor([y1, y2, y3, ...])]
            slicer_ff = [i_slice, i_slice, i_slice]
            slicer_ff[axes_rot[0]] = coord_old_floor_1
            slicer_ff[axes_rot[1]] = coord_old_floor_2

            slicer_fc = [i_slice, i_slice, i_slice]
            slicer_fc[axes_rot[0]] = coord_old_floor_1
            slicer_fc[axes_rot[1]] = tc.clamp(coord_old_ceil_2, 0, s[axes_rot[1]] - 1)

            slicer_cf = [i_slice, i_slice, i_slice]
            slicer_cf[axes_rot[0]] = tc.clamp(coord_old_ceil_1, 0, s[axes_rot[0]] - 1)
            slicer_cf[axes_rot[1]] = coord_old_floor_2

            slicer_cc = [i_slice, i_slice, i_slice]
            slicer_cc[axes_rot[0]] = tc.clamp(coord_old_ceil_1, 0, s[axes_rot[0]] - 1)
            slicer_cc[axes_rot[1]] = tc.clamp(coord_old_ceil_2, 0, s[axes_rot[1]] - 1)

            # slicer_obj is the index that retrieves a single layer in dimension N
            slicer_obj = [slice(None), slice(None), slice(None)]
            slicer_obj[axis] = i_slice

            # use slicer_obj to retrieve a single layer from obj
            # obj[slicer_obj], dimension = [H, W, C]
            # obj_slice, dimension = [H * W, C]
            obj_slice = tc.reshape(obj[slicer_obj], [-1, s[-1]])  # originally: [-1,2]

            # tuple(slicer_XX) retrieves the corner points
            obj_rot[tuple(slicer_ff)] += obj_slice * fac_ff
            obj_rot[tuple(slicer_fc)] += obj_slice * fac_fc
            obj_rot[tuple(slicer_cf)] += obj_slice * fac_cf
            obj_rot[tuple(slicer_cc)] += obj_slice * fac_cc

        obj_rot = obj_rot.permute(3,0,1,2)
        return obj_rot
    
    # unfold the dimension: [C, N, H, W] -> [C, N * H, W]
    cmap_rot = cmap_rot.view(n_element, sample_height_n * sample_size_n, sample_size_n)
    
    # pick the index from N * H that belongs the current minibatch
    cmap_rot_this_minibatch = cmap_rot[:, minibatch_size * p : minibatch_size * (p + 1), :]
    
    # fold the dimension back: [C, N(this minibach) * H, W] -> [C, N(this minibatch), H, W]
    cmap_rot_this_minibatch = cmap_rot_this_minibatch.view(n_element, minibatch_size // sample_size_n, sample_size_n, sample_size_n)
    
    # set requires_grad to True
    cmap_rot_this_minibatch = cmap_rot_this_minibatch.clone().detach().requires_grad_(True)
    
    # register the hook so that the gradient gets modified during the backward propagation
    cmap_rot_this_minibatch.register_hook(apply_rotation_transpose_to_grad)
    
    return cmap_rot_this_minibatch