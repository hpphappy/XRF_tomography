#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:34:50 2020

@author: panpanhuang
"""

from mpi4py import MPI
import datetime
from numpy.random import default_rng
import h5py
import numpy as np
import xraylib as xlib
import xraylib_np as xlib_np
import torch as tc
import torch.nn.functional as F
import os
import sys
from tqdm import tqdm
from misc_mpi_updating_realData import print_flush_root, print_flush_all
from Atomic_number import AN

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

# Note: xraylib uses keV 

# sub-lines of K, L, M lines with the required format by xraylib.

fl = {"K": np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE]),
      "L": np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE]),              
      "M": np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])               
     }


fl_line_groups = np.array(["K", "L", "M"])


def rotate(arr, theta, dev):
    """
    This function rotates the grid concentration with dimension: (n_element, sample_height_n, sample_size_n, sample_size_n)
    The rotational axis is along dim 1 of the grid
    
    Parameters
    ----------
    arr : torch tensor
        grid concentration
        
    theta : float
        rotation angle in radians (clockwise)
    
    dev : string
        specify "cpu" or the cuda diveice (ex: cuda:0)


    Returns
    -------
    q : torch tensor
        the rotated grid concentration

    """
    
    m0 = tc.tensor([tc.cos(theta), -tc.sin(theta), 0.0], device=dev)
    m1 = tc.tensor([tc.sin(theta), tc.cos(theta), 0.0], device=dev)
    m = tc.stack([m0, m1]).view(1, 2, 3)
    m = m.repeat([arr.shape[0], 1, 1])
    
    g = F.affine_grid(m, arr.shape)
    q = F.grid_sample(arr, g, padding_mode='border')
    
    return q



def attenuation_3d(src_path, theta_st, theta_end, n_theta, sample_height_n, sample_size_n,
                sample_size_cm, this_aN_dic, probe_energy, dev):
    """  
    Parameters
    ----------
    src_path : string
        the path of the elemental concentration grid
        
    theta_st: float
        The initial angle of the sample
        
    theta_end: float
        The final angle of the sample
        
    n_theta: integer
        The number of sample angles
        
    sample_height_n : integer
        The height of the sample along the rotational axis (in number of pixels)
    
    sample_size_n: int scalar
        sample size in number of pixles on the side along the probe propagation axis

    sample_size_cm: scalar
        sample size in cm on the side along the probe propagation axis
    
    this_aN_dic: dictionary
        a dictionary of items with key = element symbol (string), and value = atomic number
        e.g. this_aN_dic = {"C":6, "O": 8}
        
    probe_energy : ndarray
        This array is an array with only 1 element. The element is the keV energy of the incident beam.
        
    dev : string
        specify "cpu" or the cuda diveice (ex: cuda:0)

    Returns
    -------
    attenuation_map_flat : torch tensor
         an array of attenuation ratio before the probe enters each voxel.
         dim 0: all angles of the sample
         dim 1: all voxels (flattened 3D array)
      
    transmission : TYPE
        DESCRIPTION.
    """    
    
    n_element = len(this_aN_dic)
    theta_ls = - tc.linspace(theta_st, theta_end, n_theta + 1)[:-1]
    grid_concentration = tc.tensor(np.load(src_path)).float().to(dev)
    aN_ls = np.array(list(this_aN_dic.values()))
    probe_attCS_ls = tc.tensor(xlib_np.CS_Total(aN_ls, probe_energy).flatten()).float().to(dev)
    
    att_exponent_acc_map = tc.zeros((len(theta_ls), sample_height_n, sample_size_n, sample_size_n+1), device=dev)
    for i , theta in enumerate(theta_ls):
        theta = tc.tensor(theta,  device=dev)
        concentration_map_rot = rotate(grid_concentration, theta, dev)
        for j in range(n_element):
            lac_single = concentration_map_rot[j] * probe_attCS_ls[j]
            lac_acc = tc.cumsum(lac_single, axis=2)
            lac_acc = tc.cat((tc.zeros((sample_height_n, sample_size_n, 1), device=dev), lac_acc), dim = 2)
            att_exponent_acc = lac_acc * (sample_size_cm / sample_size_n) 
            att_exponent_acc_map[i,:,:,:] += att_exponent_acc

    attenuation_map_flat = tc.exp(-(att_exponent_acc_map[:,:,:,:-1])).view(n_theta, sample_height_n * sample_size_n * sample_size_n).float().to(dev)
    transmission = tc.exp(-att_exponent_acc_map[:,:,:,-1]).view(n_theta, sample_height_n * sample_size_n).float().to(dev)
    
    return attenuation_map_flat, transmission


def create_XRT_data_3d(src_path, theta_st, theta_end, n_theta, sample_height_n, sample_size_n,
                         sample_size_cm, this_aN_dic, probe_energy, probe_cts, save_path, save_fname, theta_sep, Poisson_noise, dev):
    """
    Parameters
    ----------
    src_path: string
        the path of the elemental concentration grid
        
    theta_st: float
        The initial angle of the sample
        
    theta_end: float
        The final angle of the sample
        
    n_theta: integer
        The number of sample angles
        
    sample_height_n : integer
        The height of the sample along the rotational axis (in number of pixels)
           
    sample_size_n: int scalar
        sample size in number of pixles on the side along the probe propagation axis

    sample_size_cm: scalar
        sample size in cm on the side along the probe propagation axis
    
    this_aN_dic: dictionary
        a dictionary of items with key = element symbol (string), and value = atomic number
        e.g. this_aN_dic = {"C":6, "O": 8}
        
    probe_energy : ndarray
        This array is an array with only 1 element. The element is the keV energy of the incident beam.
        
    probe_cts : float
        The incident photon counts/s
    
    save_path : string
        The directory of saving the XRT_data

    Returns
    -------
    XRT_data : ndarray
        The dimension of the array is (n_theta, sample_height_n * sample_size_n)
        [note: sample_size may not be the same as the input argument because of padding]
    """   
    XRT_data = probe_cts * attenuation_3d(src_path, theta_st, theta_end, n_theta, sample_height_n, sample_size_n,
                sample_size_cm, this_aN_dic, probe_energy, dev)[1]
    
    if Poisson_noise == True:
        random_noise_generator = default_rng()
        XRT_data = random_noise_generator.poisson(XRT_data)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        pass    
    
    if theta_sep == True:       
        for this_theta_idx in tqdm(range(n_theta)):
            np.save(os.path.join(save_path, save_fname +'_{}'.format(this_theta_idx)), XRT_data[this_theta_idx])
    
    else:
        np.save(os.path.join(save_path, save_fname), XRT_data.cpu())
    
    return XRT_data


def MakeFLlinesDictionary(this_aN_dic, probe_energy,
                          sample_size_n, sample_size_cm,
                          fl_line_groups = np.array(["K", "L", "M"]), fl_K = fl["K"], fl_L = fl["L"], fl_M = fl["M"],
                          group_lines = True):


    element_ls = np.array(list(this_aN_dic.keys()))
    aN_ls = np.array(list(this_aN_dic.values()))

    n_line_group = len(fl_line_groups)
    FL_all_elements_dic = {"element_Line": [], "fl_energy": np.array([]), "detected_fl_unit_concentration": np.array([])}
    voxel_size = sample_size_cm/sample_size_n   

    fl_cs_K = xlib_np.CS_FluorLine_Kissel_Cascade(aN_ls, fl_K, probe_energy)
    fl_cs_L = xlib_np.CS_FluorLine_Kissel_Cascade(aN_ls, fl_L, probe_energy)
    fl_cs_M = xlib_np.CS_FluorLine_Kissel_Cascade(aN_ls, fl_M, probe_energy)

    # Remove the extra dimension with only 1 element
    fl_cs_K = np.reshape(fl_cs_K, (fl_cs_K.shape[:-1]))
    fl_cs_L = np.reshape(fl_cs_L, (fl_cs_L.shape[:-1]))
    fl_cs_M = np.reshape(fl_cs_M, (fl_cs_M.shape[:-1]))

    fl_energy_K = xlib_np.LineEnergy(aN_ls, fl_K)
    fl_energy_L = xlib_np.LineEnergy(aN_ls, fl_L)
    fl_energy_M = xlib_np.LineEnergy(aN_ls, fl_M)

    FL_all_elements_dic = {"(element_name, Line)": [], "fl_energy": np.array([]), "detected_fl_unit_concentration": np.array([]),
                           "n_line_group_each_element": np.array([]), "n_lines": None}
    if group_lines == True:
        fl_energy_group = np.zeros((len(element_ls),n_line_group))
        fl_cs_group = np.zeros((len(element_ls),n_line_group))
        
        for i, element_name in enumerate(element_ls): 

            if np.sum(fl_cs_K[i] != 0):
                fl_energy_group[i,0] = np.average(fl_energy_K[i], weights=fl_cs_K[i]) 
                fl_cs_group[i,0] = np.sum(fl_cs_K[i])
            else:
                fl_energy_group[i,0] = 0
                fl_cs_group[i,0] = 0

            if np.sum(fl_cs_L[i] != 0):
                fl_energy_group[i,1] = np.average(fl_energy_L[i], weights=fl_cs_L[i]) 
                fl_cs_group[i,1] = np.sum(fl_cs_L[i])
            else:
                fl_energy_group[i,1] = 0
                fl_cs_group[i,1] = 0

            if np.sum(fl_cs_M[i] != 0):
                fl_energy_group[i,2] = np.average(fl_energy_M[i], weights=fl_cs_M[i]) 
                fl_cs_group[i,2] = np.sum(fl_cs_M[i])
            else:
                fl_energy_group[i,2] = 0
                fl_cs_group[i,2] = 0

            element_Line = fl_line_groups[fl_energy_group[i]!= 0]
            element_Line = [[element_name, element_Line[j]] for j in range(len(element_Line))]
            for k in range(len(element_Line)):
                FL_all_elements_dic["(element_name, Line)"].append(element_Line[k])     

            Line_energy = fl_energy_group[i][fl_energy_group[i]!=0]
            FL_all_elements_dic["fl_energy"] = np.append(FL_all_elements_dic["fl_energy"], Line_energy)
            fl_unit_con = fl_cs_group[i][fl_energy_group[i]!=0] * voxel_size
            FL_all_elements_dic["detected_fl_unit_concentration"] = np.append(FL_all_elements_dic["detected_fl_unit_concentration"], fl_unit_con)
            FL_all_elements_dic["n_line_group_each_element"] = np.append(FL_all_elements_dic["n_line_group_each_element"], len(fl_unit_con))
            
        FL_all_elements_dic["(element_name, Line)"] = np.array(FL_all_elements_dic["(element_name, Line)"])
    
    FL_all_elements_dic["n_lines"] = len(FL_all_elements_dic["(element_name, Line)"])
    return FL_all_elements_dic


def find_lines_roi_idx_from_dataset(data_path, f_XRF_data, element_lines_roi, std_sample):
    XRF_data = h5py.File(os.path.join(data_path, f_XRF_data), 'r')
    
    if std_sample:
        channel_names = XRF_data['MAPS/channel_names'][...]
    else:
        channel_names = XRF_data['exchange/elements'][...]
        
    channel_names = np.array([str(channel_name, 'utf-8') for channel_name in channel_names])
    element_lines_roi_idx = np.zeros(len(element_lines_roi)).astype(np.int)
    for i, element_line_roi in enumerate(element_lines_roi):
        if element_line_roi[1] == "K":
            channel_name_roi = element_line_roi[0]
        else:
            channel_name_roi = element_line_roi[0] + "_" + element_line_roi[1]
        element_line_idx = np.argwhere(channel_names == channel_name_roi)
        element_lines_roi_idx[i] = element_line_idx
        
    XRF_data.close()
    return element_lines_roi_idx   


def MakeFLlinesDictionary_manual(element_lines_roi,                           
                                 n_line_group_each_element, probe_energy, 
                                 sample_size_n, sample_size_cm,
                                 fl_line_groups = np.array(["K", "L", "M"]), fl_K = fl["K"], fl_L = fl["L"], fl_M = fl["M"]):


    FL_all_elements_dic = {"(element_name, Line)": [], "fl_energy": np.array([]), "detected_fl_unit_concentration": np.array([]),
                           "n_line_group_each_element": np.array([]), "n_lines": None}

    FL_all_elements_dic["(element_name, Line)"] = element_lines_roi
    FL_all_elements_dic["n_line_group_each_element"] = n_line_group_each_element
    FL_all_elements_dic["n_lines"] = len(element_lines_roi)

    voxel_size = sample_size_cm/sample_size_n   

    for i, element_line_roi in enumerate(element_lines_roi):
        fl_energy = xlib_np.LineEnergy(np.array([AN[element_line_roi[0]]]), fl[element_line_roi[1]]).flatten()
        fl_cs = xlib_np.CS_FluorLine_Kissel_Cascade(np.array([AN[element_line_roi[0]]]), fl[element_line_roi[1]], probe_energy).flatten()

        if np.sum(fl_cs) != 0:
            fl_energy_group = np.average(fl_energy, weights=fl_cs) 
            fl_cs_group = np.sum(fl_cs)
        else:
            fl_energy_group = 0.
            fl_cs_group = 0.

        FL_all_elements_dic["fl_energy"] = np.append(FL_all_elements_dic["fl_energy"], fl_energy_group)
        fl_unit_con = fl_cs_group * voxel_size
        FL_all_elements_dic["detected_fl_unit_concentration"] = np.append(FL_all_elements_dic["detected_fl_unit_concentration"], fl_unit_con)   
        
    return FL_all_elements_dic


def generate_fl_signal_from_each_voxel_3d(src_path, theta_st, theta_end, n_theta, sample_size_n, sample_height_n, sample_size_cm, this_aN_dic, probe_energy, dev):
    """
    This function calculates the ratio of fluoresence signal genenerated at each voxel at each object angle
    The rotational axis is along dim 0 of the grid

    Parameters
    ----------
    src_path: string
        the path of the elemental concentration grid
        
    theta_st: float
        The initial angle of the sample
        
    theta_end: float
        The final angle of the sample
        
    n_theta: integer
        The number of sample angles

    sample_size_n: int scalar
        sample size in number of pixles on the side along the probe propagation axis
        
    sample_height_n : integer
        The height of the sample along the rotational axis (in number of pixels)

    sample_size_cm: scalar
        sample size in cm on the side along the probe propagation axis
        
    this_aN_dic: dictionary
        a dictionary of items with key = element symbol (string), and value = atomic number
        e.g. this_aN_dic = {"C":6, "O": 8}
        
    probe_energy : ndarray
        This array is an array with only 1 element. The element is the keV energy of the incident beam.
        
    dev : string
        specify "cpu" or the cuda diveice (ex: cuda:0)

    Returns
    -------
    fl_map_tot : torch tensor with the dimension (n_theta, n_lines, sample_height_n * sample_size_n * sample_size_n)
        

    """
    element_ls = np.array(list(this_aN_dic.keys()))
    n_element = tc.tensor(len(element_ls)).to(dev)
    theta_ls = - tc.linspace(theta_st, theta_end, n_theta+1)[:-1].to(dev)

    grid_concentration = tc.tensor(np.load(src_path)).float().to(dev)

    fl_all_lines_dic = MakeFLlinesDictionary(this_aN_dic, probe_energy,
                              sample_size_n.cpu().numpy(), sample_size_cm.cpu().numpy(),
                              fl_line_groups = np.array(["K", "L", "M"]), fl_K = fl_K, fl_L = fl_L, fl_M = fl_M,
                              group_lines = True)

    fl_map_tot = tc.zeros((n_theta, fl_all_lines_dic["n_lines"], sample_height_n * sample_size_n * sample_size_n), device=dev)
    for i, theta in enumerate(theta_ls):
        concentration_map_rot = rotate(grid_concentration, tc.tensor(theta, dtype=tc.float32), dev)
        concentration_map_rot_flat = concentration_map_rot.view(len(element_ls), sample_height_n * sample_size_n * sample_size_n)
        line_idx = 0
        for j in range(n_element):
            ## fetch the generated fl signal at unit concentration for the calculated voxel size
            fl_unit = fl_all_lines_dic["detected_fl_unit_concentration"][line_idx:line_idx + int(fl_all_lines_dic["n_line_group_each_element"][j])]   
            ## FL signal over the current elemental lines for each voxel
            fl_map = [concentration_map_rot_flat[j] * fl_unit_single_line for fl_unit_single_line in fl_unit]
            fl_map = tc.stack(fl_map).float()
            fl_map_tot[i, line_idx:line_idx + fl_map.shape[0],:] = fl_map 
            line_idx = line_idx + len(fl_unit)
            
    return fl_map_tot


### The following trace_beam functions solves the intersection of a ray with planes 
### There're 3 types of plane could be specified: x = some constant (d_x), y = some constant (d_y) and z = some constant (d_z)
### The correspoinding intersecting points can be solved using trace_beam_x, trace_beam_y, trace_beam_z respectively

# The ray uses a parametric form with a parameter, t: R(t) = (1-t) * S + t * D; S and D are the coordinates of sample voxels and the detector points
# S = (z_s, x_s, y_s); D = (z_d, x_d, y_d)
# The intersecting coordinates: (z, x, y) = (Iz, Ix, Iy) at t=t'
# 4 equations are used to solve the intersecting point:
# From the parametric function of the ray
#    Iz = (1-t') * z_s + t' * z_d
#    Ix = (1-t') * x_s + t' * x_d
#    Iy = (1-t') * y_s + t' * y_d
# From the function of the plane: 
#    Ix = some constant (d_x), Iy = some constant (d_y) or Iz = some constant (d_z)

# Rearrange the equations above to solve (Iz, Ix, Iy, t')
# Define the system of equation AX = b to solve the intersecting point, A is with the dimension: (n_batch, 4, 4), b is with the dimension: (n_batch, 4, 1)
# n_batch is the number of planes we put into the equation that we want to solve the intersecting point with the the ray
def trace_beam_z(z_s, x_s, y_s, z_d, x_d, y_d, d_z_ls):
    # For the case that the voxel and the detector have the same z coordinate, the connection of them doesn't have any intersection on any plane along z-direction.
    if len(d_z_ls) == 0 or z_s == z_d:
        Z = np.stack((np.array([]), np.array([]), np.array([])), axis=-1)
    else:
        A = tc.tensor([[1, 0, 0, z_s - z_d],[0, 1, 0, x_s - x_d],[0, 0, 1, y_s - y_d],[1, 0, 0, 0]])
        A = A.repeat([len(d_z_ls), 1, 1])

        b1 = tc.tensor([[[z_s], [x_s], [y_s]]]).repeat([len(d_z_ls), 1, 1])
        b2 = tc.tensor([[[d_z]] for d_z in d_z_ls])
        b = tc.cat((b1, b2), dim=1)

        Z, LU = tc.solve(b, A)
        Z = np.array(Z[:,:-1].view(len(d_z_ls), 3))
#         t = X[:,-1] 
    
    return Z

def trace_beam_x(z_s, x_s, y_s, z_d, x_d, y_d, d_x_ls):
    # For the case that the voxel and the detector have the same x coordinate, the connection of them doesn't have any intersection on any plane along x-direction.
    if len(d_x_ls) == 0:
        X = np.stack((np.array([]), np.array([]), np.array([])), axis=-1)
    else:    
        A = tc.tensor([[1, 0, 0, z_s - z_d],[0, 1, 0, x_s - x_d],[0, 0, 1, y_s - y_d],[0, 1, 0, 0]])
        A = A.repeat([len(d_x_ls), 1, 1])

        b1 = tc.tensor([[[z_s], [x_s], [y_s]]]).repeat([len(d_x_ls), 1, 1])
        b2 = tc.tensor([[[d_x]] for d_x in d_x_ls])
        b = tc.cat((b1, b2), dim=1)

        X, LU = tc.solve(b, A)
        X = np.array(X[:,:-1].view(len(d_x_ls), 3))
#         t = Y[:,-1]
    
    return X

def trace_beam_y(z_s, x_s, y_s, z_d, x_d, y_d, d_y_ls):
    # For the case that the voxel and the detector have the same y coordinate, the connection of them doesn't have any intersection on any plane along y-direction.
    if len(d_y_ls) == 0 or y_s == y_d:
        Y = np.stack((np.array([]), np.array([]), np.array([])), axis=-1)
    else:
        A = tc.tensor([[1, 0, 0, z_s - z_d],[0, 1, 0, x_s - x_d],[0, 0, 1, y_s - y_d],[0, 0, 1, 0]])
        A = A.repeat([len(d_y_ls), 1, 1])

        b1 = tc.tensor([[[z_s], [x_s], [y_s]]]).repeat([len(d_y_ls), 1, 1])
        b2 = tc.tensor([[[d_y]] for d_y in d_y_ls])
        b = tc.cat((b1, b2), dim=1)

        Y, LU = tc.solve(b, A)
        Y = np.array(Y[:,:-1].view(len(d_y_ls), 3))
#         t = Z[:,-1]
    
    return Y

def intersecting_length_fl_detectorlet_3d_mpi_write_h5(n_ranks, rank, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n, sample_size_cm, sample_height_n, P_folder, f_P):
    
    """
    Parameters
    ----------
    det_size_cm : float
        The diameter of the circle to distribute the detector points
        
    det_from_sample_cm : float
        The distance between the detector plane and the sample boundary plane
    
    det_ds_spacing_cm : float
        The spacing between detector points
    
    sample_size_n: int scalar
        sample size in number of pixles on the side along the probe propagation axis
    
    sample_size_cm: scalar
        sample size in cm on the side along the probe propagation axis
        
    sample_height_n : integer
        The height of the sample along the rotational axis (in number of pixels)
        
    P_save_path : string
        The path that saves the tensor P

    Returns
    -------
    n_det : integer
        The number of the detector points within the circle with the diatmeter, det_size_cm.
    
    P : torch tensor
        a tensor with the dimension (n_det, 3, n_voxels *  diagnal_length_n)
        n_voxels: the number of voxels of the sample.
        diagnal_length_n: the number of voxels along the diagnol direction of the sample
        
        P tensor contains the information of intersecting voxels of the emitted XRF rays (along the connection between each FL emitting source voxel and each detector point)
        For each detector point (total: n_det), 3 rows of values representing the following values:
            1st row, the index of the FL emitting soruce voxel. The index is the index of the flattened grid of the sample.
            2nd row, the index of the intersecting voxels.
            3rd row, the intersecting length in cm.
            
            
            For example:
                [[0, 0, 0, 0, 0, 0, ..., 0, 1, 1, 1, 1, 0, ..., 0, 2, 2, 2, 0, ..., 0, ......, 0, ...,0]
                                            |_________| \________|
                                                      \          \The remain (diagnal_length_n - 4) spaces are then set to 0
                                                      \4 intersecting voxels from the emitting source at index 1  
                 
                 [5,10,15,20,25, 0, ..., 0, 6,11,16,21, 0, ..., 0, 7,12,17, 0, ..., 0, ......, 0, ...,0]
                                            |_________| \________|
                                                      \          \The remain (diagnal_length_n - 4) spaces are then set to 0
                                                      \4 intersecting voxels at index 6, 11, 16, 21 from the emitting source at index 1  
                 
                 
                 [0.1, 0.1, 0.1, 0.1, 0, 0, ..., 0, 0.2, 0.2, 0.2 ,0.2, 0, ..., 0, 0.3, 0.3, 0.3, 0, ..., 0, ......, 0, ...,0]]
                                                    |_________________| \________|
                                                      \                          \The remain (diagnal_length_n - 4) spaces are then set to 0
                                                      \4 intersecting lengths corresponging to the intersecting voxels in the 2nd row of this tensor
                
            The intersecting number of voxels from each source is not always the same. The maximal possible intersecting number of voxels
            is the number of voxels along the diagnol direction of the sample.
            Therefore, diagnal_length_n spaces are used to store the intersecting voxels for each emitting source.
            In most cases, the number of intersecting voxels for each source voxel is less than diagnal_length_n, The remaining spaces are filled with zeros.
    
    """
    if rank == 0:
        if not os.path.exists(P_folder):
            os.makedirs(P_folder)
        with open(os.path.join(P_folder, 'P_array_parameters.txt'), "w") as P_params:
            P_params.write("det_size_cm = %f\n" %det_size_cm)
            P_params.write("det_from_sample_cm = %f\n" %det_from_sample_cm)
            P_params.write("det_ds_spacing_cm = %f\n" %det_ds_spacing_cm)
            P_params.write("sample_size_n = %f\n" %sample_size_n)
            P_params.write("sample_size_cm = %f\n" %sample_size_cm)
            P_params.write("sample_height_n = %f\n" %sample_height_n)
            
    layers_divisible_by_n_ranks = sample_height_n % n_ranks
    if layers_divisible_by_n_ranks != 0:
        print("Please set n_ranks such that sample_height_n is divisible by n_ranks")
         
    P_save_path = os.path.join(P_folder, f_P)

    ### Calculating voxel size in cm
    voxel_size_cm = sample_size_cm/sample_size_n

    ### Calculating the diameter of the XRF detector with 
    det_size_n = int(np.ceil(det_size_cm/voxel_size_cm)) 

    ### Set the desired spacing between detectorlets, and then convert the unit of spacing to the number of the sample voxels
    det_ds_spacing_n = int(det_ds_spacing_cm/voxel_size_cm)

    # Define position of center of the source voxel (z_s, x_s, y_s), note that it's shifted by 0.5 from the voxel idx to represent the loc of center
    z_s, x_s, y_s = np.indices((int(sample_height_n), int(sample_size_n), int(sample_size_n))) + 0.5
    voxel_pos_ls_flat = np.stack((z_s.flatten(), x_s.flatten(), y_s.flatten()), axis=-1)


    ### Define the location of the detectorlets, the detector is parallel to the yz-plane
    ### The x-posision depends on the distance between the sample and the detecor
    ## x index of the location of the XRF detector
    det_axis_1_idx = sample_size_n + np.ceil(det_from_sample_cm/voxel_size_cm) + 0.5

    ### y, z index of the location of the XRF detector
    ## Define the center of the detector on yz-plane
    det_center_yz = (int(sample_size_n)/2., int(sample_height_n)/2.)

    ## Define the y and z loc(namely the loc along axis 2 and axis 0) of the detectorlets. The y and z loc are confined to be within a circle on the yz plane
    end_det_axis_2_idx_ls = np.array([int((sample_size_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                      int((sample_size_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

    det_axis_2_idx_ls = np.linspace(end_det_axis_2_idx_ls[0], end_det_axis_2_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))

    end_det_axis_0_idx_ls = np.array([int((sample_height_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                      int((sample_height_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

    det_axis_0_idx_ls = np.linspace(end_det_axis_0_idx_ls[0], end_det_axis_0_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))
    ## Create the meshgrid of y and z coordinates and keep only the coordinates within the detector circle
    y_d, z_d = np.meshgrid(det_axis_2_idx_ls, det_axis_0_idx_ls)

    yz_mask = ((y_d - det_center_yz[0])**2 + (z_d - det_center_yz[1])**2 <= (det_size_n/2)**2).flatten()
    y_d_flat, z_d_flat = y_d.flatten()[yz_mask], z_d.flatten()[yz_mask]


    ## The number of x posision needed to fill into the coodinates depends on the number of the y(or z) coodinates within the circle of detector
    x_d_flat = np.full((y_d_flat.shape), det_axis_1_idx)


    det_pos_ls_flat = np.stack((z_d_flat, x_d_flat, y_d_flat), axis=-1)
    n_det = len(det_pos_ls_flat)
    
    if rank == 0:
        print(f"numbder of detecting points: {n_det}")
        sys.stdout.flush()
        
    ## define sample edges: 
    ## sample_x_edge is the edge that is closer to the XRF detector
    ## sample_y_edge has two components representing the left and the right edge
    sample_x_edge = np.array([sample_size_n])
    sample_y_edge = np.array([0, sample_size_n]) 
    sample_z_edge = np.array([0, sample_height_n]) 

    dia_len_n = int(1.2*(sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)
    longest_int_length = 0
    
    n_layers_each_rank = sample_height_n // n_ranks
    voxel_pos_ls_flat_minibatch = voxel_pos_ls_flat[rank * n_layers_each_rank * sample_size_n**2 : (rank+1) * n_layers_each_rank * sample_size_n**2]
    
    f = h5py.File(P_save_path +'.h5', 'w', driver='mpio', comm=comm)
    P = f.create_dataset('P_array', (n_det, 3, dia_len_n * sample_height_n * sample_size_n**2), dtype='f4', data=np.zeros((n_det, 3, dia_len_n * sample_height_n * sample_size_n**2)))
    
    
    j_offset = rank * n_layers_each_rank * sample_size_n**2
    
    stdout_options = {'root':0, 'output_folder': './', 'save_stdout': False, 'print_terminal': True}
    for i,  det_pos in enumerate(det_pos_ls_flat):
        timestr = str(datetime.datetime.today())     
        print_flush_root(rank, val=f"detecting point: {i}, time: {timestr}", output_file='', **stdout_options)
        for j, v in enumerate(voxel_pos_ls_flat_minibatch): 

            # Solving the intersection of the ray with the sample boundary along axis-0
            bdx_int = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_x_edge) # pick the 0th component just because the coordinate is doubly braced

            # Solving the intersection of the ray with the sample boundaries along axis-1 and axis-2, we will get 2 solutions for each axis since there're 2 bdry plane on each axis
            # The desired intersecting point is within the segment(voxel - detectorlet) which is always the one with the larger x coordinate
            bdy_int = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_y_edge)
            if len(bdy_int) != 0:
                bdy_int = np.array([bdy_int[np.argmax(bdy_int[:,1])]])
            else:
                pass


            bdz_int = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_z_edge)
            if len(bdz_int) != 0:
                bdz_int = np.array([bdz_int[np.argmax(bdz_int[:,1])]])
            else:
                pass

            # Pick the intersecting point that first hit the boundary plan. This point is with the least x value among the 3 intersections.
            bd_int_ls = np.concatenate((bdz_int, bdx_int, bdy_int))
            bd_int = np.clip(np.abs((bd_int_ls[np.argmin(bd_int_ls[:,1])])), 0, sample_size_n)


            # when the beam intersects with a voxel, it either intersects with the x or y or z boundary plane of the voxel
            # find the x,y,z-value of the voxel boundary except the ones on the sample edge

            z_edge_ls = np.where(bd_int[0] > v[0], np.linspace(np.ceil(bd_int[0])-1, np.ceil(v[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))),
                                                   np.linspace(np.ceil(v[0])-1, np.ceil(bd_int[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))))

            x_edge_ls = np.where(bd_int[1] > v[1], np.linspace(np.ceil(bd_int[1])-1, np.ceil(v[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))),
                                                   np.linspace(np.ceil(v[1])-1, np.ceil(bd_int[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))))

            y_edge_ls = np.where(bd_int[2] > v[2], np.linspace(np.ceil(bd_int[2])-1, np.ceil(v[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))),
                                                   np.linspace(np.ceil(v[2])-1, np.ceil(bd_int[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))))


            z_edge_int_ls = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], z_edge_ls)
            x_edge_int_ls = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], x_edge_ls)
            y_edge_int_ls = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], y_edge_ls)

            # Collect all intersecting points and sort all intersections using the x coordinate
            int_ls = np.concatenate((x_edge_int_ls, y_edge_int_ls, z_edge_int_ls, np.array(bd_int)[np.newaxis,:]))     
            int_ls = int_ls[np.argsort(int_ls[:,1])]

            # calculate the intersecting length in the intersecting voxels
            int_length = np.sqrt(np.diff(int_ls[:,0])**2 + np.diff(int_ls[:,1])**2 + np.diff(int_ls[:,2])**2)
            # just in case that we count some intersections twice, delete the duplicates
            idx_duplicate = np.array(np.where(int_length==0)).flatten()
            int_ls = np.delete(int_ls, idx_duplicate, 0)
            int_length = np.delete(int_length, idx_duplicate) 

            # determine the indices of the intersecting voxels according to the intersecting x,y,z-coordinates
            int_ls_shift = np.zeros((int_ls.shape))
            int_ls_shift[1:] = int_ls[:-1]
            int_idx = np.floor((int_ls + int_ls_shift)/2)[1:]
#                 int_idx = (int_idx[:,0].astype('int'), int_idx[:,1].astype('int'), int_idx[:,2].astype('int'))
            int_idx_flat = int_idx[:,0] * (sample_size_n * sample_size_n) + int_idx[:,1] * sample_size_n + int_idx[:,2]

            if len(int_idx_flat) > longest_int_length:
                longest_int_length = len(int_idx_flat)
                
            P[i, 0, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = j_offset+j
            P[i, 1, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_idx_flat)
            P[i, 2, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_length * voxel_size_cm)            

    
#     f_short = h5py.File(P_save_path +'_short.h5', 'w', driver='mpio', comm=comm)
#     P_short = f_short.create_dataset('P_short_array', (n_det, 3, longest_int_length * sample_height_n * sample_size_n**2), dtype='f4')
        
#     for j, v in enumerate(voxel_pos_ls_flat_minibatch):
#         P_short[:,:,(j_offset+j) * longest_int_length: (j_offset+j+1) * longest_int_length] = \
#         P[:,:, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + longest_int_length]
        
    f.close()
#     f_short.close()
    
    return None


def intersecting_length_fl_detectorlet_3d_mpi_write_h5_2(n_ranks, minibatch_size, rank, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n, sample_size_cm, sample_height_n, P_folder, f_P):
    
 
    if rank == 0:
        if not os.path.exists(P_folder):
            os.makedirs(P_folder)
        with open(os.path.join(P_folder, 'P_array_parameters.txt'), "w") as P_params:
            P_params.write("det_size_cm = %f\n" %det_size_cm)
            P_params.write("det_from_sample_cm = %f\n" %det_from_sample_cm)
            P_params.write("det_ds_spacing_cm = %f\n" %det_ds_spacing_cm)
            P_params.write("sample_size_n = %f\n" %sample_size_n)
            P_params.write("sample_size_cm = %f\n" %sample_size_cm)
            P_params.write("sample_height_n = %f\n" %sample_height_n)
            
         
    P_save_path = os.path.join(P_folder, f_P)

    ### Calculating voxel size in cm
    voxel_size_cm = sample_size_cm/sample_size_n

    ### Calculating the diameter of the XRF detector with 
    det_size_n = int(np.ceil(det_size_cm/voxel_size_cm)) 

    ### Set the desired spacing between detectorlets, and then convert the unit of spacing to the number of the sample voxels
    det_ds_spacing_n = int(det_ds_spacing_cm/voxel_size_cm)

    # Define position of center of the source voxel (z_s, x_s, y_s), note that it's shifted by 0.5 from the voxel idx to represent the loc of center
    z_s, x_s, y_s = np.indices((int(sample_height_n), int(sample_size_n), int(sample_size_n))) + 0.5
    voxel_pos_ls_flat = np.stack((z_s.flatten(), x_s.flatten(), y_s.flatten()), axis=-1)


    ### Define the location of the detectorlets, the detector is parallel to the yz-plane
    ### The x-posision depends on the distance between the sample and the detecor
    ## x index of the location of the XRF detector
    det_axis_1_idx = sample_size_n + np.ceil(det_from_sample_cm/voxel_size_cm) + 0.5

    ### y, z index of the location of the XRF detector
    ## Define the center of the detector on yz-plane
    det_center_yz = (int(sample_size_n)/2., int(sample_height_n)/2.)

    ## Define the y and z loc(namely the loc along axis 2 and axis 0) of the detectorlets. The y and z loc are confined to be within a circle on the yz plane
    end_det_axis_2_idx_ls = np.array([int((sample_size_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                      int((sample_size_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

    det_axis_2_idx_ls = np.linspace(end_det_axis_2_idx_ls[0], end_det_axis_2_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))

    end_det_axis_0_idx_ls = np.array([int((sample_height_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                      int((sample_height_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

    det_axis_0_idx_ls = np.linspace(end_det_axis_0_idx_ls[0], end_det_axis_0_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))
    ## Create the meshgrid of y and z coordinates and keep only the coordinates within the detector circle
    y_d, z_d = np.meshgrid(det_axis_2_idx_ls, det_axis_0_idx_ls)

    yz_mask = ((y_d - det_center_yz[0])**2 + (z_d - det_center_yz[1])**2 <= (det_size_n/2)**2).flatten()
    y_d_flat, z_d_flat = y_d.flatten()[yz_mask], z_d.flatten()[yz_mask]


    ## The number of x posision needed to fill into the coodinates depends on the number of the y(or z) coodinates within the circle of detector
    x_d_flat = np.full((y_d_flat.shape), det_axis_1_idx)


    det_pos_ls_flat = np.stack((z_d_flat, x_d_flat, y_d_flat), axis=-1)
    n_det = len(det_pos_ls_flat)
    
    if rank == 0:
        print(f"numbder of detecting points: {n_det}")
        sys.stdout.flush()
        
    ## define sample edges: 
    ## sample_x_edge is the edge that is closer to the XRF detector
    ## sample_y_edge has two components representing the left and the right edge
    sample_x_edge = np.array([sample_size_n])
    sample_y_edge = np.array([0, sample_size_n]) 
    sample_z_edge = np.array([0, sample_height_n]) 

    dia_len_n = int(1.2*(sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)
    longest_int_length = 0
    
    minibatch_ls_0 = np.arange(n_ranks)
    n_batch = (sample_height_n * sample_size_n) // (n_ranks * minibatch_size)
    
    f = h5py.File(P_save_path +'.h5', 'w', driver='mpio', comm=comm)
    P = f.create_dataset('P_array', (n_det, 3, dia_len_n * sample_height_n * sample_size_n**2), dtype='f4', data=np.zeros((n_det, 3, dia_len_n * sample_height_n * sample_size_n**2)))
    
    stdout_options = {'root':0, 'output_folder': './', 'save_stdout': False, 'print_terminal': True}
    for i,  det_pos in enumerate(det_pos_ls_flat):
        timestr = str(datetime.datetime.today())     
        print_flush_root(rank, val=f"detecting point: {i}, time: {timestr}", output_file='', **stdout_options)
        
        for m in range(n_batch):
            minibatch_ls = n_ranks * m + minibatch_ls_0
            p = minibatch_ls[rank]
            j_offset = p * minibatch_size * sample_size_n
            
            voxel_pos_ls_flat_minibatch = voxel_pos_ls_flat[p * minibatch_size * sample_size_n: (p+1) * minibatch_size * sample_size_n]
            
            for j, v in enumerate(voxel_pos_ls_flat_minibatch): 

                # Solving the intersection of the ray with the sample boundary along axis-0
                bdx_int = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_x_edge)

                # Solving the intersection of the ray with the sample boundaries along axis-1 and axis-2, we will get 2 solutions for each axis since there're 2 bdry plane on each axis
                # The desired intersecting point is within the segment(voxel - detectorlet) which is always the one with the larger x coordinate
                bdy_int = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_y_edge)
                if len(bdy_int) != 0:
                    bdy_int = np.array([bdy_int[np.argmax(bdy_int[:,1])]])
                else:
                    pass


                bdz_int = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_z_edge)
                if len(bdz_int) != 0:
                    bdz_int = np.array([bdz_int[np.argmax(bdz_int[:,1])]])
                else:
                    pass

                # Pick the intersecting point that first hit the boundary plan. This point is with the least x value among the 3 intersections.
                bd_int_ls = np.concatenate((bdz_int, bdx_int, bdy_int))
                bd_int = np.clip(np.abs((bd_int_ls[np.argmin(bd_int_ls[:,1])])), 0, sample_size_n)


                # when the beam intersects with a voxel, it either intersects with the x or y or z boundary plane of the voxel
                # find the x,y,z-value of the voxel boundary except the ones on the sample edge

                z_edge_ls = np.where(bd_int[0] > v[0], np.linspace(np.ceil(bd_int[0])-1, np.ceil(v[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))),
                                                       np.linspace(np.ceil(v[0])-1, np.ceil(bd_int[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))))

                x_edge_ls = np.where(bd_int[1] > v[1], np.linspace(np.ceil(bd_int[1])-1, np.ceil(v[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))),
                                                       np.linspace(np.ceil(v[1])-1, np.ceil(bd_int[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))))

                y_edge_ls = np.where(bd_int[2] > v[2], np.linspace(np.ceil(bd_int[2])-1, np.ceil(v[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))),
                                                       np.linspace(np.ceil(v[2])-1, np.ceil(bd_int[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))))


                z_edge_int_ls = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], z_edge_ls)
                x_edge_int_ls = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], x_edge_ls)
                y_edge_int_ls = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], y_edge_ls)

                # Collect all intersecting points and sort all intersections using the x coordinate
                int_ls = np.concatenate((x_edge_int_ls, y_edge_int_ls, z_edge_int_ls, np.array(bd_int)[np.newaxis,:]))     
                int_ls = int_ls[np.argsort(int_ls[:,1])]

                # calculate the intersecting length in the intersecting voxels
                int_length = np.sqrt(np.diff(int_ls[:,0])**2 + np.diff(int_ls[:,1])**2 + np.diff(int_ls[:,2])**2)
                # just in case that we count some intersections twice, delete the duplicates
                idx_duplicate = np.array(np.where(int_length==0)).flatten()
                int_ls = np.delete(int_ls, idx_duplicate, 0)
                int_length = np.delete(int_length, idx_duplicate) 

                # determine the indices of the intersecting voxels according to the intersecting x,y,z-coordinates
                int_ls_shift = np.zeros((int_ls.shape))
                int_ls_shift[1:] = int_ls[:-1]
                int_idx = np.floor((int_ls + int_ls_shift)/2)[1:]
        #                 int_idx = (int_idx[:,0].astype('int'), int_idx[:,1].astype('int'), int_idx[:,2].astype('int'))
                int_idx_flat = int_idx[:,0] * (sample_size_n * sample_size_n) + int_idx[:,1] * sample_size_n + int_idx[:,2]

                if len(int_idx_flat) > longest_int_length:
                    longest_int_length = len(int_idx_flat)

                P[i, 0, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = j_offset+j
                P[i, 1, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_idx_flat)
                P[i, 2, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_length * voxel_size_cm)            

    
#     f_short = h5py.File(P_save_path +'_short.h5', 'w', driver='mpio', comm=comm)
#     P_short = f_short.create_dataset('P_short_array', (n_det, 3, longest_int_length * sample_height_n * sample_size_n**2), dtype='f4')
        
#     for j, v in enumerate(voxel_pos_ls_flat_minibatch):
#         P_short[:,:,(j_offset+j) * longest_int_length: (j_offset+j+1) * longest_int_length] = \
#         P[:,:, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + longest_int_length]
        
    f.close()
#     f_short.close()
    
    return None


def intersecting_length_fl_detectorlet_3d_mpi_write_h5_3(n_ranks, minibatch_size, rank, det_on_which_side, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n, sample_size_cm, sample_height_n, P_folder, f_P):
    
 
    if rank == 0:
        if not os.path.exists(P_folder):
            os.makedirs(P_folder)
        with open(os.path.join(P_folder, 'P_array_parameters.txt'), "w") as P_params:
            P_params.write("det_size_cm = %f\n" %det_size_cm)
            P_params.write("det_from_sample_cm = %f\n" %det_from_sample_cm)
            P_params.write("det_ds_spacing_cm = %f\n" %det_ds_spacing_cm)
            P_params.write("sample_size_n = %f\n" %sample_size_n)
            P_params.write("sample_size_cm = %f\n" %sample_size_cm)
            P_params.write("sample_height_n = %f\n" %sample_height_n)
            
         
    P_save_path = os.path.join(P_folder, f_P)

    ### Calculating voxel size in cm
    voxel_size_cm = sample_size_cm/sample_size_n

    ### Calculating the diameter of the XRF detector with 
    det_size_n = int(np.ceil(det_size_cm/voxel_size_cm)) 

    ### Set the desired spacing between detectorlets, and then convert the unit of spacing to the number of the sample voxels
    det_ds_spacing_n = int(det_ds_spacing_cm/voxel_size_cm)

    # Define position of center of the source voxel (z_s, x_s, y_s), note that it's shifted by 0.5 from the voxel idx to represent the loc of center
    z_s, x_s, y_s = np.indices((int(sample_height_n), int(sample_size_n), int(sample_size_n))) + 0.5
    voxel_pos_ls_flat = np.stack((z_s.flatten(), x_s.flatten(), y_s.flatten()), axis=-1)


    ### Define the location of the detectorlets, the detector is parallel to the yz-plane
    ### The x-posision depends on the distance between the sample and the detecor
    ## x index of the location of the XRF detector
    
    ## define sample edges: 
    ## sample_x_edge is the edge that is closer to the XRF detector    
    if det_on_which_side == "positive":
        det_axis_1_idx = sample_size_n + np.ceil(det_from_sample_cm/voxel_size_cm) + 0.5
        sample_x_edge = np.array([sample_size_n])
    
    else:
        det_axis_1_idx = - np.ceil(det_from_sample_cm/voxel_size_cm) + 0.5
        sample_x_edge = np.array([0.])

    ### y, z index of the location of the XRF detector
    ## Define the center of the detector on yz-plane
    det_center_yz = (int(sample_size_n)/2., int(sample_height_n)/2.)

    ## Define the y and z loc(namely the loc along axis 2 and axis 0) of the detectorlets. The y and z loc are confined to be within a circle on the yz plane
    end_det_axis_2_idx_ls = np.array([int((sample_size_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                      int((sample_size_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

    det_axis_2_idx_ls = np.linspace(end_det_axis_2_idx_ls[0], end_det_axis_2_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))

    end_det_axis_0_idx_ls = np.array([int((sample_height_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                      int((sample_height_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

    det_axis_0_idx_ls = np.linspace(end_det_axis_0_idx_ls[0], end_det_axis_0_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))
    ## Create the meshgrid of y and z coordinates and keep only the coordinates within the detector circle
    y_d, z_d = np.meshgrid(det_axis_2_idx_ls, det_axis_0_idx_ls)

    yz_mask = ((y_d - det_center_yz[0])**2 + (z_d - det_center_yz[1])**2 <= (det_size_n/2)**2).flatten()
    y_d_flat, z_d_flat = y_d.flatten()[yz_mask], z_d.flatten()[yz_mask]


    ## The number of x posision needed to fill into the coodinates depends on the number of the y(or z) coodinates within the circle of detector
    x_d_flat = np.full((y_d_flat.shape), det_axis_1_idx)


    det_pos_ls_flat = np.stack((z_d_flat, x_d_flat, y_d_flat), axis=-1)
    n_det = len(det_pos_ls_flat)
    
    if rank == 0:
        print(f"numbder of detecting points: {n_det}")
        sys.stdout.flush()
        
    ## define sample edges: 
    ## sample_y_edge has two components representing the left and the right edge
    sample_y_edge = np.array([0, sample_size_n]) 
    sample_z_edge = np.array([0, sample_height_n]) 

    dia_len_n = int(1.2*(sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)
    longest_int_length = 0
    
    minibatch_ls_0 = np.arange(n_ranks)
    n_batch = (sample_height_n * sample_size_n) // (n_ranks * minibatch_size)
    
    f = h5py.File(P_save_path +'.h5', 'w', driver='mpio', comm=comm)
    P = f.create_dataset('P_array', (n_det, 3, dia_len_n * sample_height_n * sample_size_n**2), dtype='f4', data=np.zeros((n_det, 3, dia_len_n * sample_height_n * sample_size_n**2)))
    
    stdout_options = {'root':0, 'output_folder': './', 'save_stdout': False, 'print_terminal': True}
    for i,  det_pos in enumerate(det_pos_ls_flat):
        
        timestr = str(datetime.datetime.today())     
        print_flush_root(rank, val=f"detecting point: {i}, time: {timestr}", output_file='', **stdout_options)
        
        for m in range(n_batch):
            minibatch_ls = n_ranks * m + minibatch_ls_0
            p = minibatch_ls[rank]
            j_offset = p * minibatch_size * sample_size_n
            
            voxel_pos_ls_flat_minibatch = voxel_pos_ls_flat[p * minibatch_size * sample_size_n: (p+1) * minibatch_size * sample_size_n] 
            
            for j, v in enumerate(voxel_pos_ls_flat_minibatch): 
                
                # Solving the intersection of the ray with the sample boundary along axis-0
                bdx_int = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_x_edge)

                # Solving the intersection of the ray with the sample boundaries along axis-1 and axis-2, we will get 2 solutions for each axis since there're 2 bdry plane on each axis
                # The desired intersecting point is within the segment(voxel - detectorlet) which is always the one with the larger x coordinate
                bdy_int = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_y_edge)
                if len(bdy_int) != 0:
                    if det_on_which_side == "positive": 
                        bdy_int = np.array([bdy_int[np.argmax(bdy_int[:,1])]])
                    else:
                        bdy_int = np.array([bdy_int[np.argmin(bdy_int[:,1])]])
                        
                else:
                    pass


                bdz_int = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_z_edge)
                if len(bdz_int) != 0:
                    if det_on_which_side == "positive": 
                        bdz_int = np.array([bdz_int[np.argmax(bdz_int[:,1])]])
                    else:
                        bdz_int = np.array([bdz_int[np.argmin(bdz_int[:,1])]])
                        
                else:
                    pass

                # Pick the intersecting point that first hit the boundary plan. This point is with the least(greatest) x value among the 3 intersections
                # if the detector is at positive(negative) x 
                bd_int_ls = np.concatenate((bdz_int, bdx_int, bdy_int))
                
                if det_on_which_side == "positive":
                    bd_int = np.clip(np.abs(bd_int_ls[np.argmin(bd_int_ls[:,1])]), 0, sample_size_n)
                    
                else:                  
                    bd_int = np.clip(np.abs(bd_int_ls[np.argmax(bd_int_ls[:,1])]), 0, sample_size_n)

                
                # when the beam intersects with a voxel, it either intersects with the x or y or z boundary plane of the voxel
                # find the x,y,z-value of the voxel boundary except the ones on the sample edge
                
                z_edge_ls = np.where(bd_int[0] > v[0], np.linspace(np.ceil(bd_int[0])-1, np.ceil(v[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))),
                                                       np.linspace(np.ceil(v[0])-1, np.ceil(bd_int[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))))

                x_edge_ls = np.where(bd_int[1] > v[1], np.linspace(np.ceil(bd_int[1])-1, np.ceil(v[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))),
                                                       np.linspace(np.ceil(v[1])-1, np.ceil(bd_int[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))))

                y_edge_ls = np.where(bd_int[2] > v[2], np.linspace(np.ceil(bd_int[2])-1, np.ceil(v[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))),
                                                       np.linspace(np.ceil(v[2])-1, np.ceil(bd_int[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))))

                
                z_edge_int_ls = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], z_edge_ls)
                x_edge_int_ls = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], x_edge_ls)
                y_edge_int_ls = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], y_edge_ls)
                

                # Collect all intersecting points and sort all intersections using the x coordinate
                int_ls = np.concatenate((x_edge_int_ls, y_edge_int_ls, z_edge_int_ls, np.array(bd_int)[np.newaxis,:]))
                int_ls = int_ls[np.argsort(int_ls[:,1])]

                # calculate the intersecting length in the intersecting voxels
                int_length = np.sqrt(np.diff(int_ls[:,0])**2 + np.diff(int_ls[:,1])**2 + np.diff(int_ls[:,2])**2)
                # just in case that we count some intersections twice, delete the duplicates
                idx_duplicate = np.array(np.where(int_length==0)).flatten()
                int_ls = np.delete(int_ls, idx_duplicate, 0)
                int_length = np.delete(int_length, idx_duplicate) 
             
                # determine the indices of the intersecting voxels according to the intersecting x,y,z-coordinates
                int_ls_shift = np.zeros((int_ls.shape))
                int_ls_shift[1:] = int_ls[:-1]
                int_idx = np.floor((int_ls + int_ls_shift)/2)[1:]
        #                 int_idx = (int_idx[:,0].astype('int'), int_idx[:,1].astype('int'), int_idx[:,2].astype('int'))
#                 int_idx_flat = int_idx[:,0] * (sample_height_n * sample_size_n) + int_idx[:,1] * sample_size_n + int_idx[:,2]
                int_idx_flat = int_idx[:,0] * (sample_size_n * sample_size_n) + int_idx[:,1] * sample_size_n + int_idx[:,2]
                    
#                 if len(int_idx_flat) > longest_int_length:
#                     longest_int_length = len(int_idx_flat)
                
                P[i, 0, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = j_offset+j
                P[i, 1, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_idx_flat)
                P[i, 2, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_length * voxel_size_cm)            

    
#     f_short = h5py.File(P_save_path +'_short.h5', 'w', driver='mpio', comm=comm)
#     P_short = f_short.create_dataset('P_short_array', (n_det, 3, longest_int_length * sample_height_n * sample_size_n**2), dtype='f4')
        
#     for j, v in enumerate(voxel_pos_ls_flat_minibatch):
#         P_short[:,:,(j_offset+j) * longest_int_length: (j_offset+j+1) * longest_int_length] = \
#         P[:,:, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + longest_int_length]
        
    f.close()
#     f_short.close()
    
    return None

def intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual(n_ranks, minibatch_size, rank, manual_det_coord, set_det_coord_cm, det_on_which_side,
                                                                manual_det_area, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n, 
                                                                sample_size_cm, sample_height_n, P_folder, f_P):
    
 
    if rank == 0:
        if not os.path.exists(P_folder):
            os.makedirs(P_folder)
        with open(os.path.join(P_folder, 'P_array_parameters.txt'), "w") as P_params:             
            
            if not manual_det_area:
                P_params.write("det_size_cm = %f\n" %det_size_cm)    
            
            if manual_det_coord:
                P_params.write("det_coord_cm = ")
                P_params.write(str(set_det_coord_cm))
                P_params.write("\n")
            else:
                P_params.write("det_from_sample_cm = %f\n" %det_from_sample_cm)
                P_params.write("det_ds_spacing_cm = %f\n" %det_ds_spacing_cm) 
                
            P_params.write("sample_size_n = %f\n" %sample_size_n)
            P_params.write("sample_size_cm = %f\n" %sample_size_cm)
            P_params.write("sample_height_n = %f\n" %sample_height_n)
            
         
    P_save_path = os.path.join(P_folder, f_P)

    ### Calculating voxel size in cm
    voxel_size_cm = sample_size_cm/sample_size_n


    if not manual_det_area:
        ### Calculating the diameter of the XRF detector in # of voxels
        det_size_n = int(np.ceil(det_size_cm/voxel_size_cm)) 

    # Define position of center of the source voxel (z_s, x_s, y_s), note that it's shifted by 0.5 from the voxel idx to represent the loc of center
    z_s, x_s, y_s = np.indices((int(sample_height_n), int(sample_size_n), int(sample_size_n))) + 0.5
    voxel_pos_ls_flat = np.stack((z_s.flatten(), x_s.flatten(), y_s.flatten()), axis=-1)


    ### Define the location of the detectorlets, the detector is parallel to the yz-plane
    ### The x-posision depends on the distance between the sample and the detecor
    ## x index of the location of the XRF detector
    
    ## define sample edges: 
    ## sample_x_edge is the edge that is closer to the XRF detector    

    if manual_det_coord:
        det_pos_ls_flat = np.round(set_det_coord_cm / voxel_size_cm + sample_size_n/2) + 0.5 
        if set_det_coord_cm[0,1] > 0:
            det_on_which_side == "positive"
            sample_x_edge = np.array([sample_size_n])
        else:
            det_on_which_side == "negative"
            sample_x_edge = np.array([0.])
    
    else:
        
        ### Set the desired spacing between detectorlets, and then convert the unit of spacing to the number of the sample voxels
        det_ds_spacing_n = int(det_ds_spacing_cm/voxel_size_cm)
        
        if det_on_which_side == "positive":
            det_axis_1_idx = sample_size_n + np.ceil(det_from_sample_cm/voxel_size_cm) + 0.5
            sample_x_edge = np.array([sample_size_n])

        else:
            det_axis_1_idx = - np.ceil(det_from_sample_cm/voxel_size_cm) + 0.5
            sample_x_edge = np.array([0.])
        
        ### y, z index of the location of the XRF detector
        ## Define the center of the detector on yz-plane
        det_center_yz = (int(sample_size_n)/2., int(sample_height_n)/2.)

        ## Define the y and z loc(namely the loc along axis 2 and axis 0) of the detectorlets. The y and z loc are confined to be within a circle on the yz plane
        end_det_axis_2_idx_ls = np.array([int((sample_size_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                          int((sample_size_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

        det_axis_2_idx_ls = np.linspace(end_det_axis_2_idx_ls[0], end_det_axis_2_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))

        end_det_axis_0_idx_ls = np.array([int((sample_height_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                          int((sample_height_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

        det_axis_0_idx_ls = np.linspace(end_det_axis_0_idx_ls[0], end_det_axis_0_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))
        ## Create the meshgrid of y and z coordinates and keep only the coordinates within the detector circle
        y_d, z_d = np.meshgrid(det_axis_2_idx_ls, det_axis_0_idx_ls)

        yz_mask = ((y_d - det_center_yz[0])**2 + (z_d - det_center_yz[1])**2 <= (det_size_n/2)**2).flatten()
        y_d_flat, z_d_flat = y_d.flatten()[yz_mask], z_d.flatten()[yz_mask]


        ## The number of x posision needed to fill into the coodinates depends on the number of the y(or z) coodinates within the circle of detector
        x_d_flat = np.full((y_d_flat.shape), det_axis_1_idx)

        det_pos_ls_flat = np.stack((z_d_flat, x_d_flat, y_d_flat), axis=-1)
    
    n_det = len(det_pos_ls_flat)
    
    if rank == 0:
        print(det_pos_ls_flat.shape)
        print(f"numbder of detecting points: {n_det}")
        sys.stdout.flush()
        
    ## define sample edges: 
    ## sample_y_edge has two components representing the left and the right edge
    sample_y_edge = np.array([0, sample_size_n]) 
    sample_z_edge = np.array([0, sample_height_n]) 

    dia_len_n = int(1.2*(sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)
    longest_int_length = 0
    
    minibatch_ls_0 = np.arange(n_ranks)
    n_batch = (sample_height_n * sample_size_n) // (n_ranks * minibatch_size)
    
    f = h5py.File(P_save_path +'.h5', 'w', driver='mpio', comm=comm)
    P = f.create_dataset('P_array', (n_det, 3, dia_len_n * sample_height_n * sample_size_n**2), dtype='f4')
    
    stdout_options = {'root':0, 'output_folder': './', 'save_stdout': False, 'print_terminal': True}
    for i,  det_pos in enumerate(det_pos_ls_flat):
        
        timestr = str(datetime.datetime.today())     
        print_flush_root(rank, val=f"detecting point: {i}, time: {timestr}", output_file='', **stdout_options)
        
        for m in range(n_batch):
            minibatch_ls = n_ranks * m + minibatch_ls_0
            p = minibatch_ls[rank]
            j_offset = p * minibatch_size * sample_size_n
            
            voxel_pos_ls_flat_minibatch = voxel_pos_ls_flat[p * minibatch_size * sample_size_n: (p+1) * minibatch_size * sample_size_n] 
            
            for j, v in enumerate(voxel_pos_ls_flat_minibatch): 
                
                # Solving the intersection of the ray with the sample boundary along axis-0
                bdx_int = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_x_edge)

                # Solving the intersection of the ray with the sample boundaries along axis-1 and axis-2, we will get 2 solutions for each axis since there're 2 bdry plane on each axis
                # The desired intersecting point is within the segment(voxel - detectorlet) which is always the one with the larger x coordinate
                bdy_int = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_y_edge)
                if len(bdy_int) != 0:
                    if det_on_which_side == "positive": 
                        bdy_int = np.array([bdy_int[np.argmax(bdy_int[:,1])]])
                    else:
                        bdy_int = np.array([bdy_int[np.argmin(bdy_int[:,1])]])
                        
                else:
                    pass


                bdz_int = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_z_edge)
                if len(bdz_int) != 0:
                    if det_on_which_side == "positive": 
                        bdz_int = np.array([bdz_int[np.argmax(bdz_int[:,1])]])
                    else:
                        bdz_int = np.array([bdz_int[np.argmin(bdz_int[:,1])]])
                        
                else:
                    pass

                # Pick the intersecting point that first hit the boundary plan. This point is with the least(greatest) x value among the 3 intersections
                # if the detector is at positive(negative) x 
                bd_int_ls = np.concatenate((bdz_int, bdx_int, bdy_int))
                
                if det_on_which_side == "positive":
                    bd_int = np.clip(np.abs(bd_int_ls[np.argmin(bd_int_ls[:,1])]), 0, sample_size_n)
                    
                else:                  
                    bd_int = np.clip(np.abs(bd_int_ls[np.argmax(bd_int_ls[:,1])]), 0, sample_size_n)

                
                # when the beam intersects with a voxel, it either intersects with the x or y or z boundary plane of the voxel
                # find the x,y,z-value of the voxel boundary except the ones on the sample edge
                
                z_edge_ls = np.where(bd_int[0] > v[0], np.linspace(np.ceil(bd_int[0])-1, np.ceil(v[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))),
                                                       np.linspace(np.ceil(v[0])-1, np.ceil(bd_int[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))))

                x_edge_ls = np.where(bd_int[1] > v[1], np.linspace(np.ceil(bd_int[1])-1, np.ceil(v[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))),
                                                       np.linspace(np.ceil(v[1])-1, np.ceil(bd_int[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))))

                y_edge_ls = np.where(bd_int[2] > v[2], np.linspace(np.ceil(bd_int[2])-1, np.ceil(v[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))),
                                                       np.linspace(np.ceil(v[2])-1, np.ceil(bd_int[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))))

                
                z_edge_int_ls = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], z_edge_ls)
                x_edge_int_ls = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], x_edge_ls)
                y_edge_int_ls = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], y_edge_ls)
                

                # Collect all intersecting points and sort all intersections using the x coordinate
                int_ls = np.concatenate((x_edge_int_ls, y_edge_int_ls, z_edge_int_ls, np.array(bd_int)[np.newaxis,:]))
                int_ls = int_ls[np.argsort(int_ls[:,1])]

                # calculate the intersecting length in the intersecting voxels
                int_length = np.sqrt(np.diff(int_ls[:,0])**2 + np.diff(int_ls[:,1])**2 + np.diff(int_ls[:,2])**2)
                # just in case that we count some intersections twice, delete the duplicates
                idx_duplicate = np.array(np.where(int_length==0)).flatten()
                int_ls = np.delete(int_ls, idx_duplicate, 0)
                int_length = np.delete(int_length, idx_duplicate) 
             
                # determine the indices of the intersecting voxels according to the intersecting x,y,z-coordinates
                int_ls_shift = np.zeros((int_ls.shape))
                int_ls_shift[1:] = int_ls[:-1]
                int_idx = np.floor((int_ls + int_ls_shift)/2)[1:]
        #                 int_idx = (int_idx[:,0].astype('int'), int_idx[:,1].astype('int'), int_idx[:,2].astype('int'))
#                 int_idx_flat = int_idx[:,0] * (sample_height_n * sample_size_n) + int_idx[:,1] * sample_size_n + int_idx[:,2]
                int_idx_flat = int_idx[:,0] * (sample_size_n * sample_size_n) + int_idx[:,1] * sample_size_n + int_idx[:,2]
                    
#                 if len(int_idx_flat) > longest_int_length:
#                     longest_int_length = len(int_idx_flat)
                
                P[i, 0, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = j_offset+j
                P[i, 1, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_idx_flat)
                P[i, 2, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + len(int_idx_flat)] = np.array(int_length * voxel_size_cm)            

    
#     f_short = h5py.File(P_save_path +'_short.h5', 'w', driver='mpio', comm=comm)
#     P_short = f_short.create_dataset('P_short_array', (n_det, 3, longest_int_length * sample_height_n * sample_size_n**2), dtype='f4')
        
#     for j, v in enumerate(voxel_pos_ls_flat_minibatch):
#         P_short[:,:,(j_offset+j) * longest_int_length: (j_offset+j+1) * longest_int_length] = \
#         P[:,:, (j_offset+j) * dia_len_n: (j_offset+j) * dia_len_n + longest_int_length]
        
    f.close()
#     f_short.close()
    
    return None


def intersecting_length_fl_detectorlet_3d(det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n, sample_size_cm, sample_height_n, P_folder, f_P):
    """
    Parameters
    ----------
    det_size_cm : float
        The diameter of the circle to distribute the detector points
        
    det_from_sample_cm : float
        The distance between the detector plane and the sample boundary plane
    
    det_ds_spacing_cm : float
        The spacing between detector points
    
    sample_size_n: int scalar
        sample size in number of pixles on the side along the probe propagation axis
    
    sample_size_cm: scalar
        sample size in cm on the side along the probe propagation axis
        
    sample_height_n : integer
        The height of the sample along the rotational axis (in number of pixels)
        
    P_save_path : string
        The path that saves the tensor P

    Returns
    -------
    n_det : integer
        The number of the detector points within the circle with the diatmeter, det_size_cm.
    
    P : torch tensor
        a tensor with the dimension (n_det, 3, n_voxels *  diagnal_length_n)
        n_voxels: the number of voxels of the sample.
        diagnal_length_n: the number of voxels along the diagnol direction of the sample
        
        P tensor contains the information of intersecting voxels of the emitted XRF rays (along the connection between each FL emitting source voxel and each detector point)
        For each detector point (total: n_det), 3 rows of values representing the following values:
            1st row, the index of the FL emitting soruce voxel. The index is the index of the flattened grid of the sample.
            2nd row, the index of the intersecting voxels.
            3rd row, the intersecting length in cm.
            
            
            For example:
                [[0, 0, 0, 0, 0, 0, ..., 0, 1, 1, 1, 1, 0, ..., 0, 2, 2, 2, 0, ..., 0, ......, 0, ...,0]
                                            |_________| \________|
                                                      \          \The remain (diagnal_length_n - 4) spaces are then set to 0
                                                      \4 intersecting voxels from the emitting source at index 1  
                 
                 [5,10,15,20,25, 0, ..., 0, 6,11,16,21, 0, ..., 0, 7,12,17, 0, ..., 0, ......, 0, ...,0]
                                            |_________| \________|
                                                      \          \The remain (diagnal_length_n - 4) spaces are then set to 0
                                                      \4 intersecting voxels at index 6, 11, 16, 21 from the emitting source at index 1  
                 
                 
                 [0.1, 0.1, 0.1, 0.1, 0, 0, ..., 0, 0.2, 0.2, 0.2 ,0.2, 0, ..., 0, 0.3, 0.3, 0.3, 0, ..., 0, ......, 0, ...,0]]
                                                    |_________________| \________|
                                                      \                          \The remain (diagnal_length_n - 4) spaces are then set to 0
                                                      \4 intersecting lengths corresponging to the intersecting voxels in the 2nd row of this tensor
                
            The intersecting number of voxels from each source is not always the same. The maximal possible intersecting number of voxels
            is the number of voxels along the diagnol direction of the sample.
            Therefore, diagnal_length_n spaces are used to store the intersecting voxels for each emitting source.
            In most cases, the number of intersecting voxels for each source voxel is less than diagnal_length_n, The remaining spaces are filled with zeros.
    
    """

    P_save_path = os.path.join(P_folder, f_P) #string 
    
    if os.path.isfile(P_save_path + ".npy"):
        P = np.load(P_save_path + ".npy")
        n_det = P.shape[0]
        longest_int_length_n = P.shape[2]//(sample_height_n * sample_size_n**2)
        print(f"numbder of detecting points: {n_det}")
        sys.stdout.flush()
        
    
    else:
        if not os.path.exists(P_folder):
            os.makedirs(P_folder)

        ### Calculating voxel size in cm
        voxel_size_cm = sample_size_cm/sample_size_n

        ### Calculating the diameter of the XRF detector with 
        det_size_n = int(np.ceil(det_size_cm/voxel_size_cm)) 

        ### Set the desired spacing between detectorlets, and then convert the unit of spacing to the number of the sample voxels
        det_ds_spacing_n = int(det_ds_spacing_cm/voxel_size_cm)

        # Define position of center of the source voxel (z_s, x_s, y_s), note that it's shifted by 0.5 from the voxel idx to represent the loc of center
        z_s, x_s, y_s = np.indices((int(sample_height_n), int(sample_size_n), int(sample_size_n))) + 0.5
        voxel_pos_ls_flat = np.stack((z_s.flatten(), x_s.flatten(), y_s.flatten()), axis=-1)


        ### Define the location of the detectorlets, the detector is parallel to the yz-plane
        ### The x-posision depends on the distance between the sample and the detecor
        ## x index of the location of the XRF detector
        det_axis_1_idx = sample_size_n + np.ceil(det_from_sample_cm/voxel_size_cm) + 0.5

        ### y, z index of the location of the XRF detector
        ## Define the center of the detector on yz-plane
        det_center_yz = (int(sample_size_n)/2., int(sample_size_n)/2.)

        ## Define the y and z loc(namely the loc along axis 2 and axis 0) of the detectorlets. The y and z loc are confined to be within a circle on the yz plane
        end_det_axis_2_idx_ls = np.array([int((sample_size_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                          int((sample_size_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])
       
        det_axis_2_idx_ls = np.linspace(end_det_axis_2_idx_ls[0], end_det_axis_2_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))

        end_det_axis_0_idx_ls = np.array([int((sample_height_n - det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.),
                                          int((sample_height_n + det_ds_spacing_n * np.floor(det_size_n/det_ds_spacing_n))/2.)])

        det_axis_0_idx_ls = np.linspace(end_det_axis_0_idx_ls[0], end_det_axis_0_idx_ls[1], np.int(det_size_n/det_ds_spacing_n + 1))
        ## Create the meshgrid of y and z coordinates and keep only the coordinates within the detector circle
        y_d, z_d = np.meshgrid(det_axis_2_idx_ls, det_axis_0_idx_ls)

        yz_mask = ((y_d - det_center_yz[0])**2 + (z_d - det_center_yz[1])**2 <= (det_size_n/2)**2).flatten()
        y_d_flat, z_d_flat = y_d.flatten()[yz_mask], z_d.flatten()[yz_mask]


        ## The number of x posision needed to fill into the coodinates depends on the number of the y(or z) coodinates within the circle of detector
        x_d_flat = np.full((y_d_flat.shape), det_axis_1_idx)

        ##
        det_pos_ls_flat = np.stack((z_d_flat, x_d_flat, y_d_flat), axis=-1)
        n_det = len(det_pos_ls_flat)
        print(f"number of detecting points: {n_det}")
        sys.stdout.flush()
        ## define sample edges: 
        ## sample_x_edge is the edge that is closer to the XRF detector
        ## sample_y_edge has two components representing the left and the right edge
        sample_x_edge = np.array([sample_size_n])
        sample_y_edge = np.array([0, sample_size_n]) 
        sample_z_edge = np.array([0, sample_height_n]) 

        dia_len_n = int(1.2*(sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)
        P = tc.zeros(n_det, 3, dia_len_n * sample_height_n * sample_size_n**2)
        longest_int_length = 0
        
        for i,  det_pos in enumerate(det_pos_ls_flat):
            for j, v in enumerate(tqdm(voxel_pos_ls_flat)): 

                # Solving the intersection of the ray with the sample boundary along axis-0
                bdx_int = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_x_edge) # pick the 0th component just because the coordinate is doubly braced

                # Solving the intersection of the ray with the sample boundaries along axis-1 and axis-2, we will get 2 solutions for each axis since there're 2 bdry plane on each axis
                # The desired intersecting point is within the segment(voxel - detectorlet) which is always the one with the larger x coordinate
                bdy_int = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_y_edge)
                if len(bdy_int) != 0:
                    bdy_int = np.array([bdy_int[np.argmax(bdy_int[:,1])]])
                else:
                    pass


                bdz_int = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], sample_z_edge)
                if len(bdz_int) != 0:
                    bdz_int = np.array([bdz_int[np.argmax(bdz_int[:,1])]])
                else:
                    pass

                # Pick the intersecting point that first hit the boundary plan. This point is with the least x value among the 3 intersections.
                bd_int_ls = np.concatenate((bdz_int, bdx_int, bdy_int))
                bd_int = np.clip(np.abs((bd_int_ls[np.argmin(bd_int_ls[:,1])])), 0, sample_size_n)


                # when the beam intersects with a voxel, it either intersects with the x or y or z boundary plane of the voxel
                # find the x,y,z-value of the voxel boundary except the ones on the sample edge

                z_edge_ls = np.where(bd_int[0] > v[0], np.linspace(np.ceil(bd_int[0])-1, np.ceil(v[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))),
                                                       np.linspace(np.ceil(v[0])-1, np.ceil(bd_int[0]), int(np.abs(np.ceil(bd_int[0]) - np.ceil(v[0])))))

                x_edge_ls = np.where(bd_int[1] > v[1], np.linspace(np.ceil(bd_int[1])-1, np.ceil(v[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))),
                                                       np.linspace(np.ceil(v[1])-1, np.ceil(bd_int[1]), int(np.abs(np.ceil(bd_int[1]) - np.ceil(v[1])))))

                y_edge_ls = np.where(bd_int[2] > v[2], np.linspace(np.ceil(bd_int[2])-1, np.ceil(v[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))),
                                                       np.linspace(np.ceil(v[2])-1, np.ceil(bd_int[2]), int(np.abs(np.ceil(bd_int[2]) - np.ceil(v[2])))))


                z_edge_int_ls = trace_beam_z(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], z_edge_ls)
                x_edge_int_ls = trace_beam_x(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], x_edge_ls)
                y_edge_int_ls = trace_beam_y(v[0], v[1], v[2], det_pos[0], det_pos[1], det_pos[2], y_edge_ls)

                # Collect all intersecting points and sort all intersections using the x coordinate
                int_ls = np.concatenate((x_edge_int_ls, y_edge_int_ls, z_edge_int_ls, np.array(bd_int)[np.newaxis,:]))     
                int_ls = int_ls[np.argsort(int_ls[:,1])]

                # calculate the intersecting length in the intersecting voxels
                int_length = np.sqrt(np.diff(int_ls[:,0])**2 + np.diff(int_ls[:,1])**2 + np.diff(int_ls[:,2])**2)
                # just in case that we count some intersections twice, delete the duplicates
                idx_duplicate = np.array(np.where(int_length==0)).flatten()
                int_ls = np.delete(int_ls, idx_duplicate, 0)
                int_length = np.delete(int_length, idx_duplicate) 

                # determine the indices of the intersecting voxels according to the intersecting x,y,z-coordinates
                int_ls_shift = np.zeros((int_ls.shape))
                int_ls_shift[1:] = int_ls[:-1]
                int_idx = np.floor((int_ls + int_ls_shift)/2)[1:]
#                 int_idx = (int_idx[:,0].astype('int'), int_idx[:,1].astype('int'), int_idx[:,2].astype('int'))
                int_idx_flat = int_idx[:,0] * (sample_size_n * sample_size_n) + int_idx[:,1] * sample_size_n + int_idx[:,2]
    
                if len(int_idx_flat) > longest_int_length:
                    longest_int_length = len(int_idx_flat)
                
                P[i, 0, j * dia_len_n: j * dia_len_n + len(int_idx_flat)] = j
                P[i, 1, j * dia_len_n: j * dia_len_n + len(int_idx_flat)] = tc.tensor(int_idx_flat)
                P[i, 2, j * dia_len_n: j * dia_len_n + len(int_idx_flat)] = tc.tensor(int_length * voxel_size_cm)            
                                
                tqdm._instances.clear()
                
        P_short = tc.zeros(n_det, 3, longest_int_length * sample_height_n * sample_size_n**2)
        
        for j, v in enumerate(tqdm(voxel_pos_ls_flat)):
            P_short[:,:,j * longest_int_length: (j+1) * longest_int_length] = P[:,:, j * dia_len_n: j * dia_len_n + longest_int_length]
        
        P = P.numpy()
        P_short = P_short.numpy()
        np.save(P_save_path + '_short.npy', P_short)
        np.save(P_save_path + ".npy", P)
 
    return longest_int_length, n_det, P


def self_absorption_att_ratio_single_theta_3d(src_path, n_det, P, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n, sample_size_cm, sample_height_n, 
                                             this_aN_dic, probe_energy, dev, theta):
    
    fl_all_lines_dic = MakeFLlinesDictionary(this_aN_dic, probe_energy, sample_size_n.cpu().numpy(), sample_size_cm.cpu().numpy(),
                          fl_line_groups = np.array(["K", "L", "M"]), fl_K = fl_K, fl_L = fl_L, fl_M = fl_M, group_lines = True)

    n_voxel = sample_height_n * sample_size_n * sample_size_n
    dia_len_n = int((sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)
    n_lines = tc.as_tensor(fl_all_lines_dic["n_lines"]).to(dev)
    aN_ls = np.array(list(this_aN_dic.values()))   
    grid_concentration = tc.from_numpy(np.load(src_path)).float().to(dev)
    n_element = len(this_aN_dic)
    
    # generate an arrary of total attenuation cross section with the dimension: (n_element, n_elemental_lines)
    # The component in the array represents the total attenuation cross section at some line energy in some element (with unitary concentration)
    FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, fl_all_lines_dic["fl_energy"])).float().to(dev)

    concentration_map_rot = rotate(grid_concentration, theta, dev).float()
    concentration_map_rot_flat = concentration_map_rot.view(n_element, n_voxel).float()


    # lac: linear attenuation coefficient = concentration * attenuation_cross_section, 
    # dimension: n_element, n_lines, n_voxel(FL source), n_voxel)
    lac = concentration_map_rot_flat.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1)
    lac = lac.expand(-1, -1, n_voxel, -1).float()
   
    att_exponent = tc.stack([lac[:,:, P[m][0].to(dtype=tc.long), P[m][1].to(dtype=tc.long)] * P[m][2].view(1, 1, -1).repeat(n_element, n_lines, 1) for m in range(n_det)])
    
    ## summing over the attenation exponent contributed by all intersecting voxels, dim = (n_det, n_element, n_lines, n_voxel (FL source))
    att_exponent_voxel_sum = tc.sum(att_exponent.view(n_det, n_element, n_lines, n_voxel, dia_len_n), axis=-1)
   
    ## calculate the attenuation caused by all elements and get an array of dim = (n_det, n_lines, n_voxel (FL source)), and then take the average over n_det FL ray paths
    ## Final dim = (n_lines, n_voxel (FL source)) representing the attenuation ratio of each fluorescence line emitting from each source voxel.
    SA_att =  tc.mean(tc.exp(-tc.sum(att_exponent_voxel_sum, axis=1)), axis=0)
           
    return SA_att


def create_XRF_data_single_theta_3d(n_det, P, theta_st, theta_end, n_theta, src_path, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n,
                             sample_size_cm, sample_height_n, this_aN_dic, probe_cts, probe_energy, save_path, save_fname, Poisson_noise, dev, this_theta_idx):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    # (n_theta, sample_size_n * sample_size_n)
    theta_ls = - tc.linspace(theta_st, theta_end, n_theta + 1)[:-1]
    theta = theta_ls[this_theta_idx]
    probe_before_attenuation_flat = probe_cts * tc.ones((sample_height_n * sample_size_n * sample_size_n), device=dev)
    att_ratio_map_flat = attenuation_3d(src_path, theta_st, theta_end, n_theta, sample_height_n, sample_size_n, sample_size_cm, this_aN_dic, probe_energy, dev)[0][this_theta_idx]
    SA_att_ratio =  self_absorption_att_ratio_single_theta_3d(src_path, n_det, P, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n, sample_size_cm, sample_height_n, 
                                                             this_aN_dic, probe_energy, dev, theta)
    

    
    # probe_after_attenuation_flat: dimension (sample_height_n * sample_size_n * sample_size_n)
    probe_after_attenuation_flat = probe_before_attenuation_flat * att_ratio_map_flat
    
    #(n_elemental_line, sample_height * sample_size * sample_size)
    fl_ratio_map_tot = generate_fl_signal_from_each_voxel_3d(src_path, theta_st, theta_end, n_theta, sample_size_n, sample_height_n, sample_size_cm, this_aN_dic, probe_energy, dev)[this_theta_idx]

    #calculate fluorescence after self-absorption. dimension: (n_line, n_voxel (FL source))
    fl_signal_SA = tc.unsqueeze(probe_after_attenuation_flat, dim=0) * fl_ratio_map_tot * SA_att_ratio         
    fl_signal_SA = fl_signal_SA.view(-1, sample_height_n * sample_size_n, sample_size_n)
    
    ## summing over the XRF signal collected from strip of voxels along the probe propagation direction
    fl_signal_SA = tc.sum(fl_signal_SA, axis=-1)
    
    ## Calculate the signal collected within the solid angle covered by the detector
    r = (det_from_sample_cm**2 + (det_size_cm/2)**2)**0.5
    h =  r - det_from_sample_cm
    fl_sig_collecting_cap_area = np.pi*((det_size_cm/2)**2 + h**2)
    fl_sig_collecting_ratio = fl_sig_collecting_cap_area / (4*np.pi*r**2)
    fl_signal_SA_theta = fl_signal_SA_theta * fl_sig_collecting_ratio


    if Poisson_noise == True:
        random_noise_generator = default_rng()
        fl_signal_SA = random_noise_generator.poisson(fl_signal_SA)
    np.save(os.path.join(save_path, save_fname +'_{}'.format(this_theta_idx)), fl_signal_SA)
    
    return fl_signal_SA    


def create_XRF_data_3d(n_ranks, rank, P_folder, f_P, theta_st, theta_end, n_theta, src_path, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n,
                             sample_size_cm, sample_height_n, this_aN_dic, probe_cts, probe_energy, save_path, save_fname, Poisson_noise, dev):
    
    P_save_path = os.path.join(P_folder, f_P)
    if not os.path.isfile(P_save_path + ".h5"):  
        stdout_options = {'output_folder': save_path, 'save_stdout': False, 'print_terminal': True}
        print_flush_all(rank, "please create the file that stores the intersecting info. first (P array)")
        
    else:
        if rank == 0:
            P_handle = h5py.File(P_save_path + ".h5", 'r')
            P = tc.from_numpy(P_handle['P_array'][...])
            n_det = P.shape[0] 
            theta_ls = - tc.linspace(theta_st, theta_end, n_theta + 1)[:-1] 
            for this_theta_idx, theta in enumerate(tqdm(theta_ls)):
                create_XRF_data_single_theta_3d(n_det, P, theta_st, theta_end, n_theta, src_path, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, sample_size_n,
                                     sample_size_cm, sample_height_n, this_aN_dic, probe_cts, probe_energy, save_path, save_fname, Poisson_noise, dev, this_theta_idx)
            P_handle.close()
