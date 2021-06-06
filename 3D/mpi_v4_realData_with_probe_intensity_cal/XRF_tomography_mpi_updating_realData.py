#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:58:57 2020

@author: panpanhuang
"""

import os
import sys
import datetime
import numpy as np
import h5py
from mpi4py import MPI
import xraylib as xlib
import xraylib_np as xlib_np

import torch as tc
tc.set_default_tensor_type(tc.FloatTensor)
import torch.nn as nn
from tqdm import tqdm
import time
from data_generation_fns_mpi_updating_realData import rotate, MakeFLlinesDictionary_manual, intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual, find_lines_roi_idx_from_dataset
from standard_calibration import calibrate_incident_probe_intensity
from array_ops_mpi_updating_realData import initialize_guess_3d
from forward_model_mpi_updating_realData import PPM
from misc_mpi_updating_realData import print_flush_root, print_flush_all

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick

import dxchange

import warnings
warnings.filterwarnings("ignore")

fl = {"K": np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE]),
      "L": np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE]),              
      "M": np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])               
     }

    
def reconstruct_jXRFT_tomography(recon_idx, f_recon_parameters, dev, use_std_calibation, probe_intensity, 
                                 std_path, f_std, std_element_lines_roi, density_std_elements, fitting_method,
                                 selfAb, cont_from_check_point, use_saved_initial_guess, ini_kind, init_const, ini_rand_amp,
                                 recon_path, f_initial_guess, f_recon_grid, data_path, f_XRF_data, f_XRT_data,
                                 photon_counts_us_ic_dataset_idx, photon_counts_ds_ic_dataset_idx, XRT_ratio_dataset_idx, theta_ls_dataset_idx,
                                 channel_names, this_aN_dic, element_lines_roi, n_line_group_each_element, solid_angle_adjustment_factor, 
                                 sample_size_n, sample_height_n, sample_size_cm, probe_energy,
                                 n_epoch, save_every_n_epochs, minibatch_size,
                                 b1, b2, lr,
                                 det_size_cm, det_from_sample_cm, manual_det_coord, set_det_coord_cm, det_on_which_side,
                                 manual_det_area, set_det_area_cm2,
                                 det_ds_spacing_cm,
                                 P_folder, f_P, fl_K, fl_L, fl_M):
    
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    
    loss_fn = nn.MSELoss()
    dia_len_n = int(1.2 * (sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5) #dev
    n_voxel_batch = minibatch_size * sample_size_n #dev
    n_voxel = sample_height_n * sample_size_n**2 #dev
    
    #### create the file handle for experimental data; y1: channel data, y2: scalers data ####
    y1_true_handle = h5py.File(os.path.join(data_path, f_XRF_data), 'r')
    y2_true_handle = h5py.File(os.path.join(data_path, f_XRT_data), 'r')  
    ####----------------------------------------------------------------------------------####
    
    #### Calculate the number of elements in the reconstructed object, list the atomic numbers ####
    n_element = len(this_aN_dic)
    aN_ls = np.array(list(this_aN_dic.values()))
    ####--------------------------------------------------------------####
    
    #### Make the lookup table of the fluorescence lines of interests ####
    fl_all_lines_dic = MakeFLlinesDictionary_manual(element_lines_roi,                           
                                                    n_line_group_each_element, probe_energy, 
                                                    sample_size_n, sample_size_cm,
                                                    fl_line_groups = np.array(["K", "L", "M"]), fl_K = fl["K"], fl_L = fl["L"], fl_M = fl["M"]) #cpu
    
    stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
    
    FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, fl_all_lines_dic["fl_energy"])).float().to(dev) #dev
    detected_fl_unit_concentration = tc.as_tensor(fl_all_lines_dic["detected_fl_unit_concentration"]).float().to(dev)
    n_line_group_each_element = tc.IntTensor(fl_all_lines_dic["n_line_group_each_element"]).to(dev)
    n_lines = fl_all_lines_dic["n_lines"] #scalar
    ####--------------------------------------------------------------####
    
    #### Calculate the MAC of probe ####
    probe_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, probe_energy).flatten()).to(dev)
    ####----------------------------####
    
    #### Load all object angles ####
    theta_ls = tc.from_numpy(y1_true_handle['exchange/theta'][...] * np.pi / 180).float()  #unit: rad #cpu
    n_theta = len(theta_ls)  
    ####------------------------####
   
    element_lines_roi_idx = find_lines_roi_idx_from_dataset(data_path, f_XRF_data, element_lines_roi, std_sample = False)
    
    #### pick only the element lines of interests from the channel data. flatten the data to strips
    #### original dim = (n_lines_roi, n_theta, sample_height_n, sample_size_n)
    y1_true = tc.from_numpy(y1_true_handle['exchange/data'][element_lines_roi_idx]).view(len(element_lines_roi_idx), n_theta, sample_height_n * sample_size_n).to(dev)
#     #### pick the probe photon counts after the ion chamber from the scalers data as the transmission data
#     y2_true = tc.from_numpy(y2_true_handle['exchange/data'][photon_counts_ds_ic_dataset_idx]).view(n_theta, sample_height_n * sample_size_n).to(dev)
    
    ## Use this y2_true if using the attenuating expoenent in the XRT loss calculation
    y2_true = tc.from_numpy(y2_true_handle['exchange/data'][XRT_ratio_dataset_idx]).view(n_theta, sample_height_n * sample_size_n).to(dev)
    y2_true = - tc.log(y2_true)
    
    #### pick the probe photon counts calibrated for all optics and detectors
    if use_std_calibation:
        probe_cts = calibrate_incident_probe_intensity(std_path, f_std, fitting_method, std_element_lines_roi, density_std_elements, probe_energy)
    else:
        probe_cts = probe_intensity

    minibatch_ls_0 = tc.arange(n_ranks).to(dev) #dev
    n_batch = (sample_height_n * sample_size_n) // (n_ranks * minibatch_size) #scalar
    
    
    if manual_det_area == True:
#         fl_sig_collecting_ratio = set_det_area_cm2 / (4 * np.pi * det_from_sample_cm**2)
        fl_sig_collecting_ratio = 1.0
    
    else:
#         #### Calculate the detecting solid angle covered by the area of the spherical cap covered by the detector #### 
#         # The distance from the sample to the boundary of the detector
#         r = (det_from_sample_cm**2 + (det_size_cm/2)**2)**0.5   
#         # The height of the cap
#         h =  r - det_from_sample_cm
#         # The area of the cap area
#         fl_sig_collecting_cap_area = np.pi*((det_size_cm/2)**2 + h**2)
#         # The ratio of the detecting solid angle / full soilid angle
#         fl_sig_collecting_ratio = fl_sig_collecting_cap_area / (4*np.pi*r**2)
#         fl_sig_collecting_ratio = ((np.pi * (det_size_cm/2)**2) / det_from_sample_cm**2)/(4*np.pi)
        fl_sig_collecting_ratio = 1.0
    
    if rank == 0: 
        if not os.path.exists(recon_path):
            os.makedirs(recon_path)  
    
    P_save_path = os.path.join(P_folder, f_P)
    
    #Check if the P array exists, if it doesn't exist, call the function to calculate the P array and store it as a .h5 file.
    if not os.path.isfile(P_save_path + ".h5"):   
        intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual(n_ranks, minibatch_size, rank,
                                                                    manual_det_coord, set_det_coord_cm, det_on_which_side,
                                                                    manual_det_area, det_size_cm, det_from_sample_cm, det_ds_spacing_cm,
                                                                    sample_size_n, sample_size_cm,
                                                                    sample_height_n, P_folder, f_P) #cpu
    
    comm.Barrier()
    P_handle = h5py.File(P_save_path + ".h5", 'r')


    if cont_from_check_point == False: 
        
        # load the saved_initial_guess to rank0 cpu
        if use_saved_initial_guess:
            if rank == 0:
                X = np.load(os.path.join(recon_path, f_initial_guess)+'.npy').astype(np.float32)
                X = tc.from_numpy(X).float() #cpu 
            else:
                X = None
                
        # create the initial_uess in rank0 cpu
        else:
            if rank == 0:
                X = initialize_guess_3d("cpu", ini_kind, n_element, sample_size_n, sample_height_n, recon_path, f_recon_grid, f_initial_guess, init_const) #cpu         
                ## Save the initial guess for future reference
                np.save(os.path.join(recon_path, f_initial_guess +'.npy'), X)
                dxchange.write_tiff(X, os.path.join(recon_path, f_initial_guess), dtype='float32', overwrite=True)

                ## Save the initial guess which will be used in reconstruction and will be updated to the current reconstructing result
                np.save(os.path.join(recon_path, f_recon_grid +'.npy'), X)
            else:
                X = None
        
        # bcast X from rank0 cpu to other ranks cpu and then transfer to dev
        comm.Barrier()
        X = comm.bcast(X, root=0).to(dev)
            
        if rank == 0:
            XRF_loss_whole_obj = tc.zeros(n_epoch * n_theta)
            XRT_loss_whole_obj = tc.zeros(n_epoch * n_theta)
            loss_whole_obj = tc.zeros(n_epoch * n_theta)
            with open(os.path.join(recon_path, f_recon_parameters), "w") as recon_params:
                recon_params.write("starting_epoch = 0\n")
                recon_params.write("n_epoch = %d\n" %n_epoch)
                recon_params.write("n_ranks = %d\n" %n_ranks)
                recon_params.write("element_line:\n" + str(element_lines_roi)+"\n") 
                recon_params.write("b1 = %.9f\n" %b1)
                recon_params.write("b2 = %.9f\n" %b2)
                recon_params.write("learning rate = %f\n" %lr)
                recon_params.write("theta_st = %.2f\n" %theta_ls[0])
                recon_params.write("theta_end = %.2f\n" %theta_ls[-1])
                recon_params.write("n_theta = %d\n" %n_theta)
                recon_params.write("sample_size_n = %d\n" %sample_size_n)
                recon_params.write("sample_height_n = %d\n" %sample_height_n)
                recon_params.write("sample_size_cm = %.2f\n" %sample_size_cm)
                recon_params.write("probe_energy_keV = %.2f\n" %probe_energy[0])
                recon_params.write("incident_probe_cts = %.2e\n" %probe_cts)
                
                if not manual_det_area:
                    recon_params.write("det_size_cm = %.2f\n" %det_size_cm)
                
                if not manual_det_coord:
                    recon_params.write("det_from_sample_cm = %.2f\n" %det_from_sample_cm)
                    recon_params.write("det_ds_spacing_cm = %.2f\n" %det_ds_spacing_cm)
        comm.Barrier()          
                    
        for epoch in range(n_epoch):
            t0_epoch = time.perf_counter()
            if rank == 0:
                rand_idx = tc.randperm(n_theta)
                theta_ls_rand = theta_ls[rand_idx]  
            else:
                rand_idx = tc.ones(n_theta)
                theta_ls_rand = tc.ones(n_theta)

            comm.Barrier() 
            rand_idx = comm.bcast(rand_idx, root=0).to(dev) 
            theta_ls_rand = comm.bcast(theta_ls_rand, root=0)   .to(dev)         
            comm.Barrier() 
            
            stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': True}
            timestr = str(datetime.datetime.today())     
            print_flush_root(rank, val=f"epoch: {epoch}, time: {timestr}", output_file='', **stdout_options)
 
            
            for idx, theta in enumerate(theta_ls_rand):
                this_theta_idx = rand_idx[idx] 
                              
                ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
                if selfAb == True:
                    X_ap_rot = rotate(X, theta, dev) #dev
                    lac = X_ap_rot.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1) #dev
                    lac = lac.expand(-1, -1, n_voxel_batch, -1).float() #dev
                
                else:
                    lac = 0.
                
                if rank == 0:
                    XRF_loss_n_batch = tc.zeros(n_batch)
                    XRT_loss_n_batch = tc.zeros(n_batch)
                    total_loss_n_batch = tc.zeros(n_batch)
                    
                for m in range(n_batch):                    
                    minibatch_ls = n_ranks * m + minibatch_ls_0  #dev, e.g. [5,6,7,8]
                    p = minibatch_ls[rank]
                    
                    if selfAb == True:
                        P_minibatch = tc.from_numpy(P_handle['P_array'][:,:, p * dia_len_n * minibatch_size * sample_size_n: (p+1) * dia_len_n * minibatch_size * sample_size_n]).to(dev)
                        n_det = P_minibatch.shape[0] 
                    
                    else:
                        P_minibatch = 0
                        n_det = 0
                    
#                     stdout_options = {'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
#                     print_flush_all(rank, val=p * dia_len_n * minibatch_size * sample_size_n, output_file=f'P_start_idx_{rank}.csv', **stdout_options)
#                     print_flush_all(rank, val=(p+1) * dia_len_n * minibatch_size * sample_size_n, output_file=f'P_end_idx_{rank}.csv', **stdout_options)
                  
#                     stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
#                     print_flush_root(rank, val=minibatch_ls, output_file='minibatch_ls.csv', **stdout_options)
                    
                    ## Load us_ic as the incoming probe count in this minibatch       
            
                    model = PPM(dev, selfAb, lac, X, p, n_element, n_lines, FL_line_attCS_ls,
                                 detected_fl_unit_concentration, n_line_group_each_element,
                                 sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                                 probe_energy, probe_cts, probe_attCS_ls,
                                 theta, solid_angle_adjustment_factor,
                                 n_det, P_minibatch, det_size_cm, det_from_sample_cm, fl_sig_collecting_ratio)
                    
                    optimizer = tc.optim.Adam(model.parameters(), lr=lr)              
                                    
                    ## load true data, y1: XRF_data, y2: XRT data
                    #dev #Take all lines_roi, this_theta_idx, and strips in this minibatc                    
                    y1_hat, y2_hat = model()
                    XRF_loss = loss_fn(y1_hat, y1_true[:, this_theta_idx, minibatch_size * p : minibatch_size * (p+1)])
                    XRT_loss = loss_fn(y2_hat, b2 * y2_true[this_theta_idx, minibatch_size * p : minibatch_size * (p+1)])
                    loss = XRF_loss + b1 * XRT_loss

                    optimizer.zero_grad()
                    loss.backward()                    
                    optimizer.step()
                         
                    updated_minibatch = model.xp.detach().cpu()
                    comm.Barrier()
                    updated_batch = comm.gather(updated_minibatch,root=0) #dev
    
                    ## Note that the gathered updated_batch is a list of tensor, namely a list with n_ranks tensors. 
                    ## E.g., if n_ranks=2, updated_batch = [tensor1, tensor2]
                    ## tensor1 and tensor2 both have the dim = (n_element, sample_height_n (per minibatch), sample_size_n, sample_size_n)
                    
                    XRF_loss = XRF_loss.detach().item() 
                    XRF_loss_sum = comm.reduce(XRF_loss, op=MPI.SUM, root=0)
                    
                    XRT_loss = XRT_loss.detach().item() 
                    XRT_loss_sum = comm.reduce(XRT_loss, op=MPI.SUM, root=0)                    
                                       
                    loss = loss.detach().item()           
                    loss_sum = comm.reduce(loss, op=MPI.SUM, root=0)
                    comm.Barrier()                    
                    
                    if rank == 0: 
                        XRF_loss_n_batch[m] = XRF_loss_sum/n_ranks
                        XRT_loss_n_batch[m] = XRT_loss_sum/n_ranks
                        total_loss_n_batch[m] = loss_sum/n_ranks
                        updated_batch = tc.cat(updated_batch, dim=1)
                        X[:, minibatch_size * p // sample_size_n : minibatch_size * (p + n_ranks) // sample_size_n, :, :] = updated_batch.detach()
                        X = tc.clamp(X, 0, float('inf'))
                        # Note that we need to detach the voxels in the updated_batch of the current iteration.
                        # Otherwise Pytorch will keep calculating the gradient of the updated_batch of the current iteration in the NEXT iteration
                        # The updated X needs to be broadcasted back to all ranks only at each new obj. angle
                        # because updating the remaining layers in the current obj. angle doesn't require the info of the updated layers.   
               
                    del model  
                
                ## Prepare to bcast the updated X from rank0 to all ranks in order to calculate the LAC at the next obj. angle  
                if rank == 0:
                    loss_whole_obj[n_theta * epoch + idx] = tc.mean(total_loss_n_batch)
                    XRF_loss_whole_obj[n_theta * epoch + idx] = tc.mean(XRF_loss_n_batch)
                    XRT_loss_whole_obj[n_theta * epoch + idx] = tc.mean(XRT_loss_n_batch)
                    X_cpu = X.cpu()
                else:
                    X_cpu = None
                    
                comm.Barrier()
                X = comm.bcast(X_cpu, root=0).to(dev)
                
                # Save the current object to file                  
                if rank == 0:                  
                    np.save(os.path.join(recon_path, f_recon_grid)+'.npy', X_cpu.numpy())
                comm.Barrier() 
                    
                del lac
#                 tc.cuda.empty_cache()

            stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
            per_epoch_time = time.perf_counter() - t0_epoch
            print_flush_root(rank, val=per_epoch_time, output_file=f'per_epoch_time_mb_size_{minibatch_size}.csv', **stdout_options)
            comm.Barrier()
                
            if rank ==0 and epoch != 0:
                epsilon = tc.mean((X_cpu - X_previous)**2)
                print_flush_root(rank, val=epsilon, output_file=f'model_mse_epoch.csv', **stdout_options)
                
                if epsilon < 10**(-12):             
                    if rank == 0:
                        recon_idx += 1
                        np.save(os.path.join(recon_path, f_recon_grid)+"_"+str(epoch)+"_ending_condition"+".npy", X_cpu.numpy())
                        dxchange.write_tiff(X_cpu.numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(epoch)+"_ending_condition", dtype='float32', overwrite=True)                     
                    break
                else:
                    pass
            
            else:
                pass
            
            comm.Barrier()
            if rank == 0:
                X_previous = X_cpu
            
            comm.Barrier()
#             print(rank == 0)
#             print((epoch+1)%1 == 0)
#             print((epoch+1)//40 !=0)
#             print(epoch+1 == n_epoch)
            
            if  rank == 0 and ((epoch+1)%save_every_n_epochs == 0 and (epoch+1)//save_every_n_epochs !=0 or epoch+1 == n_epoch):
                recon_idx += 1
                np.save(os.path.join(recon_path, f_recon_grid)+"_"+str(epoch)+".npy", X_cpu.numpy())
                dxchange.write_tiff(X_cpu.numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(epoch), dtype='float32', overwrite=True)            

        ## It's important to close the hdf5 file hadle in the end of the reconstruction.
        P_handle.close()         
        comm.Barrier()
        
        if rank == 0:            
            fig6 = plt.figure(figsize=(10,15))
            gs6 = gridspec.GridSpec(nrows=3, ncols=1, width_ratios=[1])

            fig6_ax1 = fig6.add_subplot(gs6[0,0])
            fig6_ax1.plot(loss_whole_obj.numpy())
            fig6_ax1.set_xlabel('epoch')
            fig6_ax1.set_ylabel('loss')
            fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

            fig6_ax2 = fig6.add_subplot(gs6[1,0])
            fig6_ax2.plot(XRF_loss_whole_obj.numpy())
            fig6_ax2.set_xlabel('epoch')
            fig6_ax2.set_ylabel('XRF loss')
            fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

            fig6_ax3 = fig6.add_subplot(gs6[2,0])
            fig6_ax3.plot(XRT_loss_whole_obj.numpy())
            fig6_ax3.set_xlabel('epoch')
            fig6_ax3.set_ylabel('XRT loss')
            fig6_ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            
            plt.savefig(os.path.join(recon_path, 'loss_signal.pdf'))
            
            np.save(os.path.join(recon_path, 'XRF_loss_signal.npy'), XRF_loss_whole_obj.numpy())
            np.save(os.path.join(recon_path, 'XRT_loss_signal.npy'), XRT_loss_whole_obj.numpy())
            np.save(os.path.join(recon_path, 'loss_signal.npy'), loss_whole_obj.numpy())
#             np.save(os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx)+".npy", X_cpu.numpy())
#             dxchange.write_tiff(X_cpu.numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx), dtype='float32', overwrite=True)
            
        comm.Barrier()
        
    if cont_from_check_point == True:
        if rank == 0:
            X = np.load(os.path.join(recon_path, f_recon_grid) + '.npy').astype(np.float32)
            X = tc.from_numpy(X).float()
            
        else:
            X = None
        
        # bcast X from rank0 cpu to other ranks cpu and then transfer to dev
        comm.Barrier()
        X = comm.bcast(X, root=0).to(dev)
        
            
        if rank == 0:
            XRF_loss_whole_obj =  tc.from_numpy(np.load(os.path.join(recon_path, 'XRF_loss_signal.npy')).astype(np.float32))
            XRT_loss_whole_obj = tc.from_numpy(np.load(os.path.join(recon_path, 'XRT_loss_signal.npy')).astype(np.float32))
            loss_whole_obj = tc.from_numpy(np.load(os.path.join(recon_path, 'loss_signal.npy')).astype(np.float32))
            
            XRF_loss_whole_obj_cont = tc.zeros(n_epoch * n_theta)
            XRT_loss_whole_obj_cont = tc.zeros(n_epoch * n_theta)
            loss_whole_obj_cont = tc.zeros(n_epoch * n_theta)
            
            with open(os.path.join(recon_path, f_recon_parameters), "r") as recon_params:
                params_list = []
                for line in recon_params.readlines():
                    params_list.append(line.rstrip("\n"))
                n_ending = len(params_list)

            with open(os.path.join(recon_path, f_recon_parameters), "a") as recon_params:
                n_start_last = n_ending - 22 + (4 - len(element_lines_roi))

                previsous_starting_epoch = int(params_list[n_start_last][params_list[n_start_last].find("=")+1:])
                previous_n_epoch = int(params_list[n_start_last+1][params_list[n_start_last+1].find("=")+1:])
                starting_epoch = previsous_starting_epoch + previous_n_epoch
                recon_params.write("\n")
                recon_params.write("###########################################\n")
                recon_params.write("starting_epoch = %d\n" %starting_epoch)
                recon_params.write("n_epoch = %d\n" %n_epoch)
                recon_params.write("n_ranks = %d\n" %n_ranks)
                recon_params.write("element_line:\n" + str(element_lines_roi)+"\n") 
                recon_params.write("b1 = %.9f\n" %b1)
                recon_params.write("b2 = %.9f\n" %b2)
                recon_params.write("learning rate = %f\n" %lr)
                recon_params.write("theta_st = %.2f\n" %theta_ls[0])
                recon_params.write("theta_end = %.2f\n" %theta_ls[-1])
                recon_params.write("n_theta = %d\n" %n_theta)
                recon_params.write("sample_size_n = %d\n" %sample_size_n)
                recon_params.write("sample_height_n = %d\n" %sample_height_n)
                recon_params.write("sample_size_cm = %.2f\n" %sample_size_cm)
                recon_params.write("probe_energy_keV = %.2f\n" %probe_energy[0])
                recon_params.write("incident_probe_cts = %.2e\n" %probe_cts)             
                if not manual_det_area:
                    recon_params.write("det_size_cm = %.2f\n" %det_size_cm)
                
                if not manual_det_coord:
                    recon_params.write("det_from_sample_cm = %.2f\n" %det_from_sample_cm)
                    recon_params.write("det_ds_spacing_cm = %.2f\n" %det_ds_spacing_cm)
        comm.Barrier()  
       
        for epoch in range(n_epoch):
            t0_epoch = time.perf_counter()
            if rank == 0:
                rand_idx = tc.randperm(n_theta)
                theta_ls_rand = theta_ls[rand_idx]  
            else:
                rand_idx = tc.ones(n_theta)
                theta_ls_rand = tc.ones(n_theta)

            comm.Barrier() 
            rand_idx = comm.bcast(rand_idx, root=0).to(dev) 
            theta_ls_rand = comm.bcast(theta_ls_rand, root=0)   .to(dev)         
            comm.Barrier()     
             
            
            stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': True}
            timestr = str(datetime.datetime.today())
            print_flush_root(rank, f"epoch: {epoch}, time: {timestr}", output_file='', **stdout_options)
            
            for idx, theta in enumerate(theta_ls_rand):
                this_theta_idx = rand_idx[idx]
                
                ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
                if selfAb == True:
                    X_ap_rot = rotate(X, theta, dev) #dev
                    lac = X_ap_rot.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1) #dev
                    lac = lac.expand(-1, -1, n_voxel_batch, -1).float() #dev
                
                else:
                    lac = 0.
                
                if rank == 0:
                    XRF_loss_n_batch = tc.zeros(n_batch)
                    XRT_loss_n_batch = tc.zeros(n_batch)
                    total_loss_n_batch = tc.zeros(n_batch)
                
                for m in range(n_batch):
                    minibatch_ls = n_ranks * m + minibatch_ls_0  #dev
                    p = minibatch_ls[rank]
                    P_minibatch = tc.from_numpy(P_handle['P_array'][:,:, p * dia_len_n * minibatch_size * sample_size_n: (p+1) * dia_len_n * minibatch_size * sample_size_n]).to(dev)                                      
                    n_det = P_minibatch.shape[0]  
                                       
                    model = PPM(dev, selfAb, lac, X, p, n_element, n_lines, FL_line_attCS_ls,
                                 detected_fl_unit_concentration, n_line_group_each_element,
                                 sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                                 probe_energy, probe_cts, probe_attCS_ls,
                                 theta, solid_angle_adjustment_factor,
                                 n_det, P_minibatch, det_size_cm, det_from_sample_cm, fl_sig_collecting_ratio)

                    optimizer = tc.optim.Adam(model.parameters(), lr=lr)   
            
                    ## load true data, y1: XRF_data, y2: XRT data
                    #dev #Take all lines_roi, this_theta_idx, and strips in this minibatc                    
                    y1_hat, y2_hat = model()
                    XRF_loss = loss_fn(y1_hat, y1_true[:, this_theta_idx, minibatch_size * p : minibatch_size * (p+1)])
                    XRT_loss = loss_fn(y2_hat, b2 * y2_true[this_theta_idx, minibatch_size * p : minibatch_size * (p+1)])
                    loss = XRF_loss + b1 * XRT_loss
            
                    optimizer.zero_grad()
                    loss.backward()             
                    optimizer.step()
                
                    updated_minibatch = model.xp.detach().cpu()
                    comm.Barrier()
                    updated_batch = comm.gather(updated_minibatch,root=0)
                    
                    
                    ## Note that the gathered updated_batch is a list of tensor, namely a list with n_ranks tensors. 
                    ## E.g., if n_ranks=2, updated_batch = [tensor1, tensor2]
                    ## tensor1 and tensor2 both have the dim = (n_element, sample_height_n (per minibatch), sample_size_n, sample_size_n)  
                
                    XRF_loss = XRF_loss.detach().item() 
                    XRF_loss_sum = comm.reduce(XRF_loss, op=MPI.SUM, root=0)
                    
                    XRT_loss = XRT_loss.detach().item() 
                    XRT_loss_sum = comm.reduce(XRT_loss, op=MPI.SUM, root=0)                    
                                       
                    loss = loss.detach().item()           
                    loss_sum = comm.reduce(loss, op=MPI.SUM, root=0)
                    comm.Barrier()                  
                
                    if rank == 0:
                        XRF_loss_n_batch[m] = XRF_loss_sum/n_ranks
                        XRT_loss_n_batch[m] = XRT_loss_sum/n_ranks
                        total_loss_n_batch[m] = loss_sum/n_ranks                        
                        updated_batch = tc.cat(updated_batch, dim=1)
                        X[:, minibatch_size * p // sample_size_n : minibatch_size * (p + n_ranks) // sample_size_n, :, :] = updated_batch.detach()
                        X = tc.clamp(X, 0, float('inf'))
                        # Note that we need to detach the voxels in the updated_batch of the current iteration.
                        # Otherwise Pytorch will keep calculating the gradient of the updated_batch of the current iteration in the NEXT iteration
                        # The updated X needs to be broadcasted back to all ranks only at each new obj. angle
                        # because updating the remaining layers in the current obj. angle doesn't require the info of the updated layers.                                                             
                    del model 
                ## Prepare to bcast the updated X from rank0 to all ranks in order to calculate the LAC at the next obj. angle                   
                if rank == 0:  
                    loss_whole_obj_cont[n_theta * epoch + idx] = tc.mean(total_loss_n_batch)
                    XRF_loss_whole_obj_cont[n_theta * epoch + idx] = tc.mean(XRF_loss_n_batch)
                    XRT_loss_whole_obj_cont[n_theta * epoch + idx] = tc.mean(XRT_loss_n_batch)
                    X_cpu = X.cpu()
                else:
                    X_cpu = None
                    
                comm.Barrier()
                X = comm.bcast(X_cpu, root=0).to(dev)
                
                # Save the current object to file                 
                if rank == 0:                  
                    np.save(os.path.join(recon_path, f_recon_grid)+'.npy', X_cpu.numpy())
                comm.Barrier() 
                    
                del lac
#                 tc.cuda.empty_cache()
                
                
            stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
            per_epoch_time = time.perf_counter() - t0_epoch
            print_flush_root(rank, val=per_epoch_time, output_file=f'per_epoch_time_mb_size_{minibatch_size}.csv', **stdout_options)
            comm.Barrier()

            if rank ==0 and epoch != 0:
                epsilon = tc.mean((X_cpu - X_previous)**2)
                print_flush_root(rank, val=epsilon, output_file=f'model_mse_epoch.csv', **stdout_options)
                
                if epsilon < 10**(-12):             
                    if rank == 0:
                        recon_idx += 1
                        np.save(os.path.join(recon_path, f_recon_grid)+"_"+str(epoch)+"_ending_condition"+".npy", X_cpu.numpy())
                        dxchange.write_tiff(X_cpu.numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(epoch)+"_ending_condition", dtype='float32', overwrite=True)                     
                    break
                else:
                    pass
            
            else:
                pass
            
            comm.Barrier()
            if rank == 0:
                X_previous = X_cpu            
            comm.Barrier()            
            
            if  rank == 0 and ((epoch+1)%save_every_n_epochs == 0 and (epoch+1)//save_every_n_epochs !=0 or epoch+1 == n_epoch):
                recon_idx += 1
                np.save(os.path.join(recon_path, f_recon_grid)+"_"+str(starting_epoch+epoch)+".npy", X_cpu.numpy())
                dxchange.write_tiff(X_cpu.numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(starting_epoch+epoch), dtype='float32', overwrite=True)    
            
        ## It's important to close the hdf5 file hadle in the end of the reconstruction.
        P_handle.close()  
        comm.Barrier()
        
        if rank == 0:                           
            loss_whole_obj = tc.cat((loss_whole_obj, loss_whole_obj_cont))
            XRF_loss_whole_obj = tc.cat((XRF_loss_whole_obj, XRF_loss_whole_obj_cont))
            XRT_loss_whole_obj = tc.cat((XRT_loss_whole_obj, XRT_loss_whole_obj_cont))
 
            fig6 = plt.figure(figsize=(10,15))
            gs6 = gridspec.GridSpec(nrows=3, ncols=1, width_ratios=[1])

            fig6_ax1 = fig6.add_subplot(gs6[0,0])
            fig6_ax1.plot(loss_whole_obj.numpy())
            fig6_ax1.set_xlabel('epoch')
            fig6_ax1.set_ylabel('loss')
            fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

            fig6_ax2 = fig6.add_subplot(gs6[1,0])
            fig6_ax2.plot(XRF_loss_whole_obj.numpy())
            fig6_ax2.set_xlabel('epoch')
            fig6_ax2.set_ylabel('XRF loss')
            fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

            fig6_ax3 = fig6.add_subplot(gs6[2,0])
            fig6_ax3.plot(XRT_loss_whole_obj.numpy())
            fig6_ax3.set_xlabel('epoch')
            fig6_ax3.set_ylabel('XRT loss')
            fig6_ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            
            plt.savefig(os.path.join(recon_path, 'loss_signal.pdf'))
            
            np.save(os.path.join(recon_path, 'XRF_loss_signal.npy'), XRF_loss_whole_obj.numpy())
            np.save(os.path.join(recon_path, 'XRT_loss_signal.npy'), XRT_loss_whole_obj.numpy())
            np.save(os.path.join(recon_path, 'loss_signal.npy'), loss_whole_obj.numpy())
#             np.save(os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx)+".npy", X_cpu.numpy())
#             dxchange.write_tiff(X_cpu.numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx), dtype='float32', overwrite=True)
             