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
from mpi4py import MPI
import xraylib_np as xlib_np

import torch as tc
tc.set_default_tensor_type(tc.FloatTensor)
import torch.nn as nn
from tqdm import tqdm
import time
from data_generation_fns_mpi_updating import rotate, MakeFLlinesDictionary, intersecting_length_fl_detectorlet_3d
from array_ops_mpi_updating import initialize_guess_3d
from forward_model_mpi_updating import PPM
from misc_mpi_updating import print_flush_root

import matplotlib.pyplot as plt
import matplotlib 
matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Helvetica'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick

import dxchange

import warnings
warnings.filterwarnings("ignore")


                                                     
def reconstruct_jXRFT_tomography(dev, selfAb, recon_idx, cont_from_check_point, use_saved_initial_guess, recon_path, f_initial_guess, f_recon_grid,
                                 grid_path, f_grid, data_path, f_XRF_data, f_XRT_data, this_aN_dic,
                                 ini_kind, f_recon_parameters, n_epoch, minibatch_size, b, lr, init_const, 
                                 fl_line_groups, fl_K, fl_L, fl_M, group_lines, theta_st, theta_end, n_theta,
                                 sample_size_n, sample_height_n, sample_size_cm,
                                 probe_energy, probe_cts, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, P_folder, f_P,
                                 ):

    
    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    
    
    loss_fn = nn.MSELoss()
    X_true = tc.from_numpy(np.load(os.path.join(grid_path, f_grid)).astype(np.float32)) #cpu
    dia_len_n = int((sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5) #dev
    n_voxel_batch = minibatch_size * sample_size_n #dev
    n_voxel = sample_height_n * sample_size_n**2 #dev
    aN_ls = np.array(list(this_aN_dic.values())) #cpu
    
    fl_all_lines_dic = MakeFLlinesDictionary(this_aN_dic, probe_energy,
              sample_size_n.cpu().numpy(), sample_size_cm.cpu().numpy(),
              fl_line_groups, fl_K, fl_L, fl_M,
              group_lines) #cpu 
    FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, fl_all_lines_dic["fl_energy"])).float().to(dev) #dev
    n_lines = fl_all_lines_dic["n_lines"] #scalar
    
    minibatch_ls_0 = tc.arange(n_ranks).to(dev) #dev
    n_batch = (sample_height_n * sample_size_n) // (n_ranks * minibatch_size) #scalar
    theta_ls = - tc.linspace(theta_st, theta_end, n_theta+1)[:-1].to(dev) #dev
    n_element = len(this_aN_dic) #scalar
    
    
    if rank == 0: 
        if not os.path.exists(recon_path):
            os.mkdir(recon_path)     
        
        longest_int_length, n_det, P = intersecting_length_fl_detectorlet_3d(det_size_cm, det_from_sample_cm, det_ds_spacing_cm,
                                                  sample_size_n.cpu().numpy(), sample_size_cm.cpu().numpy(),
                                                  sample_height_n.cpu().numpy(), P_folder, f_P) #cpu
        
        P = P.astype(np.float32)
        
    else:
        n_det = None
            
    n_det = comm.bcast(n_det, root=0)
    comm.Barrier()
    
    if cont_from_check_point == False: 
        
        if use_saved_initial_guess:
            X = np.load(os.path.join(recon_path, f_initial_guess)+'.npy').astype(np.float32)
            X = tc.from_numpy(X).float() #cpu 
        else:
            X = initialize_guess_3d(dev, ini_kind, grid_path, f_grid, recon_path, f_recon_grid, f_initial_guess, init_const).cpu() #cpu
            
            if rank == 0:
                ## Save the initial guess for future reference
                np.save(os.path.join(recon_path, f_initial_guess +'.npy'), X)
                dxchange.write_tiff(X, os.path.join(recon_path, f_initial_guess), dtype='float32', overwrite=True)

                ## Save the initial guess which will be used in reconstruction and will be updated to the current reconstructing result
                np.save(os.path.join(recon_path, f_recon_grid +'.npy'), X)
                dxchange.write_tiff(X, os.path.join(recon_path, f_recon_grid), dtype='float32', overwrite=True)
                
        X = X.to(dev) #dev  # need fo be modified, so that the initial X is generated at CPU, then sent to GPU division of each rank
            
        if rank == 0:
            with open(os.path.join(recon_path, f_recon_parameters), "w") as recon_params:
                recon_params.write("starting_epoch = 0\n")
                recon_params.write("n_epoch = %d\n" %n_epoch)
                recon_params.write(str(this_aN_dic)+"\n") 
                recon_params.write("n_ranks = %d\n" %n_ranks)
                recon_params.write("b = %.9f\n" %b)
                recon_params.write("learning rate = %f\n" %lr)
                recon_params.write("theta_st = %.2f\n" %theta_st)
                recon_params.write("theta_end = %.2f\n" %theta_end)
                recon_params.write("n_theta = %d\n" %n_theta)
                recon_params.write("sample_size_n = %d\n" %sample_size_n)
                recon_params.write("sample_height_n = %d\n" %sample_height_n)
                recon_params.write("sample_size_cm = %.2f\n" %sample_size_cm)
                recon_params.write("probe_energy = %.2f\n" %probe_energy[0])
                recon_params.write("probe_cts = %.2e\n" %probe_cts)
                recon_params.write("det_size_cm = %.2f\n" %det_size_cm)
                recon_params.write("det_from_sample_cm = %.2f\n" %det_from_sample_cm)
                recon_params.write("det_ds_spacing_cm = %.2f\n" %det_ds_spacing_cm)

                loss_epoch = tc.zeros(n_epoch)
                mse_epoch = tc.zeros(n_epoch, len(this_aN_dic))
                  
               
        
        
        for epoch in range(n_epoch):
            t0_epoch = time.perf_counter()
            if rank == 0:
                rand_idx = tc.randperm(n_theta)
                theta_ls = theta_ls[rand_idx]  
            else:
                rand_idx = tc.ones(n_theta)

            comm.Barrier() 
            rand_idx = comm.bcast(rand_idx, root=0) 
            theta_ls = comm.bcast(theta_ls, root=0)    
            
            stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': True}
            timestr = str(datetime.datetime.today())     
            print_flush_root(rank, val=f"epoch: {epoch}, time: {timestr}", output_file='', **stdout_options)
            
            for idx, theta in enumerate(theta_ls):
                this_theta_idx = rand_idx[idx]
                
                ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
                if selfAb == True:
                    X_ap_rot = rotate(X, theta, dev) #dev
                    lac = X_ap_rot.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1) #dev
                    lac = lac.expand(-1, -1, n_voxel_batch, -1).float() #dev
                
                else:
                    lac = 0.
                
                ## load true data, y1: XRF_data, y2: XRT data
                y1_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRF_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32)).to(dev) #dev
                y2_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRT_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32)).to(dev) #dev
             
                for m in range(n_batch):                    
                    minibatch_ls = n_ranks * m + minibatch_ls_0  #dev
                    p = minibatch_ls[rank]
                    P_batch = None
                    if rank == 0:
                        ## Prepare a P_batch array contains P_minibath required by other rank(s)                            
                        P_batch = np.copy(P[:,:, p * dia_len_n * minibatch_size * sample_size_n :  (p + n_ranks) * dia_len_n * minibatch_size * sample_size_n]) #cpu
                        P_batch = P_batch.reshape((n_det, 3, n_ranks, dia_len_n * minibatch_size * sample_size_n)) #cpu
                        P_batch = np.transpose(P_batch, (2,0,1,3)) #cpu
                        P_batch = np.ascontiguousarray(P_batch, dtype=np.float32)
                    
                    P_minibatch = np.zeros((n_det, 3, dia_len_n * minibatch_size * sample_size_n)).astype(np.float32)
           
                    comm.Scatter(P_batch, P_minibatch, root=0)  #cpu
                    comm.Barrier()
                    P_minibatch = tc.from_numpy(P_minibatch).to(dev) #dev                                               
                    
                    model = PPM(dev, selfAb, lac, X, p, n_element, sample_height_n, minibatch_size,
                                 sample_size_n, sample_size_cm,
                                 this_aN_dic, probe_energy, probe_cts,
                                 theta_st, theta_end, n_theta, this_theta_idx,
                                 n_det, P_minibatch, det_size_cm, det_from_sample_cm)
                    
                    optimizer = tc.optim.Adam(model.parameters(), lr=lr)              
                    
                    stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
                    t0_fModel = time.perf_counter()
                    y1_hat, y2_hat = model()
                    fModel_time = time.perf_counter() - t0_fModel
                    print_flush_root(rank, val=fModel_time, output_file=f'forward_model_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)

                    XRF_loss = loss_fn(y1_hat, y1_true[:, minibatch_size * p : minibatch_size * (p+1)])
                    XRT_loss = loss_fn(y2_hat, y2_true[minibatch_size * p : minibatch_size * (p+1)])
                    loss = XRF_loss + b * XRT_loss

                    optimizer.zero_grad()
                    
                    t0_calGradient = time.perf_counter()
                    loss.backward()
                    calGradient_time = time.perf_counter() - t0_calGradient 
                    print_flush_root(rank, val=calGradient_time, output_file=f'gradient_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                    
                    t0_bPropagation = time.perf_counter()
                    optimizer.step()
                    bPropagation_time = time.perf_counter() - t0_bPropagation
                    print_flush_root(rank, val=bPropagation_time, output_file=f'backward_propagation_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)

                    updated_minibatch = model.xp.detach().cpu()
                    updated_batch = comm.gather(updated_minibatch,root=0)
                    comm.Barrier()
                    ## Note that the gathered updated_batch is a list of tensor, namely a list with n_ranks tensors. 
                    ## E.g., if n_ranks=2, updated_batch = [tensor1, tensor2]
                    ## tensor1 and tensor2 both have the dim = (n_element, sample_height_n (per minibatch), sample_size_n, sample_size_n)
               
                    if rank == 0: 
                        updated_batch = tc.cat(updated_batch, dim=1)
                        X[:, minibatch_size * p // sample_size_n : minibatch_size * (p + n_ranks) // sample_size_n, :, :] = updated_batch.detach()
                        X = tc.clamp(X, 0, float('inf'))
             
                    comm.Barrier()
                    X = comm.bcast(X, root=0).to(dev)
                    comm.Barrier()
                   
                    del model               
            
                del lac
                tc.cuda.empty_cache()
                
            loss = loss.detach().item()           
            loss_sum = comm.reduce(loss, op=MPI.SUM, root=0) 
            if rank == 0:
                X_cpu = X.cpu()
                loss = loss_sum / n_ranks
                loss_epoch[epoch] = loss
                mse_epoch[epoch] = tc.mean(tc.square(X_cpu - X_true).view(X_cpu.shape[0], X_cpu.shape[1]*X_cpu.shape[2]*X_cpu.shape[3]), dim=1)                
                np.save(os.path.join(recon_path, f_recon_grid)+'.npy', X_cpu.numpy())
                     
            stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
            per_epoch_time = time.perf_counter() - t0_epoch
            print_flush_root(rank, val=per_epoch_time, output_file=f'per_epoch_time_mb_size_{minibatch_size}.csv', **stdout_options)
                 
            
        if rank == 0:    
            mse_epoch_tot = tc.mean(mse_epoch, dim=1)
        
            fig6 = plt.figure(figsize=(15,5))
            gs6 = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1,1])

            fig6_ax1 = fig6.add_subplot(gs6[0,0])
            fig6_ax1.plot(loss_epoch.numpy())
            fig6_ax1.set_xlabel('epoch')
            fig6_ax1.set_ylabel('loss')
            fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))

            fig6_ax2 = fig6.add_subplot(gs6[0,1])
            fig6_ax2.plot(mse_epoch_tot.numpy())
            fig6_ax2.set_xlabel('epoch')
            fig6_ax2.set_ylabel('mse of model')
            fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%g'))
            plt.savefig(os.path.join(recon_path, 'loss_and_tot_mse.pdf'))


            fig7 = plt.figure(figsize=(X_cpu.shape[0]*6, 4))
            gs7 = gridspec.GridSpec(nrows=1, ncols=X_cpu.shape[0], width_ratios=[1]*X_cpu.shape[0])
            for i in range(X_cpu.shape[0]):
                fig7_ax1 = fig7.add_subplot(gs7[0,i])
                fig7_ax1.plot(mse_epoch[:,i].numpy())
                fig7_ax1.set_xlabel('epoch')
                fig7_ax1.set_ylabel('mse of model (each element)')
                fig7_ax1.set_title(str(list(this_aN_dic.keys())[i]))
                fig7_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))         
            plt.savefig(os.path.join(recon_path, 'mse_model.pdf'))
        
            np.save(os.path.join(recon_path, 'loss_epoch.npy'), loss_epoch.numpy())
            np.save(os.path.join(recon_path, 'mse_model_elements.npy'), mse_epoch.numpy())
            np.save(os.path.join(recon_path, 'mse_model.npy'), mse_epoch_tot.numpy()) 
            dxchange.write_tiff(X_cpu.numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx), dtype='float32', overwrite=True)
        
    if cont_from_check_point == True:
        X = np.load(os.path.join(recon_path, f_recon_grid) + '.npy').astype(np.float32)
        X = tc.from_numpy(X).to(dev)
        recon_idx += 1
        
        if rank == 0:
            loss_epoch =  tc.from_numpy(np.load(os.path.join(recon_path, 'loss_epoch.npy')).astype(np.float32))
            mse_epoch = tc.from_numpy(np.load(os.path.join(recon_path, 'mse_model_elements.npy')).astype(np.float32))
            mse_epoch_tot = tc.from_numpy(np.load(os.path.join(recon_path, 'mse_model.npy')).astype(np.float32))
            
            with open(os.path.join(recon_path, f_recon_parameters), "r") as recon_params:
                params_list = []
                for line in recon_params.readlines():
                    params_list.append(line.rstrip("\n"))
                n_ending = len(params_list)

            with open(os.path.join(recon_path, f_recon_parameters), "a") as recon_params:
                n_start_last = n_ending - 17

                previous_epoch = int(params_list[n_start_last][params_list[n_start_last].find("=")+1:])   
                recon_params.write("\n")
                recon_params.write("###########################################\n")
                recon_params.write("starting_epoch = %d\n" %(previous_epoch + n_epoch))
                recon_params.write("n_epoch = %d\n" %n_epoch)
                recon_params.write(str(this_aN_dic)+"\n") 
                recon_params.write("n_ranks = %d\n" %n_ranks)
                recon_params.write("b = %f\n" %b)
                recon_params.write("learning rate = %f\n" %lr)
                recon_params.write("theta_st = %.2f\n" %theta_st)
                recon_params.write("theta_end = %.2f\n" %theta_end)
                recon_params.write("n_theta = %d\n" %n_theta)
                recon_params.write("sample_size_n = %d\n" %sample_size_n)
                recon_params.write("sample_height_n = %d\n" %sample_height_n)
                recon_params.write("sample_size_cm = %.2f\n" %sample_size_cm)
                recon_params.write("probe_energy = %.2f\n" %probe_energy[0])
                recon_params.write("probe_cts = %.2e\n" %probe_cts)
                recon_params.write("det_size_cm = %.2f\n" %det_size_cm)
                recon_params.write("det_from_sample_cm = %.2f\n" %det_from_sample_cm)
                recon_params.write("det_ds_spacing_cm = %.2f\n" %det_ds_spacing_cm)
        
                loss_epoch_cont = tc.zeros(n_epoch)
                mse_epoch_cont = tc.zeros(n_epoch, len(this_aN_dic))
                
        for epoch in range(n_epoch):
            t0_epoch = time.perf_counter()
            
            if rank == 0:
                rand_idx = tc.randperm(n_theta)
                theta_ls = theta_ls[rand_idx]  
            else:
                rand_idx = tc.ones(n_theta)

            comm.Barrier() 
            rand_idx = comm.bcast(rand_idx, root=0) 
            theta_ls = comm.bcast(theta_ls, root=0)                
             
            stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': True}
            timestr = str(datetime.datetime.today())
            print_flush_root(rank, f"epoch: {epoch}, time: {timestr}", output_file='', **stdout_options)
            
            for idx, theta in enumerate(theta_ls):
                this_theta_idx = rand_idx[idx]
                
                ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
                if selfAb == True:
                    X_ap_rot = rotate(X, theta, dev).view(n_element, sample_height_n * sample_size_n, sample_size_n) #dev
                    lac = X_ap_rot.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1) #dev
                    lac = lac.expand(-1, -1, n_voxel_batch, -1).float() #dev
                else:
                    lac = 0.
                
                ## load true data, y1: XRF_data, y2: XRT data
                y1_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRF_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32)).to(dev) #dev
                y2_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRT_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32)).to(dev) #dev
                
                for m in range(n_batch):
                    minibatch_ls = n_ranks * m + minibatch_ls_0  #dev
                    p = minibatch_ls[rank]
                    P_batch = None
                    if rank == 0:
                        ## Prepare a list of P_minibatch required by other rank(s)                            
                        P_batch = np.copy(P[:,:, p * dia_len_n * minibatch_size * sample_size_n :  (p + n_ranks) * dia_len_n * minibatch_size * sample_size_n]) #cpu
                        P_batch = P_batch.reshape((n_det, 3, n_ranks, dia_len_n * minibatch_size * sample_size_n)) #cpu
                        P_batch = np.transpose(P_batch, (2,0,1,3)) #cpu
                        P_batch = np.ascontiguousarray(P_batch, dtype=np.float32)

                    P_minibatch = np.zeros((n_det, 3, dia_len_n * minibatch_size * sample_size_n)).astype(np.float32)
           
                    comm.Barrier()
                    comm.Scatter(P_batch, P_minibatch, root=0)  #cpu
                    P_minibatch = tc.from_numpy(P_minibatch).to(dev) #dev                                               
                    comm.Barrier()
                                
                    model = PPM(dev, selfAb, lac, X, p, n_element, sample_height_n, minibatch_size,
                                 sample_size_n, sample_size_cm,
                                 this_aN_dic, probe_energy, probe_cts,
                                 theta_st, theta_end, n_theta, this_theta_idx,
                                 n_det, P_minibatch, det_size_cm, det_from_sample_cm)

                    optimizer = tc.optim.Adam(model.parameters(), lr=lr)   
            
                    stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
                    t0_fModel = time.perf_counter()
                    y1_hat, y2_hat = model()
                    print_flush_root(rank, val=fModel_time, output_file=f'forward_model_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)

                    XRF_loss = loss_fn(y1_hat, y1_true[:, minibatch_size * p : minibatch_size * (p+1)])
                    XRT_loss = loss_fn(y2_hat, y2_true[minibatch_size * p : minibatch_size * (p+1)])
                    loss = XRF_loss + b * XRT_loss 
            
                    optimizer.zero_grad()
                
                    t0_calGradient = time.perf_counter()
                    loss.backward()
                    calGradient_time = time.perf_counter() - t0_calGradient 
                    print_flush_root(rank, val=calGradient_time, output_file=f'gradient_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)                   
                            
                    t0_bPropagation = time.perf_counter()
                    optimizer.step()
                    bPropagation_time = time.perf_counter() - t0_bPropagation
                    print_flush_root(rank, val=bPropagation_time, output_file=f'backward_propagation_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                     
                    updated_minibatch = model.xp.detach().cpu()
                    comm.Barrier()
                    updated_batch = comm.gather(updated_minibatch,root=0)
                    
                    
                    ## Note that the gathered updated_batch is a list of tensor, namely a list with n_ranks tensors. 
                    ## E.g., if n_ranks=2, updated_batch = [tensor1, tensor2]
                    ## tensor1 and tensor2 both have the dim = (n_element, sample_height_n (per minibatch), sample_size_n, sample_size_n)  
                    if rank == 0: 
                        updated_batch = tc.cat(updated_batch, dim=1)
                        X[:, minibatch_size * p // sample_size_n : minibatch_size * (p + n_ranks) // sample_size_n, :, :] = updated_batch.detach()
                        X = tc.clamp(X, 0, float('inf'))
                        
                    comm.Barrier()
                    X = comm.bcast(X, root=0).to(dev)
                    comm.Barrier()
                    del model
                    
                del lac
                tc.cuda.empty_cache() 
                    
            loss = loss.detach().item()           
            loss_sum = comm.reduce(loss, op=MPI.SUM, root=0) 
            if rank == 0:
                X_cpu = X.cpu()
                loss = loss_sum / n_ranks
                loss_epoch_cont[epoch] = loss
                mse_epoch_cont[epoch] = tc.mean(tc.square(X - X_true).view(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]), dim=1)
                np.save(os.path.join(recon_path, f_recon_grid)+'.npy', X_cpu.numpy())
                
            stdout_options = {'root':0, 'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False}
            per_epoch_time = time.perf_counter() - t0_epoch
            print_flush_root(rank, val=per_epoch_time, output_file=f'per_epoch_time_mb_size_{minibatch_size}.csv', **stdout_options)

            
        if rank == 0:    
            mse_epoch_tot_cont = tc.mean(mse_epoch_cont, dim=1)
                       
            loss_epoch = tc.cat((loss_epoch, loss_epoch_cont))
            mse_epoch = tc.cat((mse_epoch, mse_epoch_cont))
            mse_epoch_tot = tc.cat((mse_epoch_tot, mse_epoch_tot_cont))


            fig6 = plt.figure(figsize=(15,5))
            gs6 = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1,1])

            fig6_ax1 = fig6.add_subplot(gs6[0,0])
            fig6_ax1.plot(loss_epoch.numpy())
            fig6_ax1.set_xlabel('epoch')
            fig6_ax1.set_ylabel('loss')
            fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))

            fig6_ax2 = fig6.add_subplot(gs6[0,1])
            fig6_ax2.plot(mse_epoch_tot.numpy())
            fig6_ax2.set_xlabel('epoch')
            fig6_ax2.set_ylabel('mse of model')
            fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
            plt.savefig(os.path.join(recon_path, 'loss_and_tot_mse.pdf'))


            fig7 = plt.figure(figsize=(X_cpu.shape[0]*6, 4))
            gs7 = gridspec.GridSpec(nrows=1, ncols=X_cpu.shape[0], width_ratios=[1]*X_cpu.shape[0])
            for i in range(X_cpu.shape[0]):
                fig7_ax1 = fig7.add_subplot(gs7[0,i])
                fig7_ax1.plot(mse_epoch[:,i].detach().numpy())
                fig7_ax1.set_xlabel('epoch')
                fig7_ax1.set_ylabel('mse of model (each element)')
                fig7_ax1.set_title(str(list(this_aN_dic.keys())[i]))
                fig7_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f')) 

            plt.savefig(os.path.join(recon_path, 'mse_model.pdf'))

            np.save(os.path.join(recon_path, 'loss_minibatch.npy'), loss_minibatch.cpu().numpy())  
            np.save(os.path.join(recon_path, 'mse_model_elements.npy'), mse_epoch_cont.cpu().numpy()) 
            np.save(os.path.join(recon_path, 'mse_model.npy'), mse_epoch_tot.cpu().numpy()) 
            dxchange.write_tiff(X.cpu().numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx), dtype='float32', overwrite=True)
        
        
        
        
        
        