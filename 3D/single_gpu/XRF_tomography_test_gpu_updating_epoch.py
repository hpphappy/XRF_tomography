#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:58:57 2020

@author: panpanhuang
"""

import os
import numpy as np
import xraylib_np as xlib_np
import time

import torch as tc
tc.set_default_tensor_type(tc.FloatTensor)
import torch.nn as nn
from tqdm import tqdm
import time
from data_generation_fns_updating import rotate, MakeFLlinesDictionary, intersecting_length_fl_detectorlet_3d
from array_ops_test_gpu import initialize_guess_3d
from forward_model_test_gpu_updating import PPM
from misc_test_gpu_updating import print_flush


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
                                 ini_kind, f_recon_parameters, n_epoch, n_minibatch, minibatch_size, b, lr, init_const, 
                                 fl_line_groups, fl_K, fl_L, fl_M, group_lines, theta_st, theta_end, n_theta,
                                 sample_size_n, sample_height_n, sample_size_cm,
                                 probe_energy, probe_cts, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, f_P,
                                ):
    stdout_options = {'output_folder': recon_path, 'save_stdout': True, 'print_terminal': False} 
    if not os.path.exists(recon_path):
        os.mkdir(recon_path) 
        
    loss_fn = nn.MSELoss()
    X_true = tc.from_numpy(np.load(os.path.join(grid_path, f_grid)).astype(np.float32)).to(dev)   
    
    dia_len_n = int((sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)
    n_voxel_minibatch = minibatch_size * sample_size_n
    n_voxel = sample_height_n * sample_size_n**2
    aN_ls = np.array(list(this_aN_dic.values()))
    
    fl_all_lines_dic = MakeFLlinesDictionary(this_aN_dic, probe_energy,
              sample_size_n.cpu().numpy(), sample_size_cm.cpu().numpy(),
              fl_line_groups, fl_K, fl_L, fl_M,
              group_lines)
    FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, fl_all_lines_dic["fl_energy"])).float().to(dev)
    n_lines = fl_all_lines_dic["n_lines"]
    
    minibatch_ls_0 = tc.arange(n_minibatch).to(dev)
    n_batch = (sample_height_n * sample_size_n) // (n_minibatch * minibatch_size) 
    theta_ls = - tc.linspace(theta_st, theta_end, n_theta+1)[:-1].to(dev)
    n_element = len(this_aN_dic)
    P_save_path = os.path.join(recon_path, f_P)
    
    longest_int_length, n_det, P = intersecting_length_fl_detectorlet_3d(det_size_cm, det_from_sample_cm, det_ds_spacing_cm,
                                              sample_size_n.cpu().numpy(), sample_size_cm.cpu().numpy(),
                                              sample_height_n.cpu().numpy(), P_save_path)
    P = tc.from_numpy(P).float()
#    P = P.view(n_det, 3, dia_len_n * sample_height_n * sample_size_n * sample_size_n) 

    if cont_from_check_point == False:
        if use_saved_initial_guess:
            X = np.load(os.path.join(recon_path, f_initial_guess)+'.npy')
            X = tc.from_numpy(X).float().to(dev)
            ## Save the initial guess which will be used in reconstruction and will be updated to the current reconstructing result
            np.save(os.path.join(recon_path, f_recon_grid)+'.npy', X.cpu())   
            
        else:
            X = initialize_guess_3d(dev, ini_kind, grid_path, f_grid, recon_path, f_recon_grid, f_initial_guess, init_const)
                
         
        with open(os.path.join(recon_path, f_recon_parameters), "w") as recon_params:
            recon_params.write("starting_epoch = 0\n")
            recon_params.write("n_epoch = %d\n" %n_epoch)
            recon_params.write(str(this_aN_dic)+"\n") 
            recon_params.write("n_minibatch = %d\n" %n_minibatch)
            recon_params.write("minibatch_size = %d\n" %minibatch_size)
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
        
        loss_minibatch = tc.zeros(n_minibatch * n_batch * n_theta * n_epoch, device = dev)
        mse_epoch = tc.zeros(n_epoch, len(this_aN_dic), device = dev)
         
    
        for epoch in tqdm(range(n_epoch)):
            t0_epoch = time.perf_counter()

            # for each epoch, load the current object and update the grid concentration that is used to calculate the absorption.
            # X dimension: [C, N, H, W]
            t0_loadCurrObj = time.perf_counter()
            X = np.load(os.path.join(recon_path, f_recon_grid)+'.npy').astype(np.float32)
            X = tc.from_numpy(X).to(dev)
            loadCurrObj_time = time.perf_counter() - t0_loadCurrObj                 
            print_flush(val=loadCurrObj_time, output_file=f'load_current_recons_obj_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                                   
            ## Copy X as X_ap in each epoch to calculate lac in the iteration of theta
            if selfAb == True:
                X_ap = tc.clone(X)

            rand_idx = tc.randperm(n_theta)
            theta_ls = theta_ls[rand_idx]                 
            for idx, theta in enumerate(theta_ls):
                this_theta_idx = rand_idx[idx]
                
                ## Calculate lac using the current X_ap. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
                t0_calLac = time.perf_counter()
                if selfAb == True:
                    X_ap_rot = rotate(X_ap, theta, dev).view(n_element, sample_height_n * sample_size_n, sample_size_n)
                    lac = X_ap_rot.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1)
                    lac = lac.expand(-1, -1, n_voxel_minibatch, -1).float()
                else:
                    lac = 0.
                calLac_time = time.perf_counter() - t0_calLac
                print_flush(val=calLac_time, output_file=f'calculate_lac_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                
                ## load data y1: XRF data, y2: XRT data
                t0_loadData = time.perf_counter()
                y1_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRF_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32)).to(dev)
                y2_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRT_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32)).to(dev)
                loadData_time = time.perf_counter() - t0_loadData
                print_flush(val=loadData_time, output_file=f'load_data_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                
                for m in range(n_batch):                  
                    minibatch_ls = n_minibatch * m + minibatch_ls_0 
                    
                    t0_distriP = time.perf_counter()
                    P_this_batch = P[:,:, minibatch_ls[0] * dia_len_n * minibatch_size * sample_size_n : 
                                        (minibatch_ls[0] + len(minibatch_ls)) * dia_len_n * minibatch_size * sample_size_n]              
                    P_this_batch = P_this_batch.view(n_det, 3, len(minibatch_ls), dia_len_n * minibatch_size * sample_size_n)
                    P_this_batch = P_this_batch.permute(2,0,1,3)
                    distriP_time = time.perf_counter() - t0_distriP
                    print_flush(val=distriP_time, output_file=f'distribute_P_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                                                                            
                    for ip, p in enumerate(minibatch_ls):                               
                        
                        t0_createModel = time.perf_counter()
                        model = PPM(dev, selfAb, lac, X, p, n_element, sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                        this_aN_dic, probe_energy, probe_cts,
                        theta_st, theta_end, n_theta, this_theta_idx,
                        n_det, P_this_batch[ip]).to(dev)
                        createModel_time = time.perf_counter() - t0_createModel
                        print_flush(val=createModel_time, output_file=f'create_model_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                        tc.cuda.empty_cache()              
                        
                        optimizer = tc.optim.Adam(model.parameters(), lr=lr)
                        
                        t0_fModel = time.perf_counter()
                        y1_hat, y2_hat = model()
                        fModel_time = time.perf_counter() - t0_fModel                  
                        print_flush(val=fModel_time, output_file=f'forward_model_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                        
                        t0_calLoss = time.perf_counter()
                        XRF_loss = loss_fn(y1_hat, y1_true[:, minibatch_size * p : minibatch_size * (p+1)])
                        XRT_loss = loss_fn(y2_hat, y2_true[minibatch_size * p : minibatch_size * (p+1)])
                        loss = XRF_loss + b * XRT_loss
                        calLoss_time = time.perf_counter() - t0_calLoss
                        print_flush(val=calLoss_time, output_file=f'calculate_loss_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                        
                        loss_minibatch[(n_minibatch * n_batch * n_theta) * epoch + (n_minibatch * n_batch) * this_theta_idx + n_minibatch * m + ip] = float(loss)
                        
                        optimizer.zero_grad()
                        
                        t0_calGradient = time.perf_counter()
                        loss.backward()
                        calGradient_time = time.perf_counter() - t0_calGradient 
                        print_flush(val=calGradient_time, output_file=f'gradient_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                        
                        t0_bPropagation = time.perf_counter()
                        optimizer.step()
                        bPropagation_time = time.perf_counter() - t0_bPropagation
                        print_flush(val=bPropagation_time, output_file=f'backward_propagation_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                        
                        X[:, minibatch_size * p // sample_size_n : minibatch_size*(p+1) // sample_size_n, :, :] = model.xp.detach()

                        del model
                    del P_this_batch
                
                X = tc.clamp(X, 0, float('inf'))
                t0_saveObj = time.perf_counter()
                np.save(os.path.join(recon_path, f_recon_grid)+'.npy', X.detach().cpu().numpy())
                saveObj_time = time.perf_counter() - t0_saveObj
                print_flush(val=saveObj_time, output_file=f'save_object_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                
                del lac
                tc.cuda.empty_cache()           
            mse_epoch[epoch] = tc.mean(tc.square(X - X_true).view(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]), dim=1)
            per_epoch_time = time.perf_counter() - t0_epoch
            print_flush(val=per_epoch_time, output_file=f'per_epoch_time_mb_size_{minibatch_size}.csv', **stdout_options)
            tqdm._instances.clear()   
            
        mse_epoch_tot = tc.mean(mse_epoch, dim=1)
        
        fig6 = plt.figure(figsize=(15,5))
        gs6 = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1,1])
    
        fig6_ax1 = fig6.add_subplot(gs6[0,0])
        fig6_ax1.plot(loss_minibatch.detach().cpu().numpy())
        fig6_ax1.set_xlabel('minibatch')
        fig6_ax1.set_ylabel('loss')
        fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
        
        fig6_ax2 = fig6.add_subplot(gs6[0,1])
        fig6_ax2.plot(mse_epoch_tot.detach().cpu().numpy())
        fig6_ax2.set_xlabel('epoch')
        fig6_ax2.set_ylabel('mse of model')
        fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        plt.savefig(os.path.join(recon_path, 'loss_and_tot_mse.pdf'))
        
        fig7 = plt.figure(figsize=(X.shape[0]*6, 4))
        gs7 = gridspec.GridSpec(nrows=1, ncols=X.shape[0], width_ratios=[1]*X.shape[0])
        for i in range(X.shape[0]):
            fig7_ax1 = fig7.add_subplot(gs7[0,i])
            fig7_ax1.plot(mse_epoch[:,i].detach().cpu().numpy())
            fig7_ax1.set_xlabel('epoch')
            fig7_ax1.set_ylabel('mse of model (each element)')
            fig7_ax1.set_title(str(list(this_aN_dic.keys())[i]))
            fig7_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))         
        plt.savefig(os.path.join(recon_path, 'mse_model.pdf'))
        
        np.save(os.path.join(recon_path, 'loss_minibatch.npy'), loss_minibatch.detach().cpu().numpy()) 
        np.save(os.path.join(recon_path, 'mse_model_elements.npy'), mse_epoch.detach().cpu().numpy())
        np.save(os.path.join(recon_path, 'mse_model.npy'), mse_epoch_tot.detach().cpu().numpy()) 
        dxchange.write_tiff(X.detach().cpu().numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx), dtype='float32', overwrite=True)
        
    if cont_from_check_point == True:
        recon_idx += 1
        
        loss_minibatch =  tc.from_numpy(np.load(os.path.join(recon_path, 'loss_minibatch.npy')).astype(np.float32))
        mse_epoch = tc.from_numpy(np.load(os.path.join(recon_path, 'mse_model_elements.npy')).astype(np.float32))
        mse_epoch_tot = tc.from_numpy(np.load(os.path.join(recon_path, 'mse_model.npy')).astype(np.float32))
            
        with open(os.path.join(recon_path, f_recon_parameters), "r") as recon_params:
            params_list = []
            for line in recon_params.readlines():
                params_list.append(line.rstrip("\n"))
            n_ending = len(params_list)
            
        with open(os.path.join(recon_path, f_recon_parameters), "a") as recon_params:
            n_start_last = n_ending - 18
            
            previous_epoch = int(params_list[n_start_last][params_list[n_start_last].find("=")+1:])   
            recon_params.write("\n")
            recon_params.write("###########################################\n")
            recon_params.write("starting_epoch = %d\n" %(previous_epoch + n_epoch))
            recon_params.write("n_epoch = %d\n" %n_epoch)
            recon_params.write(str(this_aN_dic)+"\n") 
            recon_params.write("n_minibatch = %d\n" %n_minibatch)
            recon_params.write("minibatch_size = %d\n" %minibatch_size)
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
            
        del recon_params
        del params_list
        
        loss_minibatch_cont = tc.zeros(n_minibatch * n_batch * n_theta * n_epoch, device = dev)
        mse_epoch_cont = tc.zeros(n_epoch, len(this_aN_dic), device = dev)
        
        rand_idx = tc.randperm(n_theta)
        theta_ls = theta_ls[rand_idx]        
        for epoch in tqdm(range(n_epoch)):
            t0_epoch = time.perf_counter()
            for idx, theta in enumerate(theta_ls):
                this_theta_idx = rand_idx[idx]
                
                # for each theta, load the current object and update the grid concentration that is used to calculate the absorption.
                # X dimension: [C, N, H, W]
                t0_loadCurrObj = time.perf_counter()
                X = np.load(os.path.join(recon_path, f_recon_grid)+'.npy').astype(np.float32)
                X = tc.from_numpy(X).to(dev)
                loadCurrObj_time = time.perf_counter() - t0_loadCurrObj                 
                print_flush(val=loadCurrObj_time, output_file=f'load_current_recons_obj_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
      
                ## Calculate lac using the current X. lac (linear attenuation coefficient) has the dimension of [n_element, n_lines, n_voxel_minibatch, n_voxel]
                t0_calLac = time.perf_counter()
                if selfAb == True:
                    X_ap_rot = rotate(X, theta, dev).view(n_element, sample_height_n * sample_size_n, sample_size_n)
                    lac = X_ap_rot.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1)
                    lac = lac.expand(-1, -1, n_voxel_minibatch, -1).float()
                else:
                    lac = 0.
                calLac_time = time.perf_counter() - t0_calLac
                print_flush(val=calLac_time, output_file=f'calculate_lac_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
            
                ## load data y1: XRF data, y2: XRT data
                t0_loadData = time.perf_counter()
                y1_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRF_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32)).to(dev)
                y2_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRT_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32)).to(dev)
                loadData_time = time.perf_counter() - t0_loadData
                print_flush(val=loadData_time, output_file=f'load_data_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                
                for m in range(n_batch):
                    minibatch_ls = n_minibatch * m + minibatch_ls_0
                    
                    t0_distriP = time.perf_counter()
                    P_this_batch = P[:,:, minibatch_ls[0] * dia_len_n * minibatch_size * sample_size_n : 
                                        (minibatch_ls[0] + len(minibatch_ls)) * dia_len_n * minibatch_size * sample_size_n]              
                    P_this_batch = P_this_batch.view(n_det, 3, len(minibatch_ls), dia_len_n * minibatch_size * sample_size_n)
                    P_this_batch = P_this_batch.permute(2,0,1,3)
                    distriP_time = time.perf_counter() - t0_distriP
                    print_flush(val=distriP_time, output_file=f'distribute_P_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)                    
                    
                    for ip, p in enumerate(minibatch_ls):
                        
                        t0_createModel = time.perf_counter()
                        model = PPM(dev, selfAb, lac, X, p, n_element, sample_height_n, minibatch_size, sample_size_n, sample_size_cm,
                        this_aN_dic, probe_energy, probe_cts,
                        theta_st, theta_end, n_theta, this_theta_idx,
                        n_det, P_this_batch[ip]).to(dev)
                        createModel_time = time.perf_counter() - t0_createModel
                        print_flush(val=createModel_time, output_file=f'create_model_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)                                 
                        tc.cuda.empty_cache()
                        
                        optimizer = tc.optim.Adam(model.parameters(), lr=lr)                   
    
                        t0_fModel = time.perf_counter()
                        y1_hat, y2_hat = model()
                        fModel_time = time.perf_counter() - t0_fModel                    
                        print_flush(val=fModel_time, output_file=f'forward_model_computing_time_{minibatch_size}.csv', **stdout_options)

                    
                        t0_calLoss = time.perf_counter()
                        XRF_loss = loss_fn(y1_hat, y1_true[:, minibatch_size * p : minibatch_size * (p+1)])
                        XRT_loss = loss_fn(y2_hat, y2_true[minibatch_size * p : minibatch_size * (p+1)])
                        loss = XRF_loss + b * XRT_loss
                        calLoss_time = time.perf_counter() - t0_calLoss
                        print_flush(val=calLoss_time, output_file=f'calculate_loss_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)                    
                        
                        loss_minibatch_cont[(n_minibatch * n_batch * n_theta) * epoch + (n_minibatch * n_batch) * this_theta_idx + n_minibatch * m + ip] = float(loss)                
                        optimizer.zero_grad()
                        
                        t0_calGradient = time.perf_counter()
                        loss.backward()
                        calGradient_time = time.perf_counter() - t0_calGradient 
                        print_flush(val=calGradient_time, output_file=f'gradient_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                        
                        t0_bPropagation = time.perf_counter()
                        optimizer.step()
                        bPropagation_time = time.perf_counter() - t0_bPropagation
                        print_flush(val=bPropagation_time, output_file=f'backward_propagation_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)
                        
                        X[:, minibatch_size * p // sample_size_n : minibatch_size*(p+1) // sample_size_n, :, :] = model.xp.detach()
                        del model
                        del P_this_batch

                X = tc.clamp(X, 0, float('inf'))
                t0_saveObj = time.perf_counter()
                np.save(os.path.join(recon_path, f_recon_grid)+'.npy', X.detach().cpu().numpy())
                saveObj_time = time.perf_counter() - t0_saveObj
                print_flush(val=saveObj_time, output_file=f'save_object_computing_time_mb_size_{minibatch_size}.csv', **stdout_options)                        
                        
                del lac
                tc.cuda.empty_cache()          
            mse_epoch_cont[epoch] = tc.mean(tc.square(X - X_true).view(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]), dim=1)
            per_epoch_time = time.perf_counter() - t0_epoch
            print_flush(val=per_epoch_time, output_file=f'per_epoch_time_mb_size_{minibatch_size}.csv', **stdout_options)
            tqdm._instances.clear() 

        mse_epoch_tot_cont = tc.mean(mse_epoch_cont, dim=1)
        
        loss_minibatch = tc.cat((loss_minibatch, loss_minibatch_cont.cpu()))
        mse_epoch = tc.cat((mse_epoch, mse_epoch_cont.cpu()))
        mse_epoch_tot = tc.cat((mse_epoch_tot, mse_epoch_tot_cont.cpu()))
     
        fig6 = plt.figure(figsize=(15,5))
        gs6 = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1,1])
    
        fig6_ax1 = fig6.add_subplot(gs6[0,0])
        fig6_ax1.plot(loss_minibatch.detach().cpu().numpy())
        fig6_ax1.set_xlabel('minibatch')
        fig6_ax1.set_ylabel('loss')
        fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
        
        fig6_ax2 = fig6.add_subplot(gs6[0,1])
        fig6_ax2.plot(mse_epoch_tot.detach().cpu().numpy())
        fig6_ax2.set_xlabel('epoch')
        fig6_ax2.set_ylabel('mse of model')
        fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        plt.savefig(os.path.join(recon_path, 'loss_and_tot_mse.pdf'))
        
        
        fig7 = plt.figure(figsize=(X.shape[0]*6, 4))
        gs7 = gridspec.GridSpec(nrows=1, ncols=X.shape[0], width_ratios=[1]*X.shape[0])
        for i in range(X.shape[0]):
            fig7_ax1 = fig7.add_subplot(gs7[0,i])
            fig7_ax1.plot(mse_epoch_cont[:,i].detach().cpu().numpy())
            fig7_ax1.set_xlabel('epoch')
            fig7_ax1.set_ylabel('mse of model (each element)')
            fig7_ax1.set_title(str(list(this_aN_dic.keys())[i]))
            fig7_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f')) 
    
        plt.savefig(os.path.join(recon_path, 'mse_model.pdf'))
        
        np.save(os.path.join(recon_path, 'loss_minibatch.npy'), loss_minibatch.detach().cpu().numpy())
        np.save(os.path.join(recon_path, 'mse_model_elements.npy'), mse_epoch.detach().cpu().numpy()) 
        np.save(os.path.join(recon_path, 'mse_model.npy'), mse_epoch_tot.detach().cpu().numpy()) 
        dxchange.write_tiff(X.detach().cpu().numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx), dtype='float32', overwrite=True)
        
        
        
        
        
        