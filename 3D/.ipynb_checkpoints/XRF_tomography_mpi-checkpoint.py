#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 15:58:57 2020

@author: panpanhuang
"""

import os
import numpy as np
from mpi4py import MPI
import xraylib_np as xlib_np

import torch as tc
tc.set_default_tensor_type(tc.FloatTensor)
import torch.nn as nn
from tqdm import tqdm
import time
from data_generation_fns import rotate, MakeFLlinesDictionary, intersecting_length_fl_detectorlet_3d
from array_ops import initialize_guess_3d
from forward_model import PPM, PPM_cont

import dxchange

import warnings
warnings.filterwarnings("ignore")

                                                     
def reconstruct_jXRFT_tomography(dev, recon_idx, cont_from_check_point, use_saved_initial_guess, recon_path, f_initial_guess, f_recon_grid,
                                 grid_path, f_grid, data_path, f_XRF_data, f_XRT_data, this_aN_dic,
                                 ini_kind, f_recon_parameters, n_epoch, n_minibatch, minibatch_size, b, lr, init_const, 
                                 fl_line_groups, fl_K, fl_L, fl_M, group_lines, theta_st, theta_end, n_theta,
                                 sample_size_n, sample_height_n, sample_size_cm,
                                 probe_energy, probe_cts, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, f_P,
                                 ):

    comm = MPI.COMM_WORLD
    n_ranks = comm.Get_size()
    rank = comm.Get_rank()
    
    loss_fn = nn.MSELoss()
    X_true = tc.from_numpy(np.load(os.path.join(grid_path, f_grid)).astype(np.float32))
    dia_len_n = int((sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)
    n_voxel_batch = minibatch_size * sample_size_n
    n_voxel = sample_height_n * sample_size_n**2
    aN_ls = np.array(list(this_aN_dic.values()))
    
    fl_all_lines_dic = MakeFLlinesDictionary(this_aN_dic, probe_energy,
              sample_size_n.numpy(), sample_size_cm.numpy(),
              fl_line_groups, fl_K, fl_L, fl_M,
              group_lines)
    FL_line_attCS_ls = tc.as_tensor(xlib_np.CS_Total(aN_ls, fl_all_lines_dic["fl_energy"])).float()
    n_lines = fl_all_lines_dic["n_lines"]
    
    minibatch_ls_0 = tc.arange(n_minibatch)
    n_batch = (sample_height_n * sample_size_n) // (n_minibatch * minibatch_size) 
    theta_ls = - tc.linspace(theta_st, theta_end, n_theta+1)[:-1]
    n_element = len(this_aN_dic)
    P_save_path = os.path.join(recon_path, f_P)
    
    if rank == 0: 
        n_det, P = intersecting_length_fl_detectorlet_3d(det_size_cm, det_from_sample_cm, det_ds_spacing_cm,
                                                  sample_size_n.cpu().numpy(), sample_size_cm.cpu().numpy(),
                                                  sample_height_n.cpu().numpy(), P_save_path)
        P = P.float()      
#         P = P.view(n_det, 3, dia_len_n * sample_height_n * sample_size_n * sample_size_n) 

        if not os.path.exists(recon_path):
            os.mkdir('recon_path')
        
    else:
        n_det = None
        
    comm.Barrier()    
    comm.Bcast(n_det, root=0)
    
    if cont_from_check_point == False:        
        if rank == 0:
            if use_saved_initial_guess:
                X = np.load(os.path.join(recon_path, f_initial_guess)+'.npy')
                X = tc.from_numpy(X).float()  
            else:
                X = initialize_guess_3d(dev, ini_kind, grid_path, f_grid, recon_path, f_recon_grid, f_initial_guess, init_const)
                
            with open(os.path.join(recon_path, f_recon_parameters), "w") as recon_params:
                recon_params.write("starting_epoch = 0\n")
                recon_params.write("n_epoch = %d\n" %n_epoch)
                recon_params.write(str(this_aN_dic)+"\n") 
                recon_params.write("n_minibatch = %d\n" %n_minibatch)
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

                loss_minibatch = tc.zeros(n_minibatch * n_batch * n_theta * n_epoch)
                mse_epoch = tc.zeros(n_epoch, len(this_aN_dic)) 
                
        else:
            pass
        comm.Barrier()  
               
        for epoch in tqdm(range(n_epoch)):
            for this_theta_idx, theta in enumerate(theta_ls):
        #         print("this_theta_idx = %d" %(this_theta_idx))
                X_ap = np.load(os.path.join(recon_path, f_recon_grid) + '.npy').astype(np.float32)
                X_ap = tc.from_numpy(X_ap)
                
                ## Calculate the linear attenuation coefficient (lac) using the current X_ap
                theta = theta_ls[this_theta_idx]
                ap_map_rot = rotate(X_ap, theta, dev).view(n_element, sample_height_n * sample_size_n, sample_size_n)        
                lac = ap_map_rot.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1)
                lac = lac.expand(-1, -1, n_voxel_batch, -1).float()
                
                ## load true data, y1: XRF_data, y2: XRT data
                y1_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRF_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32))
                y2_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRT_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32))
             
                for m in range(n_batch):
                    
                    minibatch_ls = n_minibatch * m + minibatch_ls_0
                    p = minibatch_ls[rank]
                    if rank == 0:
                        ## Prepare a list of P_batch required by other rank(s)                            
                        P_batch = P[:,:, p * dia_len_n * minibatch_size * sample_size_n :  (p + n_ranks) * dia_len_n * minibatch_size * sample_size_n]
                        P_batch = P_batch.view(n_det, 3, n_ranks, dia_len_n * minibatch_size * sample_size_n)
                        P_batch = P_batch.premute(2,0,1,3)
                                                           
                    else:
                        P_batch = None
           
                    comm.Barrier()
                    comm.Scatter(P_batch, root=0)
                    
                    if epoch == 0 and theta_idx == 0 and m == 0:
                        # print("check point 0 (before creating model):")
                        # %memit
                        # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                        # t0 = time.perf_counter()
                        
                        model = PPM(dev, lac, ini_kind, init_const, X, p, n_element, sample_height_n, minibatch_size,
                                    sample_size_n, sample_size_cm,
                                    this_aN_dic, probe_energy, probe_cts,
                                    theta_st, theta_end, n_theta, this_theta_idx,
                                    n_det, P_batch)                                                    
        
                        # t1 = time.perf_counter()
                        # print("creating models takes %g s \n" %(t1 - t0))
                        # print("check point 1 (model created):")
                        # %memit
                        # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                    
                    else:
                        # print("check point 0 (before creating model):")
                        # %memit 
                        # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                        # t0 = time.perf_counter()                       
                        
                        model = PPM_cont(dev, lac, X, p, n_element, sample_height_n, minibatch_size,
                                         sample_size_n, sample_size_cm,
                                         this_aN_dic, probe_energy, probe_cts,
                                         theta_st, theta_end, n_theta, this_theta_idx,
                                         n_det, P_batch)

                        # t1 = time.perf_counter()
                        # print("creating models takes %g s \n" %(t1 - t0))
                        # print("check point 1 (model created):")
                        # %memit
                        # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                                    
                    # t2 = time.perf_counter()
                    
                    optimizer = tc.optim.Adam(model[rank].parameters(), lr=lr)              
                    ## loading temparary reconstructing results

                    # print("check point 2 (beginning loading temp. recon.):")
                    # %memit
                    # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                    X = np.load(os.path.join(recon_path, f_recon_grid) + '.npy').astype(np.float32)
                    X = tc.from_numpy(X)
                    # t3 = time.perf_counter()
                    # print("creating optimizer and loading recon file takes %g s \n" %(t3 - t2))

                    ## forward propagation
                    # print("check point 3 (forward propagation starts):")
                    # %memit
                    # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                    y1_hat, y2_hat, concentration_map_rot = model[rank](X, this_theta_idx)
                    # t4 = time.perf_counter()
                    # print("forward propagation takes %g s \n" %(t4 - t3))

                    ## calculating loss
                    # print("check point 4 (calculating loss):")
                    # %memit
                    # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                    XRF_loss = loss_fn(y1_hat, y1_true[:, minibatch_size * p : minibatch_size * (p+1)])
                    XRT_loss = loss_fn(y2_hat, y2_true[minibatch_size * p : minibatch_size * (p+1)])
                    loss = XRF_loss + b * XRT_loss
                    
                    # t5 = time.perf_counter()
                    # print("calculating loss takes %g s \n" %(t5 - t4))

                    ## recording loss
                    # print("check point 5 (append loss to the loss list):")
                    # %memit
                    # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                    loss_minibatch[(n_minibatch * n_batch * n_theta) * epoch + (n_minibatch * n_batch) * this_theta_idx + n_minibatch * m + rank] = loss.detach()
                    # t6 = time.perf_counter()
                    # print("appending loss takes %g s \n" %(t6 - t5))

                    ### backward propagation starts
                    # print("check point 6 (initializing gradient):")
                    # %memit
                    # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                    optimizer.zero_grad()

                    # print("check point 7 (calculating gradient) :")
                    # %memit
                    # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                    loss.backward()  # this step reserves the memory for gradient and gradient^2 if uses Adam
                    
                    # print("check point 8 (updating parameters) :")
                    # %memit
                    # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                    optimizer.step()
                    # t7 = time.perf_counter()
                    # print("backward propagation takes %g s \n" %(t7 - t6))

                    # print("check point 9 (storing temp. recon. results) :\n")
                    # %memit
                    # print("current reserved memory: %g GB" %(tc.cuda.memory_stats(dev)['reserved_bytes.all.current']/1.0E9))
                    updated_minibatch = model.xp
                    updated_batch = comm.Gather(updated_minibatch,root=0).detach()
                    
                    ## Note that the gathered updated_batch has the dimension of (n_ranks, n_element, minibactch_size, sample_size_n)
                    ## need to do permutation to swap axis-0 and axis-1
                    ## After permutation, the dimension becomes (n_element, n_ranks, minibactch_size, sample_size_n)                    
                    if rank == 0:   
                        updated_batch = updated_batch.permute(1,0,2,3)
                        updated_batch = updated_batch.view(n_element, n_ranks * minibactch_size, sample_size_n)
                        concentration_map_rot[:, minibatch_size * p : minibatch_size * (p + n_ranks),:] = updated_batch
                        
                        X_update = rotate(concentration_map_rot.view(n_element, sample_height_n, sample_size_n, sample_size_n), -theta, dev)  
                        X = tc.clamp(X_update, 0, float('inf'))
                        np.save(os.path.join(recon_path, f_recon_grid)+'.npy', X.cpu().numpy())
                        
                    # t8 = time.perf_counter()
                    # print("storing temp. recon. results takes %g s" %(t8 - t7))
                    # print("total time: %g s \n" %(t8 - t0))
                    # print("\n") 
            if rank == 0:           
                mse_epoch[epoch] = tc.mean(tc.square(X - X_true).view(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]), dim=1)
                
            tqdm._instances.clear()  
            
        if rank == 0:    
            mse_epoch_tot = tc.mean(mse_epoch, dim=1)
        
            # fig6 = plt.figure(figsize=(15,5))
            # gs6 = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1,1])

            # fig6_ax1 = fig6.add_subplot(gs6[0,0])
            # fig6_ax1.plot(loss_minibatch.detach().numpy())
            # fig6_ax1.set_xlabel('minibatch')
            # fig6_ax1.set_ylabel('loss')
            # fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))

            # fig6_ax2 = fig6.add_subplot(gs6[0,1])
            # fig6_ax2.plot(mse_epoch_tot.detach().numpy())
            # fig6_ax2.set_xlabel('epoch')
            # fig6_ax2.set_ylabel('mse of model')
            # fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
            # plt.savefig(os.path.join(recon_path, 'loss_and_tot_mse.pdf'))


            # fig7 = plt.figure(figsize=(X.shape[0]*6, 4))
            # gs7 = gridspec.GridSpec(nrows=1, ncols=X.shape[0], width_ratios=[1]*X.shape[0])
            # for i in range(X.shape[0]):
            #     fig7_ax1 = fig7.add_subplot(gs7[0,i])
            #     fig7_ax1.plot(mse_epoch[:,i].detach().numpy())
            #     fig7_ax1.set_xlabel('epoch')
            #     fig7_ax1.set_ylabel('mse of model (each element)')
            #     fig7_ax1.set_title(str(list(this_aN_dic.keys())[i]))
            #     fig7_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))         
            # plt.savefig(os.path.join(recon_path, 'mse_model.pdf'))
        
            np.save(os.path.join(recon_path, 'loss_minibatch.npy'), loss_minibatch.detach().cpu().numpy())  
            np.save(os.path.join(recon_path, 'mse_model.npy'), mse_epoch_tot.detach().cpu().numpy()) 
            dxchange.write_tiff(X.detach().cpu().numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx), dtype='float32', overwrite=True)
        
    if cont_from_check_point == True:
        recon_idx += 1
        
        if rak == 0:
            loss_minibatch =  tc.from_numpy(np.load(os.path.join(recon_path, 'loss_minibatch.npy')).astype(np.float32))
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
                recon_params.write("n_minibatch = %d\n" %n_minibatch)
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
        
                loss_minibatch_cont = tc.zeros(n_minibatch * n_batch * n_theta * n_epoch).to(dev)
                mse_epoch_cont = tc.zeros(n_epoch, len(this_aN_dic)).to(dev)
    
        for epoch in tqdm(range(n_epoch)):
            for this_theta_idx, theta in enumerate(theta_ls):
                X_ap = np.load(os.path.join(recon_path, f_recon_grid) + '.npy').astype(np.float32)
                X_ap = tc.from_numpy(X_ap)
                
                ## Calculate the linear attenuation coefficient (lac) using the current X_ap
                theta = theta_ls[this_theta_idx]
                ap_map_rot = rotate(X_ap, theta, dev).view(n_element, sample_height_n * sample_size_n, sample_size_n)        
                lac = ap_map_rot.view(n_element, 1, 1, n_voxel) * FL_line_attCS_ls.view(n_element, n_lines, 1, 1)
                lac = lac.expand(-1, -1, n_voxel_batch, -1).float()
                
                ## load true data, y1: XRF_data, y2: XRT data
                y1_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRF_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32))
                y2_true = tc.from_numpy(np.load(os.path.join(data_path, f_XRT_data)+'_{}'.format(this_theta_idx)+'.npy').astype(np.float32))
                
                for m in range(n_batch):
                    
                    minibatch_ls = n_minibatch * m + minibatch_ls_0
                    p = minibatch_ls[rank]
                    if rank == 0:
                        ## Prepare a list of P_batch required by other rank(s)                            
                        P_batch = P[:,:, p * dia_len_n * minibatch_size * sample_size_n :  (p + n_ranks) * dia_len_n * minibatch_size * sample_size_n]
                        P_batch = P_batch.view(n_det, 3, len(minibatch_ls), dia_len_n * minibatch_size * sample_size_n)
                        P_batch = P_batch.premute(2,0,1,3)
                                                           
                    else:
                        P_batch = None

                    comm.Barrier()
                    comm.scatter(P_batch, root=0)                   
#                     tc.cuda.empty_cache()
                                
                    model = PPM_cont(dev, lac, X, p, n_element, sample_height_n, minibatch_size,
                                     sample_size_n, sample_size_cm,
                                     this_aN_dic, probe_energy, probe_cts,
                                     theta_st, theta_end, n_theta, this_theta_idx,
                                     n_det, P_batch)
        
                    optimizer = tc.optim.Adam(model[rank].parameters(), lr=lr)   
            
                    X = np.load(os.path.join(recon_path, f_recon_grid) + '.npy').astype(np.float32)
                    X = tc.from_numpy(X)
                    y1_hat, y2_hat, concentration_map_rot = model[rank](X, this_theta_idx)
                    

                    XRF_loss = loss_fn(y1_hat, y1_true[:, minibatch_size * p : minibatch_size * (p+1)])
                    XRT_loss = loss_fn(y2_hat, y2_true[minibatch_size * p : minibatch_size * (p+1)])
    
                    loss = XRF_loss + b * XRT_loss 
                        
                    loss_minibatch_cont[(n_minibatch * n_batch * n_theta) * epoch + (n_minibatch * n_batch) * this_theta_idx + n_minibatch * m + rank] = loss.detach()
    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    updated_minibatch = model.xp
                    updated_batch = comm.Gather(updated_minibatch,root=0).detach 
                    
                    ## Note that the gathered updated_batch has the dimension of (n_ranks, n_element, minibactch_size, sample_size_n)
                    ## need to do permutation to swap axis-0 and axis-1
                    ## After permutation, the dimension becomes (n_element, n_ranks, minibactch_size, sample_size_n)    
                    if rank == 0:   
                        updated_batch = updated_batch.permute(1,0,2,3)
                        updated_batch = updated_batch.view(n_element, n_ranks * minibactch_size, sample_size_n)
                        concentration_map_rot[:, minibatch_size * p : minibatch_size * (p + n_ranks),:] = updated_batch
                        
                        X_update = rotate(concentration_map_rot.view(n_element, sample_height_n, sample_size_n, sample_size_n), -theta, dev)  
                        X = tc.clamp(X_update, 0, float('inf'))
                        np.save(os.path.join(recon_path, f_recon_grid)+'.npy', X.cpu().numpy())                   
                    
            if rank == 0:            
                mse_epoch_cont[epoch] = tc.mean(tc.square(X - X_true).view(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]), dim=1)
            tqdm._instances.clear() 
            
        if rank == 0:    
            mse_epoch_tot_cont = tc.mean(mse_epoch_cont, dim=1)
            
            
        loss_minibatch = tc.cat((loss_minibatch, loss_minibatch_cont))
        mse_epoch_tot = tc.cat((mse_epoch_tot, mse_epoch_tot_cont))

        
        
        # fig6 = plt.figure(figsize=(15,5))
        # gs6 = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[1,1])
    
        # fig6_ax1 = fig6.add_subplot(gs6[0,0])
        # fig6_ax1.plot(loss_minibatch.detach().numpy())
        # fig6_ax1.set_xlabel('minibatch')
        # fig6_ax1.set_ylabel('loss')
        # fig6_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.5e'))
        
        # fig6_ax2 = fig6.add_subplot(gs6[0,1])
        # fig6_ax2.plot(mse_epoch_tot.detach().numpy())
        # fig6_ax2.set_xlabel('epoch')
        # fig6_ax2.set_ylabel('mse of model')
        # fig6_ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
        # plt.savefig(os.path.join(recon_path, 'loss_and_tot_mse.pdf'))
        
        
        # fig7 = plt.figure(figsize=(X.shape[0]*6, 4))
        # gs7 = gridspec.GridSpec(nrows=1, ncols=X.shape[0], width_ratios=[1]*X.shape[0])
        # for i in range(X.shape[0]):
        #     fig7_ax1 = fig7.add_subplot(gs7[0,i])
        #     fig7_ax1.plot(mse_epoch[:,i].detach().numpy())
        #     fig7_ax1.set_xlabel('epoch')
        #     fig7_ax1.set_ylabel('mse of model (each element)')
        #     fig7_ax1.set_title(str(list(this_aN_dic.keys())[i]))
    #         fig7_ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f')) 
    
        # plt.savefig(os.path.join(recon_path, 'mse_model.pdf'))
        
        np.save(os.path.join(recon_path, 'loss_minibatch.npy'), loss_minibatch.cpu().numpy())  
        np.save(os.path.join(recon_path, 'mse_model.npy'), mse_epoch_tot.cpu().numpy()) 
        dxchange.write_tiff(X.cpu().numpy(), os.path.join(recon_path, f_recon_grid)+"_"+str(recon_idx), dtype='float32', overwrite=True)
        
        
        
        
        
        