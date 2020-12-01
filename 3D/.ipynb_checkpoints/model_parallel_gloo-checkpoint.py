import os
import torch.distributed as dist
from torch.multiprocessing import Process
from  time import sleep
import datetime

        
def init_processes(rank, size, fn,
                   dev, recon_idx, cont_from_check_point, use_saved_initial_guess, recon_path, f_initial_guess, f_recon_grid,
                   grid_path, f_grid, data_path, f_XRF_data, f_XRT_data, this_aN_dic,
                   ini_kind, f_recon_parameters, n_epoch, n_minibatch, minibatch_size, b, lr, init_const, 
                   fl_line_groups, fl_K, fl_L, fl_M, group_lines, theta_st, theta_end, n_theta,
                   sample_size_n, sample_height_n, sample_size_cm,
                   probe_energy, probe_cts, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, f_P, backend='gloo'):
    """ Initialize the distributed environment. """
    
    print('{} : started process for rank : {}'.format(os.getpid(),rank))
    
    #Remove init_method if initializing through environment variable
    dist.init_process_group(backend = backend,
                            init_method='tcp://127.0.0.1:29500', #Provide rank 0 IP and open port
                            rank=rank,
                            world_size=size,
                            timeout=datetime.timedelta(0,seconds =  20))
    fn(rank, size, dev, recon_idx, cont_from_check_point, use_saved_initial_guess, recon_path, f_initial_guess, f_recon_grid,
                        grid_path, f_grid, data_path, f_XRF_data, f_XRT_data, this_aN_dic,
                        ini_kind, f_recon_parameters, n_epoch, n_minibatch, minibatch_size, b, lr, init_const, 
                        fl_line_groups, fl_K, fl_L, fl_M, group_lines, theta_st, theta_end, n_theta,
                        sample_size_n, sample_height_n, sample_size_cm,
                        probe_energy, probe_cts, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, f_P)


def startprocesses(ranks, size, *kwargs, fn):
    processes = []
    for rank in ranks:
        p = Process(target=init_processes, args=(rank, size, fn,
                                                 dev, recon_idx, cont_from_check_point, use_saved_initial_guess, recon_path, f_initial_guess, f_recon_grid,
                                                 grid_path, f_grid, data_path, f_XRF_data, f_XRT_data, this_aN_dic,
                                                 ini_kind, f_recon_parameters, n_epoch, n_minibatch, minibatch_size, b, lr, init_const, 
                                                 fl_line_groups, fl_K, fl_L, fl_M, group_lines, theta_st, theta_end, n_theta,
                                                 sample_size_n, sample_height_n, sample_size_cm,
                                                 probe_energy, probe_cts, det_size_cm, det_from_sample_cm, det_ds_spacing_cm, f_P))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print('finished')


    