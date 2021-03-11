#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:58:22 2020

@author: panpanhuang
"""

import os
import numpy as np
import torch as tc
from data_generation_fns_updating import create_XRF_data_3d
import warnings
warnings.filterwarnings("ignore")

dev = "cpu"



params_3d_5_5_5 = {'P_save_path': os.path.join('./data/sample3_recon', 'Intersecting_Length_5_5_5'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(16).to(dev),
                   'src_path': os.path.join('./data/sample3_pad', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(5).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(5).to(dev),
                   'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample3_data',
                   'save_fname': 'XRF_sample3',
                   'dev': dev
                  }

params_3d_64_64_64 = {'P_save_path': os.path.join('./data/sample1_recon', 'Intersecting_Length_64_64_64'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev),
                   'src_path': os.path.join('./data/sample1_pad', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(64).to(dev),
                   'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample1_data',
                   'save_fname': 'XRF_sample1',
                   'dev': dev
                  }

params_3d_64_64_64_nElements_2 = {'P_save_path': os.path.join('./data/sample2_recon', 'Intersecting_Length_64_64_64'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev),
                   'src_path': os.path.join('./data/sample2_pad/nElements_2', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(64).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample2_data/nElements_2',
                   'save_fname': 'XRF_sample2',
                   'dev': dev
                  }

params_3d_64_64_64_nElements_1 = {'P_save_path': os.path.join('./data/sample2_recon', 'Intersecting_Length_64_64_64'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev),
                   'src_path': os.path.join('./data/sample2_pad/nElements_1', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(64).to(dev),
                   'this_aN_dic': {"Ca": 20},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample2_data/nElements_1',
                   'save_fname': 'XRF_sample2',
                   'dev': dev
                  }

params_3d_64_64_64_nElements_4 = {'P_save_path': os.path.join('./data/sample2_recon', 'Intersecting_Length_64_64_64'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev),
                   'src_path': os.path.join('./data/sample2_pad/nElements_4', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(64).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21, "Ti": 22, "V":23},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample2_data/nElements_4',
                   'save_fname': 'XRF_sample2',
                   'dev': dev
                  }

params_3d_64_64_64_nElements_8 = {'P_save_path': os.path.join('./data/sample2_recon', 'Intersecting_Length_64_64_64'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev),
                   'src_path': os.path.join('./data/sample2_pad/nElements_8', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(64).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21, "Ti": 22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample2_data/nElements_8',
                   'save_fname': 'XRF_sample2',
                   'dev': dev
                  }

params_3d_64_64_64_nElements_12 = {'P_save_path': os.path.join('./data/sample2_recon', 'Intersecting_Length_64_64_64'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev),
                   'src_path': os.path.join('./data/sample2_pad/nElements_12', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(64).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21, "Ti": 22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27,
                                   "Ni": 28, "Cu":29, "Zn":30, "Mo":42},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample2_data/nElements_12',
                   'save_fname': 'XRF_sample2',
                   'dev': dev
                  }

params_3d_8_8_8_nElements_2 = {'P_save_path': os.path.join('./data/sample6_size_8_recon', 'Intersecting_Length_8_8_8'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(25).to(dev),
                   'src_path': os.path.join('./data/sample6_size_8_pad/nElements_2', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(8).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(8).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample6_size_8_data/nElements_2',
                   'save_fname': 'XRF_sample6',
                   'dev': dev
                  }

params_3d_16_16_16_nElements_2 = {'P_save_path': os.path.join('./data/sample4_size_16_recon', 'Intersecting_Length_16_16_16'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(50).to(dev),
                   'src_path': os.path.join('./data/sample4_size_16_pad/nElements_2', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(16).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(16).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample4_size_16_data/nElements_2',
                   'save_fname': 'XRF_sample4',
                   'dev': dev
                  }


params_3d_32_32_32_nElements_2 = {'P_save_path': os.path.join('./data/sample5_size_32_recon', 'Intersecting_Length_32_32_32'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(100).to(dev),
                   'src_path': os.path.join('./data/sample5_size_32_pad/nElements_2', 'grid_concentration.npy'),
                   'det_size_cm': 0.24,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.1,
                   'sample_size_n': tc.tensor(32).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(32).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample5_size_32_data/nElements_2',
                   'save_fname': 'XRF_sample5',
                   'dev': dev
                  }

params_3d_64_64_64_nElements_2_2 = {'P_save_path': './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5/Intersecting_Length_64_64_64', 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev),
                   'src_path': os.path.join('./data/sample8_size_64_pad/nElements_2', 'grid_concentration.npy'),
                   'det_size_cm': 0.9,
                   'det_from_sample_cm': 1.6,
                   'det_ds_spacing_cm': 0.4,
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),                   
                   'sample_height_n': tc.tensor(64).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_cts': tc.tensor(1.0E7).to(dev),
                   'probe_energy': np.array([20.0]),                    
                   'save_path': './data/sample8_size_64_data/nElements_2/nThetas_200_limitedSolidAngle/solidAngle_frac_0.0156/Noise',
                   'save_fname': 'XRF_sample8',
                   'Poisson_noise': True,                
                   'dev': dev
                  }


params = params_3d_64_64_64_nElements_2_2


if __name__ == "__main__":  
    XRF_data  = create_XRF_data_3d(**params)

    save_path = params["save_path"]
    with open(os.path.join(save_path, 'XRF_data_parameters.txt'), "w") as recon_paras:
        print(str(params).replace(",", ",\n"), file=recon_paras, sep=',')