#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 23:49:56 2020

@author: panpanhuang
"""

import os
import numpy as np
import torch as tc
from data_generation_fns_updating import create_XRT_data_3d
import warnings
warnings.filterwarnings("ignore")

dev = "cpu"


params_3d_5_5_5 = {'src_path': os.path.join('./data/sample3_pad', 'grid_concentration.npy'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(16).to(dev), 
                   'sample_height_n': tc.tensor(5).to(dev),
                   'sample_size_n': tc.tensor(5).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample3_data',
                   'save_fname': 'XRT_sample3',
                   'theta_sep': True,
                   'dev': dev
                  }

params_3d_64_64_64 = {'src_path': os.path.join('./data/sample1_pad', 'grid_concentration.npy'), 
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev), 
                   'sample_height_n': tc.tensor(64).to(dev),
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample1_data',
                   'save_fname': 'XRT_sample1',
                   'theta_sep': True,
                   'dev': dev
                  }

params_3d_64_64_64_nElements_2 = {'src_path': os.path.join('./data/sample2_pad/nElements_2', 'grid_concentration.npy'),
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev), 
                   'sample_height_n': tc.tensor(64).to(dev),
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample2_data/nElements_2',
                   'save_fname': 'XRT_sample2',
                   'theta_sep': True,
                   'dev': dev
                  }

params_3d_64_64_64_nElements_1 = {'src_path': os.path.join('./data/sample2_pad/nElements_1', 'grid_concentration.npy'),
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev), 
                   'sample_height_n': tc.tensor(64).to(dev),
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"Ca": 20},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample2_data/nElements_1',
                   'save_fname': 'XRT_sample2',
                   'theta_sep': True,
                   'dev': dev
                  }


params_3d_64_64_64_nElements_4 = {'src_path': os.path.join('./data/sample2_pad/nElements_4', 'grid_concentration.npy'),
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev), 
                   'sample_height_n': tc.tensor(64).to(dev),
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21, "Ti": 22, "V":23},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample2_data/nElements_4',
                   'save_fname': 'XRT_sample2',
                   'theta_sep': True,
                   'dev': dev
                  }

params_3d_64_64_64_nElements_8 = {'src_path': os.path.join('./data/sample2_pad/nElements_8', 'grid_concentration.npy'),
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev), 
                   'sample_height_n': tc.tensor(64).to(dev),
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21, "Ti": 22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample2_data/nElements_8',
                   'save_fname': 'XRT_sample2',
                   'theta_sep': True,
                   'dev': dev
                  }

params_3d_64_64_64_nElements_12 = {'src_path': os.path.join('./data/sample2_pad/nElements_12', 'grid_concentration.npy'),
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev), 
                   'sample_height_n': tc.tensor(64).to(dev),
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21, "Ti": 22, "V":23, "Cr":24, "Mn":25, "Fe":26, "Co":27,
                                   "Ni": 28, "Cu":29, "Zn":30, "Mo":42},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample2_data/nElements_12',
                   'save_fname': 'XRT_sample2',
                   'theta_sep': True,
                   'dev': dev
                  }


params_3d_8_8_8_nElements_2 = {'src_path': os.path.join('./data/sample6_size_8_pad/nElements_2', 'grid_concentration.npy'),
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(25).to(dev), 
                   'sample_height_n': tc.tensor(8).to(dev),
                   'sample_size_n': tc.tensor(8).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample6_size_8_data/nElements_2',
                   'save_fname': 'XRT_sample6',
                   'theta_sep': True,
                   'dev': dev
                  }

params_3d_16_16_16_nElements_2 = {'src_path': os.path.join('./data/sample4_size_16_pad/nElements_2', 'grid_concentration.npy'),
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(50).to(dev), 
                   'sample_height_n': tc.tensor(16).to(dev),
                   'sample_size_n': tc.tensor(16).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/sample4_size_16_data/nElements_2',
                   'save_fname': 'XRT_sample4',
                   'theta_sep': True,
                   'dev': dev
                  }

params_3d_32_32_32_nElements_2 = {'src_path': os.path.join('./data/Xtal1_fake_size_32_pad', 'grid_concentration.npy'),
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(100).to(dev), 
                   'sample_height_n': tc.tensor(32).to(dev),
                   'sample_size_n': tc.tensor(32).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                   'save_path': './data/Xtal1_fake_size_32_data/nElements_2',
                   'save_fname': 'XRT_Xtal1_fake',
                   'theta_sep': True,
                   'dev': dev
                  }

params_3d_64_64_64_nElements_2_2 = {'src_path': os.path.join('./data/sample8_size_64_pad/nElements_2', 'grid_concentration.npy'),
                   'theta_st': tc.tensor(0).to(dev), 
                   'theta_end': tc.tensor(2 * np.pi).to(dev),
                   'n_theta': tc.tensor(200).to(dev), 
                   'sample_height_n': tc.tensor(64).to(dev),
                   'sample_size_n': tc.tensor(64).to(dev),
                   'sample_size_cm': tc.tensor(0.01).to(dev),
                   'this_aN_dic': {"Ca": 20, "Sc": 21},
                   'probe_energy': np.array([20.0]),
                   'probe_cts': tc.tensor(1.0E7).to(dev), 
                    'save_path': './data/sample8_size_64_data/nElements_2/nThetas_200_limitedSolidAngle/solidAngle_frac_0.0156/Noise',
                   'save_fname': 'XRT_sample8',
                   'theta_sep': True,
                   'Poisson_noise': True,             
                   'dev': dev
                  }

    
params = params_3d_64_64_64_nElements_2_2


if __name__ == "__main__":  
    XRT_data  = create_XRT_data_3d(**params)

    save_path = params["save_path"]
    with open(os.path.join(save_path, 'XRT_data_parameters.txt'), "w") as recon_paras:
        print(str(params).replace(",", ",\n"), file=recon_paras, sep=',')
