#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch as tc
from data_generation_fns import rotate, MakeFLlinesDictionary, trace_beam_z, trace_beam_x, trace_beam_y, intersecting_length_fl_detectorlet_3d

import warnings
warnings.filterwarnings("ignore")


## For a 64 x 64 x 64 sample: sample1 ##
######################################################################
# experiemtal parameters #
theta_st = tc.tensor(0).to(dev)
theta_end = tc.tensor(2 * np.pi).to(dev)
n_theta =  tc.tensor(200).to(dev)
theta_ls = - tc.linspace(theta_st, theta_end, n_theta+1)[:-1].to(dev)
sample_size_n = tc.tensor(64).to(dev)
sample_height_n = tc.tensor(64).to(dev)
sample_size_cm = tc.tensor(0.01).to(dev)
this_aN_dic = {"C": 6, "O": 8, "Si": 14, "Ca": 20, "Fe": 26}
probe_energy = np.array([20.0])
probe_cts = tc.tensor(1.0E7).to(dev)
det_size_cm = 0.24
det_from_sample_cm = 1.6
det_ds_spacing_cm = 0.1

# path of true grid concentration of the sample #
grid_path = './data/sample1_pad'
f_grid = 'grid_concentration.npy'

# XRF and XRT data path #
data_path = './data/sample1_data'
f_XRF_data = 'XRF_sample1'
f_XRT_data = 'XRT_sample1'

# path of storing the intersecting information and the reconstructing results #
recon_path = 'data/sample1_recon'
if not os.path.exists(recon_path):
    os.mkdir(recon_path)
P_save_path = os.path.join(recon_path, 'Intersecting_Length_64_64_64')
f_recon_parameters = 'recon_parameters.txt'
f_recon_grid = 'grid_concentration'
f_initial_guess = 'initialized_grid_concentration'
######################################################################

# xraylib uses keV
fl_K = np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE])

fl_L = np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE])

fl_M = np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])

fl_line_groups = np.array(["K", "L", "M"])
group_lines = True


XRT_data = np.load(os.path.join(data_path, f_XRT_data + '.npy')).astype(np.float32)
XRT_data = tc.from_numpy(XRT_data)


