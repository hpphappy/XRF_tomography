#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import h5py
import xraylib as xlib
import xraylib_np as xlib_np

from util import find_lines_roi_idx_from_dataset
from Atomic_number import AN

fl = {"K": np.array([xlib.KA1_LINE, xlib.KA2_LINE, xlib.KA3_LINE, xlib.KB1_LINE, xlib.KB2_LINE,
                 xlib.KB3_LINE, xlib.KB4_LINE, xlib.KB5_LINE]),
      "L": np.array([xlib.LA1_LINE, xlib.LA2_LINE, xlib.LB1_LINE, xlib.LB2_LINE, xlib.LB3_LINE,
                 xlib.LB4_LINE, xlib.LB5_LINE, xlib.LB6_LINE, xlib.LB7_LINE, xlib.LB9_LINE,
                 xlib.LB10_LINE, xlib.LB15_LINE, xlib.LB17_LINE]),              
      "M": np.array([xlib.MA1_LINE, xlib.MA2_LINE, xlib.MB_LINE])               
     }

def calibrate_incident_probe_intensity(std_path, f_std, fitting_method, std_element_lines_roi, density_std_elements, probe_energy):
    
    XRF_pcs_sum = np.zeros((std_element_lines_roi.shape[0]))
    for i, element_line in enumerate(std_element_lines_roi):
        XRF_pcs = np.squeeze(xlib_np.CS_FluorLine_Kissel_Cascade(np.array([AN[element_line[0]]]), fl[element_line[1]], probe_energy))
        XRF_pcs_sum[i] = np.sum(XRF_pcs)
        
    with h5py.File(os.path.join(std_path, f_std), "r") as f:
        dset_XRF = f[os.path.join("MAPS", fitting_method)][...]  

    std_element_idx = find_lines_roi_idx_from_dataset(std_path, f_std, std_element_lines_roi, std_sample=True)   
    std_XRF_count = dset_XRF[std_element_idx]
    std_XRF_count = np.reshape(std_XRF_count, (std_XRF_count.shape[0], std_XRF_count.shape[1]*std_XRF_count.shape[2]))
    I_i_ave = np.average(std_XRF_count, axis=1) 
    I_0_cal = I_i_ave/(XRF_pcs_sum*density_std_elements)
    I_0_cal_ave = np.average(I_0_cal)

    return I_0_cal_ave