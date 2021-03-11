import os
import torch as tc
from data_generation_fns_mpi_updating_hdfParray import intersecting_length_fl_detectorlet_3d


params_3d_64_64_64 ={'det_size_cm': 0.8,
                     'det_from_sample_cm': 1.6,
                     'det_ds_spacing_cm': 0.4,
                     'sample_size_n': tc.tensor(64),
                     'sample_size_cm': tc.tensor(0.01),
                     'sample_height_n': tc.tensor(64),
                     'P_folder': './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',
                     'f_P': 'Intersecting_Length_64_64_64'}
        
params = params_3d_64_64_64
longest_int_length, n_det, P = intersecting_length_fl_detectorlet_3d(**params)