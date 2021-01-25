import os
import torch as tc
from data_generation_fns_updating import intersecting_length_fl_detectorlet_3d


params_3d_64_64_64 ={'det_size_cm': 0.24,
                     'det_from_sample_cm': 1.6,
                     'det_ds_spacing_cm': 0.075,
                     'sample_size_n': tc.tensor(64),
                     'sample_size_cm': tc.tensor(0.01),
                     'sample_height_n': tc.tensor(64),
                     'P_save_path': os.path.join('./data/sample2_recon/detSpacing_0.075_dpts_12', 'Intersecting_Length_64_64_64')}
        
params = params_3d_64_64_64
longest_int_length, n_det, P = intersecting_length_fl_detectorlet_3d(**params)
print(n_det)