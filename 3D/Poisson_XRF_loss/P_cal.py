from mpi4py import MPI
from data_generation_fns_mpi_updating_realData import intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual
import numpy as np

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()



params_0 = {"n_ranks": n_ranks,
          "minibatch_size": 44,
          "rank": rank,
          "manual_det_coord":True, 
          "set_det_coord_cm":np.array([[0.70, -2.0, 0.70], [0.70, -2.0, -0.70], [-0.70, -2.0, 0.70], [-0.70, -2.0, -0.70]]),      
          "det_on_which_side": "negative",
          "det_size_cm": 2.4,
          "det_from_sample_cm": 3.0,
          "det_ds_spacing_cm": 2.4/2.0,
          "sample_size_n": 44,
          "sample_size_cm": 0.007,
          'sample_height_n': 20,
          "P_folder": f'data/P_array/sample_44_44_20_n/Dis_2.0_detSize_2.4_manual_dpts_4',
          'f_P': 'Intersecting_Length_44_44_20'
}


params_1 = {"n_ranks": n_ranks,
          "minibatch_size": 44,
          "rank": rank,
          "manual_det_coord":True, 
          "set_det_coord_cm":np.array([[0.8, -2.0, 0.8], [0.8, -2.0, 0.6], [0.6, -2.0, 0.6], [0.60, -2.0, 0.8],
                                       [0.8, -2.0, -0.6], [0.8, -2.0, -0.8], [0.6, -2.0, -0.8], [0.6, -2.0, -0.6],
                                       [-0.6, -2.0, -0.6], [-0.6, -2.0, -0.8], [-0.8, -2.0, -0.8], [-0.8, -2.0, -0.6],                                       
                                       [-0.6, -2.0, 0.8], [-0.6, -2.0, 0.6], [-0.8, -2.0, 0.6], [-0.8, -2.0, 0.8]]),      
          "det_on_which_side": "negative",
          "det_size_cm": 2.4,
          "det_from_sample_cm": 3.0,
          "det_ds_spacing_cm": 2.4/2.0,
          "sample_size_n": 44,
          "sample_size_cm": 0.007,
          'sample_height_n': 20,
          "P_folder": f'data/P_array/sample_44_44_20_n/Dis_2.0_detSize_2.4_manual_dpts_16',
          'f_P': 'Intersecting_Length_44_44_20'
}
    
params_2 = {"n_ranks": n_ranks,
          "minibatch_size": 44,
          "rank": rank,
          "manual_det_coord":False, 
          "set_det_coord_cm":np.array([[8.0, -3.0, 8.0], [8.0, -3.0, -8.0], [-8.0, -3.0, 8.0], [-8.0, -3.0, -8.0]]),             
          "det_on_which_side": "negative",
          "det_size_cm": 2.4,
          "det_from_sample_cm": 3.0,
          "det_ds_spacing_cm": 2.4/2.0,
          "sample_size_n": 44,
          "sample_size_cm": 0.007,
          'sample_height_n': 20,
          "P_folder": f'data/P_array/sample_44_44_20_n/Dis_3.0_detSize_2.4_detSpacing_1.2_dpts_5',
          'f_P': 'Intersecting_Length_44_44_20'
}

params_3 = {"n_ranks": n_ranks,
          "minibatch_size": 44,
          "rank": rank,
          "manual_det_coord":False, 
          "set_det_coord_cm":np.array([[8.0, -3.0, 8.0], [8.0, -3.0, -8.0], [-8.0, -3.0, 8.0], [-8.0, -3.0, -8.0]]),      
          "det_on_which_side": "negative",
          "det_size_cm": 2.4,
          "det_from_sample_cm": 2.0,
          "det_ds_spacing_cm": 2.4/2.0,
          "sample_size_n": 44,
          "sample_size_cm": 0.007,
          'sample_height_n': 20,
          "P_folder": f'data/P_array/sample_44_44_20_n/Dis_2.0_detSize_2.4_detSpacing_1.2_dpts_5',
          'f_P': 'Intersecting_Length_44_44_20'
}

params_4 = {"n_ranks": n_ranks,
          "minibatch_size": 44,
          "rank": rank,
          "manual_det_coord":False, 
          "set_det_coord_cm":np.array([[8.0, -3.0, 8.0], [8.0, -3.0, -8.0], [-8.0, -3.0, 8.0], [-8.0, -3.0, -8.0]]),      
          "det_on_which_side": "negative",
          "det_size_cm": 2.4,
          "det_from_sample_cm": 3.0,
          "det_ds_spacing_cm": 2.4/6.0,
          "sample_size_n": 44,
          "sample_size_cm": 0.007,
          'sample_height_n': 20,
          "P_folder": f'data/P_array/sample_44_44_20_n/Dis_3.0_detSize_2.4_detSpacing_0.4_dpts_29',
          'f_P': 'Intersecting_Length_44_44_20'
}

params_5 = {"n_ranks": n_ranks,
          "minibatch_size": 44,
          "rank": rank,
          "manual_det_coord":False, 
          "set_det_coord_cm":np.array([[8.0, -3.0, 8.0], [8.0, -3.0, -8.0], [-8.0, -3.0, 8.0], [-8.0, -3.0, -8.0]]),      
          "det_on_which_side": "negative",
          "det_size_cm": 2.4,
          "det_from_sample_cm": 3.0,
          "det_ds_spacing_cm": 2.4/2,
          "sample_size_n": 88,
          "sample_size_cm": 0.007,
          'sample_height_n': 40,
          "P_folder": f'data/P_array/sample_88_88_40_n/Dis_3.0_detSize_2.4_detSpacing_1.2_dpts_5',
          'f_P': 'Intersecting_Length_88_88_40'
}


intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual(**params_1)
