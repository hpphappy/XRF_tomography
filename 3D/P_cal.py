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
          "manual_det_area": True,
          "det_size_cm": None,
          "det_from_sample_cm": 2.0,
          "det_ds_spacing_cm": None,
          "sample_size_n": 44,
          "sample_size_cm": 0.007,
          'sample_height_n': 16,
          "P_folder": f'data/P_array/sample_44_44_16_n/Dis_2.0_manual_dpts_4',
          'f_P': 'Intersecting_Length_44_44_16'
}


params_1 = {"n_ranks": n_ranks,
          "minibatch_size": 44,
          "rank": rank,
          "manual_det_coord":True, 
          "set_det_coord_cm":np.array([[0.8, -2.0, 0.8], [0.8, -2.0, 0.6], [0.6, -2.0, 0.6], [0.60, -2.0, 0.8],
                                       [0.8, -2.0, -0.6], [0.8, -2.0, -0.8], [0.6, -2.0, -0.8], [0.6, -2.0, -0.6],
                                       [-0.6, -2.0, -0.6], [-0.6, -2.0, -0.8], [-0.8, -2.0, -0.8], [-0.8, -2.0, -0.6],                                       
                                       [-0.6, -2.0, 0.8], [-0.6, -2.0, 0.6], [-0.8, -2.0, 0.6], [-0.8, -2.0, 0.8]]),      vim 
          "det_on_which_side": "negative",
          "manual_det_area": True,
          "det_size_cm": None,
          "det_from_sample_cm":None,
          "det_ds_spacing_cm": None,
          "sample_size_n": 44,
          "sample_size_cm": 0.007,
          'sample_height_n': 20,
          "P_folder": f'data/P_array/sample_44_44_20_n/Dis_2.0_detSize_2.4_manual_dpts_16',
          'f_P': 'Intersecting_Length_44_44_20'
}
    

    
params_32 = {"n_ranks": n_ranks,
          "minibatch_size": 32,
          "rank": rank,
          "manual_det_coord":False, 
          "set_det_coord_cm":None,      
          "det_on_which_side": "positive",
          "manual_det_area":False,
          "det_size_cm": 0.9,
          "det_from_sample_cm": 1.6,
          "det_ds_spacing_cm": 0.4,
          "sample_size_n": 32,
          "sample_size_cm": 0.01,
          'sample_height_n': 32,
          "P_folder": './data/P_array/sample_32_32_32_detSpacing_0.4_dpts_5',
          'f_P': 'Intersecting_Length_32_32_32'
}     
    
params_64 = {"n_ranks": n_ranks,
          "minibatch_size": 64,
          "rank": rank,
          "manual_det_coord":False, 
          "set_det_coord_cm":None,      
          "det_on_which_side": "positive",
          "manual_det_area":False,
          "det_size_cm": 0.9,
          "det_from_sample_cm": 1.6,
          "det_ds_spacing_cm": 0.4,
          "sample_size_n": 64,
          "sample_size_cm": 0.01,
          'sample_height_n': 64,
          "P_folder": './data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5',
          'f_P': 'Intersecting_Length_64_64_64'
}

params_128 = {"n_ranks": n_ranks,
          "minibatch_size": 128,
          "rank": rank,
          "manual_det_coord":False, 
          "set_det_coord_cm":None, 
          "det_on_which_side": "positive",
          "manual_det_area":False,
          "det_size_cm": 0.9,
          "det_from_sample_cm": 1.6,
          "det_ds_spacing_cm": 0.4,
          "sample_size_n": 128,
          "sample_size_cm": 0.01,
          'sample_height_n': 128,
          "P_folder": './data/P_array/sample_128_128_128/detSpacing_0.4_dpts_5',
          'f_P': 'Intersecting_Length_128_128_128'
}

params_256 = {"n_ranks": n_ranks,
          "minibatch_size": 128,
          "rank": rank,
          "manual_det_coord":False, 
          "set_det_coord_cm":None, 
          "det_on_which_side": "positive",
          "manual_det_area":False,
          "det_size_cm": 0.9,
          "det_from_sample_cm": 1.6,
          "det_ds_spacing_cm": 0.4,
          "sample_size_n": 256,
          "sample_size_cm": 0.01,
          'sample_height_n': 256,
          "P_folder": './data/P_array/sample_256_256_256/detSpacing_0.4_dpts_5',
          'f_P': 'Intersecting_Length_256_256_256'
}



intersecting_length_fl_detectorlet_3d_mpi_write_h5_3_manual(**params_32)
