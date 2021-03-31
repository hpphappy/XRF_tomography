from mpi4py import MPI
from data_generation_fns_mpi_updating_realData import intersecting_length_fl_detectorlet_3d_mpi_write_h5_2


comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

    
params = {"n_ranks": n_ranks,
          "minibatch_size": 22,
          "rank": rank,
          "det_size_cm": 2.4,
          "det_from_sample_cm": 3.0,
          "det_ds_spacing_cm": 2.4/2.0,
          "sample_size_n": 44,
          "sample_size_cm": 0.007,
          'sample_height_n': 20,
          "P_folder": f'data/P_array/sample_88_88_40/detSize_2.4_detSpacing_1.2_dpts_5',
          'f_P': 'Intersecting_Length_44_44_20'
}


intersecting_length_fl_detectorlet_3d_mpi_write_h5_2(**params)
