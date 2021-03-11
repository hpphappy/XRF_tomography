from mpi4py import MPI
import h5py
import numpy as np
import torch as tc
import sys

comm=MPI.COMM_WORLD
rank = comm.Get_rank()
n_ranks = comm.Get_size()


sample_height_n = tc.tensor(64)
sample_size_n = tc.tensor(64)
n_layers_each_rank = sample_height_n // n_ranks

j_offset = rank * n_layers_each_rank * sample_size_n**2
dia_len_n = int((sample_height_n**2 + sample_size_n**2 + sample_size_n**2)**0.5)

f = h5py.File('data/P_array/sample_64_64_64/detSpacing_0.4_dpts_5/Intersecting_Length_64_64_64.h5', 'r')
f_data_array = f['P_array'][:,:, rank * n_layers_each_rank * sample_size_n**2 * dia_len_n: (rank+1) * n_layers_each_rank * sample_size_n**2 * dia_len_n]


print("rank", rank, f_data_array)
sys.stdout.flush()
