from mpi4py import MPI
import h5py
import numpy as np

rank = MPI.COMM_WORLD.rank

f_data = h5py.File('parallel_test.hdf5', 'r')
f_data_array = f_data['test'][rank]
print("rank", rank, f_data_array)

