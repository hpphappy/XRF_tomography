from mpi4py import MPI
import torch 
import numpy as np
import tqdm
import sys

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# if rank == 0:
#     data = [(i+1)**2 for i in range(size)]
# else:
#     data = None
    
# print("Process ", rank, " before n = ", data)   
# data = comm.scatter(data, root=0)
# print("Process ", rank, " after n = ", data)

n_loops = 10
x = np.arange(size)

    
for n in range(n_loops):
    x[rank] += 1
    

print(rank, x)  