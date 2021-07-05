import torch as tc
import numpy as np
import time
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_rank = comm.Get_size()
rank = comm.Get_rank()


                
if rank == 0:  
    X_cpu = tc.ones(5)

else:
    X_cpu = None
    
comm.Barrier()
X = comm.bcast(X_cpu, root=0)
comm.Barrier()
 

# print("Process ", rank, " before theta_ls = ", rand_idx)
# sys.stdout.flush()
# X = comm.gather(X, root=0)   
# print("Process ", rank, " after X = ", X)
print("Process ", rank, " before X_cpu = ", X_cpu)
print("Process ", rank, " after X_cpu = ", X_cpu)
sys.stdout.flush()



