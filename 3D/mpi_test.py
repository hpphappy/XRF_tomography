import torch as tc
import numpy as np
import time
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_rank = comm.Get_size()
rank = comm.Get_rank()


comm.Barrier()
seed = int(time.time() / 60)
seed = comm.bcast(seed, root=0)
comm.Barrier()
np.random.seed(seed)

rand_idx = tc.randperm(10)
 

# print("Process ", rank, " before theta_ls = ", rand_idx)
# sys.stdout.flush()
# X = comm.gather(X, root=0)   
print("Process ", rank, " after rand_idx = ", rand_idx)
sys.stdout.flush()



