import torch as tc
import numpy as np
import sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_rank = comm.Get_size()
rank = comm.Get_rank()


X = tc.ones((1,3,3)) #cpu


 

print("Process ", rank, " before theta_ls = ", X)
sys.stdout.flush()
X = comm.gather(X, root=0)   
print("Process ", rank, " after theta_ls = ", X, X)
sys.stdout.flush()



