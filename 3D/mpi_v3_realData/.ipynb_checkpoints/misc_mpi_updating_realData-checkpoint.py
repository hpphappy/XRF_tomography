import os
import sys
import csv
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()

def print_flush(val, output_file='', output_folder='./', save_stdout=True, print_terminal=False):
    # I want the file name to be the name of the quantity, the val is the value of the quantity
    if print_terminal:
        print(val)
        
    if save_stdout:   
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)   

        if not output_file:
            output_file = "stdo.csv"

        file_path = os.path.join(output_folder, output_file)
        with open(file_path, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([val])
     
    return None

def print_flush_root(this_rank, val, output_file='', root=0, output_folder='./', save_stdout=True, print_terminal=False):
    # I want the file name to be the name of the quantity, the val is the value of the quantity
    if print_terminal:
        print("rank:", this_rank, val)
        
    if save_stdout and this_rank == root:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)   

        if not output_file:
            output_file = "stdo.csv"

        file_path = os.path.join(output_folder, output_file)
        with open(file_path, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([this_rank ,val])
            
    sys.stdout.flush() 
    return None
        
def print_flush_all(this_rank, val, output_file='', output_folder='./', save_stdout=True, print_terminal=False):
    # I want the file name to be the name of the quantity, the val is the value of the quantity
    if print_terminal:
        print("rank:", this_rank, val)
        
    if save_stdout:
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)   

        if not output_file:
            output_file = f"stdo_{this_rank}.csv"

        file_path = os.path.join(output_folder, output_file)
        with open(file_path, 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([this_rank ,val])
            
    sys.stdout.flush() 
    return None        
