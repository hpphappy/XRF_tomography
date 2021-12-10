import os
import sys
import csv
from mpi4py import MPI

comm = MPI.COMM_WORLD
n_ranks = comm.Get_size()
rank = comm.Get_rank()


def print_flush_root(this_rank, val, output_file='', root=0, output_folder='./', save_stdout=True, print_terminal=False):
    # print(or not) the argument, val, for all ranks.
    # save(or not) the argument, val, if the current rank is the root rank
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
    # print and/or save the argument, val, for all ranks.
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

def create_summary(save_path, locals_dict, verbose=False):

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    f = open(os.path.join(save_path, 'summary.txt'), 'w')

    f.write('============== PARAMETERS ==============\n')
    for var_name in locals_dict:
        try:
            line = '{:<40}{}\n'.format(var_name, str(locals_dict[var_name]))
            if verbose:
                print(line)
            f.write(line)
        except:
            pass
    f.write('========================================')
    f.close()
    return None
