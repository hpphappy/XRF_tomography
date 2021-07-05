import os
import csv

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
        
        
