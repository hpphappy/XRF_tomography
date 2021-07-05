import os
import torch as tc
import torch.distributed as dist
from torch.multiprocessing import Process
from  time import sleep
import datetime

        
def init_processes(rank, size, fn, name, backend='gloo'):
    """ Initialize the distributed environment. """
    
    #Alternate way to provice rank 0 IP and open port
    #os.environ['MASTER_ADDR'] = '52.250.110.24'
    #os.environ['MASTER_PORT'] = '29500'
    
    
    print('{} : started process for rank : {}'.format(os.getpid(),rank))
    
    #Remove init_method if initializing through environment variable
    dist.init_process_group(backend = backend,
                            init_method='tcp://127.0.0.1:29500', #Provide rank 0 IP and open port
                            rank=rank,
                            world_size=size,
                            timeout=datetime.timedelta(0,seconds =  20))
    #dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, name)


def startprocesses(ranks, size, fn, name):
    processes = []
    for rank in ranks:
        p = Process(target=init_processes, args=(rank, size, fn, name))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print('finished')

    
def run(rank, size, name):
    """ Distributed function. """
    print('{} :Inside rank {}, total processes = {}'\
          .format (os.getpid(),rank,size))
    print(name)
    #sleep(5)
    print('{} exiting process'.format(os.getpid()))
    
   
    
if __name__ == '__main__':
    startprocesses(range(2), 2, run, "Panpan")