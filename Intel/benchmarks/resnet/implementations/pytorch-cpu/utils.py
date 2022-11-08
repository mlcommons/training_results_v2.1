import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
import torch.distributed as dist
import os
from platform_utils import *

class MLPerfLRScheduler:
    '''
    Implements LR schedule according to MLPerf Tensorflow2 reference for Resnet50
    This scheduler needs to be called before optimizer.step()    
    '''
    def __init__(self, optimizer, train_epochs, warmup_epochs, steps_per_epoch, base_lr, end_lr=0.0001, power=2.0):
        
        self.optimizer = optimizer
        self.base_lr =  base_lr
        self.end_lr = end_lr
        self.power = power
        self.train_steps = train_epochs*steps_per_epoch
        self.warmup_steps = warmup_epochs*steps_per_epoch
        self.decay_steps = self.train_steps - self.warmup_steps + 1
        self.current_lr = None 
        self.current_step = 0

    def step(self):
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            self.current_lr = self._get_warmup_rate(self.current_step)
        else: 
            self.current_lr = self._get_poly_rate(self.current_step) 

        self._update_optimizer_lr(self.current_lr) 

    def _get_warmup_rate(self, step):

        return self.base_lr*(step/self.warmup_steps) 

    def _get_poly_rate(self, step): 

        poly_step = step - self.warmup_steps
        poly_rate = (self.base_lr - self.end_lr)*(1-(poly_step/self.decay_steps))**self.power + self.end_lr
        return poly_rate 

    def _update_optimizer_lr(self, lr): 

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def barrier():
    if not is_dist_avail_and_initialized():
        return
    torch.distributed.barrier()

def get_rank():
    
    if not is_dist_avail_and_initialized():
        print("Using Environment Rank")
        return int(os.environ["RANK"])
    return dist.get_rank()

def get_affinity_list(rank, worker_count):
    '''
    Assign last 'worker_count' number of cores in a socket
    to data loader workers, assuming sequential rank placement
    and 1 process/socket.

    TODO: generalize to multiple processes per socket   
    '''  
    cpu_info = CPUInfo()
    core_count = cpu_info.cores_per_socket
    socket_count = cpu_info.sockets

    affinity_map = dict()
    for i in range(socket_count):
       start = (i+1)*core_count - 1
       end = start - worker_count
       cores=[*range(start,end,-1)]
       affinity_map[i] = cores

    return affinity_map[rank % socket_count]
