import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

def cleanup():
    dist.destroy_process_group()

def run_train(train_fn, world_size):
    mp.spawn(train_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def train_parallel(rank, world_size):
    setup(rank, world_size)


if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    run_train(train_parallel, n_gpus)