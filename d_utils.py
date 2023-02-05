import torch
import random
import os
import numpy as np

def ADD_TO_LEFT(lhs_tensor, rhs_tensor):
    lhs_tensor.data.add_(rhs_tensor.data)

def ZERO_(tensor):
    tensor.data.zero_()

def LEFT_COPY_(lhs_tensor, rhs_tensor):
    lhs_tensor.data.copy_(rhs_tensor.data)

def RIGHT_COPY_(lhs_tensor, rhs_tensor):
    rhs_tensor.data.copy_(lhs_tensor.data)

def AVERAGE_BY_(num):
    return lambda lhs_tensor: lhs_tensor.data.copy_(lhs_tensor.data / num)


def print_rank_0(rank, *args, **kw):
    if rank == 0:
        print(*args, **kw)

def print_rank_i(rank, desired_rank, *args, **kw):
    if rank == desired_rank:
        print(*args, **kw)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
