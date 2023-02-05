from d_protocol import *
import torch
from typing import Union, Sequence
from d_utils import *


class DReal_TensorBuffer:
    def __init__(self, rank, world_size, tensors: Union[Sequence[torch.Tensor], torch.Tensor], protocol: Protocol):
        self.rank = rank
        self.world_size = world_size
        self.tensors = tensors
        self.protocol = protocol

    def get_buffer(self):
        if isinstance(self.tensors, torch.Tensor):
            return self.tensors.detach().clone()
        else:
            return torch.cat([t.view(-1).detach().clone() for t in self.tensors])

    def communicate_copy(self):
        buffer = self.get_buffer()
        return self.protocol.update(buffer)

    def unary_op_(self, func):
        if isinstance(self.tensors, torch.Tensor):
            func(self.tensors)
        else:
            for t in self.tensors:
                func(t)

    # test it
    def binary_op_(self, tensors, func):
        if isinstance(self.tensors, torch.Tensor):
            func(self.tensors, tensors)
        else:
            for i, t in enumerate(self.tensors):
                func(t, tensors[i])

    def communicate_inplace(self):
        buffer = self.get_buffer()
        communicated_tensor = self.protocol.update(buffer)
        if isinstance(self.tensors, torch.Tensor):
            self.tensors.data.copy_(communicated_tensor)
        else:
            ptr = 0
            for t in self.tensors:
                flatten_t = t.view(-1)
                flatten_t.data.copy_(communicated_tensor[ptr : ptr + len(flatten_t)])
                ptr += len(flatten_t)

    # size [buffer_size, n]
    def gather(self):
        buffer = self.get_buffer()
        if self.rank == 0:
            tensor_list = [torch.empty_like(buffer) for _ in range(self.world_size)]
            return self.protocol.gather(buffer, tensor_list=tensor_list)
        else:
            self.protocol.gather(buffer, tensor_list=None)
            return None

    # size [buffer_size,]
    def _reduce(self, op):
        buffer = self.get_buffer()
        dist.reduce(buffer, dst=0, op=op)
        return buffer

    # size [buffer_size,]
    def global_average(self):
        self._reduce(dist.ReduceOp.SUM) / self.world_size        
        return self._reduce(dist.ReduceOp.SUM) / self.world_size