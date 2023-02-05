import torch
import torch.distributed as dist
from typing import Optional, List


class Protocol:
    def __init__(self, rank, world_size) -> None:
        self.rank = rank
        self.world_size = world_size

    def update(self, data):
        raise NotImplementedError()

    def gather(self, tensor: torch.Tensor, tensor_list:Optional[List[torch.Tensor]]=None, dst=0):
        if self.rank == dst:
            dist.gather(tensor, gather_list=tensor_list)            
            return torch.vstack(tensor_list).T
        else:
            dist.gather(tensor, dst=dst)


class DecentralizedProtocol(Protocol):
    def __init__(self, rank, world_size, neighbor_list):
        super(DecentralizedProtocol, self).__init__(rank, world_size)
        #neighbor_list: list of indices
        self.neighbor_list = neighbor_list
        self.avg_ratio = {
            'self': 1 / (len(neighbor_list) + 1),
            'neighbor': 1 / (len(neighbor_list) + 1),
        }

    def update(self, data):
        comm_buffer = {i: torch.empty_like(data) for i in self.neighbor_list}
        comm_buffer[self.rank] = data

        reqs = []
        for neighbor_index in self.neighbor_list:
            reqs.append(dist.isend(tensor=comm_buffer[self.rank], dst=neighbor_index))
            reqs.append(dist.irecv(tensor=comm_buffer[neighbor_index], src=neighbor_index))
        for req in reqs:
            req.wait()
        ret = torch.zeros_like(data)
        for k in comm_buffer.keys():
            if k == self.rank:
                ret += comm_buffer[k] * self.avg_ratio['self']
            else:
                ret += comm_buffer[k] * self.avg_ratio['neighbor']
        return ret


class CentralizedProtocol(Protocol):
    def __init__(self, rank, world_size):
        super(CentralizedProtocol, self).__init__(rank, world_size)

    def update(self, data):
        dist.all_reduce(data)
        return data / self.world_size
