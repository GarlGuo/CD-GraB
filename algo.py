import torch
import random
from utils import flatten_grad


class Sorter:
    def sort(self):
        raise NotImplementedError()


class GraB(Sorter):
    def __init__(self, n, d, device=None):
        self.n = n
        self.d = d
        self.avg_grad = torch.zeros(d, device=device)
        self.cur_sum = torch.zeros_like(self.avg_grad)
        self.next_epoch_avg_grad = torch.zeros_like(self.avg_grad)
        self.orders = torch.arange(self.n, device=device, dtype=torch.int64)
        self.next_orders = torch.arange(self.n, device=device, dtype=torch.int64)
        self.left_ptr = 0
        self.right_ptr = self.n - 1

    def sort(self):
        self.avg_grad.copy_(self.next_epoch_avg_grad)
        self.next_epoch_avg_grad.zero_()
        self.cur_sum.zero_()
        self.left_ptr = 0
        self.right_ptr = self.n - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        return self.orders.clone()

    def step(self, optimizer, batch_idx):
        cur_grad = flatten_grad(optimizer)
        self.next_epoch_avg_grad.add_(cur_grad / self.n)
        cur_grad.add_(-1 * self.avg_grad)
        if torch.norm(self.cur_sum + cur_grad, p=2) <= torch.norm(self.cur_sum - cur_grad, p=2):
            self.next_orders[self.left_ptr] = self.orders[batch_idx]
            self.left_ptr += 1
            self.cur_sum.add_(cur_grad)
        else:
            self.next_orders[self.right_ptr] = self.orders[batch_idx]
            self.right_ptr -= 1
            self.cur_sum.add_(-1 * cur_grad)


class RandomShuffling(Sorter):
    def __init__(self, n) -> None:
        super().__init__()
        self.n = n
    
    def step(self, *args, **kw):
        pass

    def sort(self, *args, **kw): 
        return [i for i in torch.randperm(self.n)]


class PairBalance_Sorter(Sorter):
    def __init__(self, n:int, d:int, device=None):
        assert n % 2 == 0, "pair balance only supports even number"
        self.pair_diff = torch.zeros(d, device=device)
        self.n = n 
        self.d = d
        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.pair_cache = torch.zeros(d, device=device)
        self.next_orders = torch.arange(n, dtype=torch.int64)
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.n - 1

    def reorder_online(self, grad_vecs, i):
        # grad at even step subtract grad at odd step
        # equivalent to vecs[i] - vecs[i + 1] 
        self.pair_cache -= grad_vecs
        plus_res, minus_res = self.run_pair_diff_sum + self.pair_cache, self.run_pair_diff_sum - self.pair_cache
        if torch.norm(plus_res, p=2) <= torch.norm(minus_res, p=2):
            self.next_orders[self.left_ptr]  = self.orders[i - 1]
            self.next_orders[self.right_ptr] = self.orders[i]
            self.run_pair_diff_sum = plus_res
        else:
            self.next_orders[self.right_ptr] = self.orders[i - 1]
            self.next_orders[self.left_ptr]  = self.orders[i]        
            self.run_pair_diff_sum = minus_res
    
        self.left_ptr += 1
        self.right_ptr -= 1
        self.pair_cache.zero_()

    def store_grad(self, grad_vecs):
        self.pair_cache += grad_vecs

    def step(self, optimizer, i: int):
        # d, n
        grad_vecs = flatten_grad(optimizer)
        if i % 2 == 0:
            # store gradients to use in next step
            self.store_grad(grad_vecs)
        else:
            # perform pair balance reorder online
            self.reorder_online(grad_vecs, i)

    def sort(self):
        self.pair_diff = 0
        self.left_ptr  = 0
        self.right_ptr = self.n - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        return self.orders.clone()
