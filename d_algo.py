from algo import *
from collections.abc import Callable
import torch.distributed as dist


class D_Sort(Sort):
    def __init__(self, rank, node_cnt: int, sort_maker: Callable[[], Sort]) -> None:
        super().__init__()
        self.node_cnt = node_cnt
        self.rank = rank
        self.sorter = sort_maker()

    def sort(self):
        return self.sorter.sort()

    def save_after_training(self, addr):
        pass


class CD_GraB(D_Sort):
    def __init__(self, rank: int, args, n: int, m: int, d: int, microbatch: int, device):
        assert m % 2 == 0, "pair balance only supports even number"
        self.args = args
        self.rank = rank
        self.n = n
        self.m = m
        self.d = d
        self.device = device
        self.microbatch = microbatch
        self.local_balance_step = microbatch // 2

        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.next_orders = torch.vstack([torch.arange(m, device=device) for _ in range(n)])
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.m - 1
        self.args = args

    @torch.no_grad()
    def reorder_online(self, batch_idx):
        # grad at even step subtract grad at odd step
        for i, (idx_1, idx_2) in enumerate(batch_idx.view(len(batch_idx) // 2, 2)):
            for j in range(self.n):
                pair_diff = self.local_pair_diff_cache[j, i]
                if torch.inner(self.run_pair_diff_sum, pair_diff) <= 0:
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx_1]
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx_2]
                    self.run_pair_diff_sum.add_(pair_diff)
                else:
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx_1]
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx_2]
                    self.run_pair_diff_sum.sub_(pair_diff)
            self.left_ptr += 1
            self.right_ptr -= 1

    # we assume cur_grad has even number of examples.
    @torch.no_grad()
    # cur_grad: (n, microbatch, d) or (n, d)
    def step(self, cur_grad, batch_idx: int):
        if cur_grad.dim() == 3 and cur_grad.shape[1] == self.microbatch:
            self.local_pair_diff_cache = cur_grad[:,1:self.microbatch:2,:] - cur_grad[:,::2,:]
        elif cur_grad.dim() == 2:
            self.local_pair_diff_cache = cur_grad[1:self.microbatch:2,:] - cur_grad[::2,:]
        else:
            raise RuntimeError(f"wrong shape of input: {cur_grad.shape}!")

        self.reorder_online(batch_idx)
        del self.local_pair_diff_cache

    @torch.no_grad()
    def sort(self):
        self.left_ptr = 0
        self.right_ptr = self.m - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        self.run_pair_diff_sum.zero_()
        return self.orders.clone()[self.rank]


class CD_GraB_SingleGrad(D_Sort):
    def __init__(self, rank: int, args, n: int, m: int, d: int, device):
        assert m % 2 == 0, "pair balance only supports even number"
        self.args = args
        self.rank = rank
        self.n = n
        self.m = m
        self.d = d
        self.device = device

        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.next_orders = torch.vstack([torch.randperm(m, device=device) for _ in range(n)])
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.m - 1
        self.args = args

    @torch.no_grad()
    def reorder_online(self, batch_idx): # batch_idx is odd
        # grad at even step subtract grad at odd step
        for j in range(self.n):
            pair_diff = self.local_pair_diff_cache[j]
            if torch.inner(self.run_pair_diff_sum, pair_diff) <= 0:
                self.next_orders[j, self.left_ptr] = self.orders[j, batch_idx - 1]
                self.next_orders[j, self.right_ptr] = self.orders[j, batch_idx]
                self.run_pair_diff_sum.add_(pair_diff)
            else:
                self.next_orders[j, self.right_ptr] = self.orders[j, batch_idx - 1]
                self.next_orders[j, self.left_ptr] = self.orders[j, batch_idx]
                self.run_pair_diff_sum.sub_(pair_diff)
        self.left_ptr += 1
        self.right_ptr -= 1

    # we assume cur_grad has even number of examples.
    @torch.no_grad()
    # cur_grad: n, d
    def step(self, cur_grad, batch_idx: int):
        if batch_idx % 2 == 0:
            self.local_pair_diff_cache = cur_grad
        else:
            self.local_pair_diff_cache -= cur_grad
            self.reorder_online(batch_idx)
            del self.local_pair_diff_cache

    @torch.no_grad()
    def sort(self):
        self.left_ptr = 0
        self.right_ptr = self.m - 1
        self.orders = self.next_orders.clone()
        self.next_orders.zero_()
        self.run_pair_diff_sum.zero_()
        return self.orders[self.rank]


class Independent_Balance(D_Sort):
    def __init__(self, rank, n: int, m: int, d: int, device):
        def sort_maker(): return GraB(m, d, device=device)
        super().__init__(rank, n, sort_maker)

    def step(self, cur_grad: torch.Tensor, batch_idx: int):
        self.sorter.step(cur_grad, batch_idx)

    def sort(self):
        return super().sort()


class D_RR(D_Sort):
    def __init__(self, rank, n, m, device=None):
        def sort_maker(): return RandomShuffle(m, device=device)
        super().__init__(rank, n, sort_maker)
        self.num_batches = m
        self.device = device

    def step(self, *args, **kw):
        pass

    def sort(self, *args, **kw):
        return super().sort()

    def save_after_training(self, addr):
        pass



class Independent_PairBalance(D_Sort):
    def __init__(self, rank: int, m: int, n: int, d: int, device=None):
        def sort_maker(): return PairBalance_Sorter(m, d, device=device)
        super().__init__(rank, n, sort_maker)

    def step(self, optimizer, num_batch, *args, **kw):
        self.sorter.step(optimizer, num_batch)

    def sort(self, *args, **kw):
        return super().sort()


class CD_GraB_Simulated(D_Sort):
    def __init__(self, args, n: int, m: int, d: int, device):
        assert m % 2 == 0, "pair balance only supports even number"
        self.args = args
        self.n = n
        self.m = m
        self.d = d
        self.device = device

        self.run_pair_diff_sum = torch.zeros(d, device=device)
        self.next_orders = torch.vstack(
            [torch.randperm(m, device=device) for _ in range(n)])
        self.orders = self.next_orders.clone()
        self.left_ptr, self.right_ptr = 0, self.m - 1
        self.args = args

    # we assume cur_grad has even number of examples.
    @torch.no_grad()
    # cur_grad: n, microbatch, d, or n, d
    def step(self, cur_grad, idx: int):
        if idx % 2 == 0:
            self.local_pair_diff_cache = cur_grad
        else:
            self.local_pair_diff_cache -= cur_grad
            for j in range(self.n):
                pair_diff = self.local_pair_diff_cache[j]
                if torch.inner(self.run_pair_diff_sum, pair_diff) <= 0:
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx - 1]
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx]
                    self.run_pair_diff_sum.add_(pair_diff)
                else:
                    self.next_orders[j, self.right_ptr] = self.orders[j, idx - 1]
                    self.next_orders[j, self.left_ptr] = self.orders[j, idx]
                    self.run_pair_diff_sum.sub_(pair_diff)
            self.left_ptr += 1
            self.right_ptr -= 1
            del self.local_pair_diff_cache

    @torch.no_grad()
    def sort(self):
        self.left_ptr = 0
        self.right_ptr = self.m - 1
        self.orders = self.next_orders
        self.next_orders = torch.zeros_like(self.next_orders)
        self.run_pair_diff_sum.zero_()
        return self.orders.clone()
