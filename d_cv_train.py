import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
from algo import Sort
from d_data import *
from d_model import DReal_Model
from d_utils import *
from d_eventTimer import EventTimer
from d_algo import *
from tqdm import tqdm
import torch.nn.functional as F
import torchopt


def d_cv_train_functorch(d_trainset_X,
                       d_trainset_y,
                       func_per_example_grad,
                       fmodel,
                       params,
                       buffers,
                       optimizer,
                       opt_state,
                       sorter,
                       counter,
                       epoch,
                       n,
                       B,
                       update_B,
                       d,
                       device=None):
    microbatch = update_B // B
    if isinstance(sorter, CReal_PairBalance_Simulated):
        perm_list = sorter.sort()
    else:
        perm_list = torch.vstack([s.sort() for s in sorter])
    this_epoch_avg_grad = torch.zeros(d, device=device)
    for batch_idx in range(0, d_trainset_X.shape[1], microbatch):
        batch = torch.arange(batch_idx, min(
            batch_idx + microbatch, d_trainset_X.shape[1])).cuda()
        # Using the obtained order, we get the training examples
        X = torch.vstack([d_trainset_X[node_i, idx_x]
                         for node_i, idx_x in enumerate(perm_list[:, batch])])
        Y = torch.cat([d_trainset_y[node_i, idx_x]
                      for node_i, idx_x in enumerate(perm_list[:, batch])])
        # X = d_trainset_X[torch.arange(n, device=device), perm_list[:, batch]]
        # Y = d_trainset_y[torch.arange(n, device=device), perm_list[:, batch]]

        if type(sorter) == list and (isinstance(sorter[0], RandomShuffle)):
            avg_grad = torch.autograd.grad(F.cross_entropy(
                fmodel(params, buffers, X), Y), params)

        elif isinstance(sorter, CReal_PairBalance_Simulated):
            ft_per_sample_grads = func_per_example_grad(params, buffers, X, Y)
            with torch.no_grad():
                ft_per_sample_grads = torch.hstack(
                    [g.view(g.shape[0], g.numel() // g.shape[0]) for g in ft_per_sample_grads])
                index = torch.arange(len(X)).reshape(n, len(X) // n)
                for i, idx in enumerate(batch):
                    sorter.step(ft_per_sample_grads[index[:, i]], idx)
                avg_grad = ft_per_sample_grads.mean(dim=0)

        elif type(sorter) == list and (isinstance(sorter[0], PairBalance_Single) or isinstance(sorter[0], GraB_Single)):
            ft_per_sample_grads = func_per_example_grad(params, buffers, X, Y)
            with torch.no_grad():
                ft_per_sample_grads = torch.hstack(
                    [g.view(g.shape[0], g.numel() // g.shape[0]) for g in ft_per_sample_grads])
                for i in range(n):
                    for j, x in enumerate(batch):
                        sorter[i].step(
                            ft_per_sample_grads[i * len(batch) + j], x)
                avg_grad = ft_per_sample_grads.mean(dim=0)

        else:
            raise NotImplementedError()

        with torch.no_grad():
            if isinstance(avg_grad, torch.Tensor):
                avg_grad_list = []
                grad_cnt = 0
                this_epoch_avg_grad += avg_grad * len(batch)
                for p in params:
                    avg_grad_list.append(
                        avg_grad[grad_cnt: p.numel() + grad_cnt].view(p.shape))
                    grad_cnt += p.numel()
            else:
                this_epoch_avg_grad += torch.cat([g.view(-1)
                                                 for g in avg_grad]) * len(batch)
                avg_grad_list = avg_grad
            updates, opt_state = optimizer.update(
                avg_grad_list, opt_state, params=params)
            torchopt.apply_updates(params, tuple(updates), inplace=True)

            counter.update(len(batch))
    return this_epoch_avg_grad / d_trainset_X.shape[1]


@torch.no_grad()
def d_cv_test(testset_X, testset_Y, model, params, device=None):
    model.eval()
    acc = 0
    loss = 0
    for i, p in enumerate(model.parameters()):
        p.data.copy_(params[i])

    for i in DataLoader(torch.arange(len(testset_X)), batch_size=1024):
        data, targets = testset_X[i], testset_Y[i]
        outputs = model(data)
        loss += len(outputs) * F.cross_entropy(outputs, targets)
        preds = outputs.argmax(dim=1)
        acc += (preds == targets).sum()

    acc = acc / len(testset_X) * 100.0
    loss /= len(testset_X)
    return acc, loss


@torch.no_grad()
def parallel_herding_bound(
        d_trainset_X,
        d_trainset_y,
        func_per_example_grad,
        fmodel,
        params,
        buffers,
        avg_grad,
        perm_list):
    cum_sum = torch.zeros_like(avg_grad)
    bound = 0
    B = perm_list.shape[0]
    stochastic_grad_error = []
    for batch_idx in range(0, d_trainset_X.shape[1], 16):
        batch = torch.arange(batch_idx, min(
            batch_idx + 16, d_trainset_X.shape[1])).cuda()

        X = torch.vstack([d_trainset_X[node_i, idx_x]
                         for node_i, idx_x in enumerate(perm_list[:, batch])])
        Y = torch.cat([d_trainset_y[node_i, idx_x]
                      for node_i, idx_x in enumerate(perm_list[:, batch])])

        per_sample_grads = func_per_example_grad(params, buffers, X, Y)
        per_sample_grads = torch.hstack(
            [g.view(g.shape[0], g.numel() // g.shape[0]) for g in per_sample_grads])
        per_sample_grads -= avg_grad
        stochastic_grad_error.append(
            torch.linalg.norm(per_sample_grads, ord=2, dim=-1))
        per_sample_grads = per_sample_grads.view(
            B, len(batch), per_sample_grads.shape[-1])

        for within_minibatch_idx in range(len(batch)):
            for z_idx in range(perm_list.shape[0]):
                cum_sum += per_sample_grads[z_idx, within_minibatch_idx]
                bound = max(bound, torch.linalg.norm(cum_sum, float('inf')))
        del per_sample_grads
    return bound, torch.cat(stochastic_grad_error)


@torch.no_grad()
def empirical_parallel_herding_bound(d_trainset_X,
                                     d_trainset_y,
                                     func_per_example_grad,
                                     fmodel,
                                     params,
                                     buffers,
                                     optimizer,
                                     opt_state,
                                     counter,
                                     epoch,
                                     n,
                                     B,
                                     update_B,
                                     d,
                                     perm_list,
                                     this_epoch_avg_grad,
                                     device=None):
    microbatch = update_B // B
    cum_sum = torch.zeros_like(this_epoch_avg_grad)
    bound = 0
    B = perm_list.shape[0]
    stochastic_grad_error = []
    sanity_check = torch.zeros_like(this_epoch_avg_grad)
    for batch_idx in range(0, d_trainset_X.shape[1], microbatch):
        batch = torch.arange(batch_idx, min(
            batch_idx + microbatch, d_trainset_X.shape[1])).cuda()
        # Using the obtained order, we get the training examples
        X = torch.vstack([d_trainset_X[node_i, idx_x]
                         for node_i, idx_x in enumerate(perm_list[:, batch])])
        Y = torch.cat([d_trainset_y[node_i, idx_x]
                      for node_i, idx_x in enumerate(perm_list[:, batch])])

        per_sample_grads = func_per_example_grad(params, buffers, X, Y)
        with torch.no_grad():
            per_sample_grads = torch.hstack(
                [g.view(g.shape[0], g.numel() // g.shape[0]) for g in per_sample_grads])
            avg_grad = per_sample_grads.mean(dim=0)

            per_sample_grad_errors = per_sample_grads - this_epoch_avg_grad
            del per_sample_grads
            stochastic_grad_error.append(torch.linalg.norm(
                per_sample_grad_errors, ord=2, dim=-1))

            per_sample_grad_errors = per_sample_grad_errors.view(
                B, len(batch), per_sample_grad_errors.shape[-1])

            for within_minibatch_idx in range(len(batch)):
                for z_idx in range(perm_list.shape[0]):
                    cum_sum += per_sample_grad_errors[z_idx,
                                                      within_minibatch_idx]
                    bound = max(bound, torch.linalg.norm(
                        cum_sum, float('inf')))

            sanity_check += len(batch) * avg_grad
            avg_grad_list = []
            grad_cnt = 0
            for p in params:
                avg_grad_list.append(
                    avg_grad[grad_cnt: p.numel() + grad_cnt].view(p.shape))
                grad_cnt += p.numel()

            updates, opt_state = optimizer.update(
                avg_grad_list, opt_state, params=params)
            torchopt.apply_updates(params, tuple(updates), inplace=True)
            counter.update(len(batch))

    sanity_check /= d_trainset_X.shape[1]
    return bound, torch.cat(stochastic_grad_error)

