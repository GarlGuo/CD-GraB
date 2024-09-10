import numpy as np
import torch
import torch.nn as nn
from d_model import *
from d_eventTimer import EventTimer
from d_algo import *
import torchopt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist


def round(tensor: torch.Tensor):
    return torch.as_tensor(np.round(tensor.cpu().numpy(), 3).item())


def sMAPE(
    # models' predictions with shape of (num_samples, forecasting horizons, ....)
    test_preds,
    test_tgts  # ground truth that has the same shape as test_preds
):
    """
    Metric for M4 dataset
    Refer to https://github.com/Mcompetitions/M4-methods/blob/master/ML_benchmarks.py
    """
    h_sMAPE = torch.abs(test_preds - test_tgts) * 200 / \
        (torch.abs(test_preds) + torch.abs(test_tgts))
    fh = test_preds.shape[1]
    # return short, medium, long forecasting horizon and total sMAPE
    return round(torch.mean(h_sMAPE[:, :fh//3])), round(torch.mean(h_sMAPE[:, fh//3:fh//3*2])), \
        round(torch.mean(h_sMAPE[:, -fh//3:])), round(torch.mean(h_sMAPE))


def wRMSE_cryptos(
    # models' predictions with shape of (number of stocks, number of samples for each stock, forecasting horizons, 8 features)
    test_preds,
    test_tgts,  # ground truth that has the same shape as test_preds
):
    """
    Metric for Cryptos return predictions
    RMSE should be weighted by the importance of each stock
    Refer to https://www.kaggle.com/competitions/g-research-crypto-forecasting/data?select=asset_details.csv
    """
    # Importance weights for 14 Cryptos
    weights = np.array([4.30406509320417,
                        6.779921907472252,
                        2.3978952727983707,
                        4.406719247264253,
                        3.555348061489413,
                        1.3862943611198906,
                        5.8944028342648505,
                        2.079441541679836,
                        1.0986122886681098,
                        2.3978952727983707,
                        1.0986122886681098,
                        1.6094379124341005,
                        2.079441541679836,
                        1.791759469228055])
    weights = weights.reshape(14, 1, 1, 1)

    fh = test_preds.shape[2]
    wrmse = ((test_preds - test_tgts)*weights)**2

    # only evaluate predictions based on the last feature (15-min ahead residulized returns)
    # return short, medium, long forecasting horizon and total Weighted RMSE
    return np.sqrt(np.mean(wrmse[..., :fh//3, -1])), np.sqrt(np.mean(wrmse[..., fh//3:fh//3*2, -1])), np.sqrt(np.mean(wrmse[..., -fh//3, -1])), np.sqrt(np.mean(wrmse[..., -1]))


def RMSE(
    # models' predictions with shape of (number of trajectories, number of samples for traj, forecasting horizons, 2 velocity components)
    test_preds,
    test_tgts  # ground truth that has the same shape as test_preds
):
    """
    Regular RMSE metric for basketball player trajectory predictions
    """
    fh = test_preds.shape[2]
    mse = np.mean((test_preds - test_tgts)**2, axis=0)
    # return short, medium, long forecasting horizon and total RMSE
    return np.sqrt(np.mean(mse[:, :fh//3])),  np.sqrt(np.mean(mse[:, fh//3:fh//3*2])),  np.sqrt(np.mean(mse[:, fh//3*2:])), np.sqrt(np.mean(mse))


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


@torch.no_grad()
def d_time_series_eval_epoch(loader: DataLoader, model, params, device=None):
    eval_loss = []
    all_preds = []
    all_trues = []
    for i, p in enumerate(model.parameters()):
        p.data.copy_(params[i])

    for inps, tgts in loader:
        if len(inps.shape) > 2:
            inps = inps.to(device)
            tgts = tgts.to(device)
        else:
            inps = inps.unsqueeze(-1).to(device)
            tgts = tgts.unsqueeze(-1).to(device)
        denorm_outs, norm_outs, norm_tgts = model(inps, tgts)
        loss = F.mse_loss(norm_outs[:, :norm_tgts.shape[1]], norm_tgts)
        eval_loss.append(loss)
        all_preds.append(denorm_outs[:, :norm_tgts.shape[1]])
        all_trues.append(tgts)
    return torch.sqrt(torch.tensor(eval_loss).mean()), torch.vstack(all_preds), torch.vstack(all_trues)



def d_time_series_train_epoch_single_grad(
                                  cur_rank,
                                  d_trainset,
                                  fmodel,
                                  params,
                                  buffers,
                                  optimizer,
                                  opt_state,
                                  sorter,
                                  counter,
                                  eventTimer: EventTimer,
                                  epoch,
                                  node_cnt,
                                  d,
                                  device=None):
    with eventTimer(f'epoch-{epoch}'):
        with eventTimer('sorter'):
            perm_list = sorter.sort()

    if isinstance(sorter, CD_GraB_SingleGrad):
        with eventTimer(f'epoch-{epoch}'):
            with eventTimer("communication"):
                gathered_grads = torch.empty(node_cnt, d, device=device)

    for batch in range(0, len(d_trainset), 1):
        # Using the obtained order, we get the training examples
        inps, tgts = d_trainset[perm_list[batch]]

        if isinstance(sorter, D_RR):
            with eventTimer(f'epoch-{epoch}'):
                with eventTimer("forward-backward"):
                    denorm_outs, norm_outs, norm_tgts = fmodel(params, buffers, inps, tgts)
                    avg_grads = torch.autograd.grad(F.mse_loss(norm_outs[:, :norm_tgts.shape[1]], norm_tgts), params)
                    with torch.no_grad():
                        avg_grads = torch.cat([g.view(-1) for g in avg_grads])
                with torch.no_grad():
                    with eventTimer("communication"):
                        dist.all_reduce(avg_grads, op=dist.ReduceOp.SUM)
                        avg_grads /= node_cnt

        elif isinstance(sorter, CD_GraB_SingleGrad):
            with eventTimer(f'epoch-{epoch}'):
                with eventTimer("forward-backward"):
                    denorm_outs, norm_outs, norm_tgts = fmodel(params, buffers, inps, tgts)
                    avg_grads = torch.autograd.grad(F.mse_loss(norm_outs[:, :norm_tgts.shape[1]], norm_tgts), params)
                    with torch.no_grad():
                        avg_grads = torch.cat([g.view(-1) for g in avg_grads])

            with torch.no_grad():
                with eventTimer(f'epoch-{epoch}'):
                    with eventTimer("communication"):
                        dist.all_gather_into_tensor(gathered_grads, avg_grads.unsqueeze(0), 
                                                    async_op=False)
                        avg_grads = gathered_grads.mean(dim=0)

                    with eventTimer("sorter"):
                        sorter.step(gathered_grads, batch)

        else:
            raise NotImplementedError()

        with torch.no_grad():
            with eventTimer(f'epoch-{epoch}'):
                with eventTimer("SGD"):
                    # compute gradient and do SGD step
                    avg_grad_list = []
                    grad_cnt = 0
                    for p in params:
                        avg_grad_list.append(
                            avg_grads[grad_cnt: p.numel() + grad_cnt].view(p.shape))
                        grad_cnt += p.numel()

                    updates, opt_state = optimizer.update(
                        avg_grad_list, opt_state, params=params)
                    torchopt.apply_updates(
                        params, tuple(updates), inplace=True)
            if cur_rank == 0:
                counter.update(1)
