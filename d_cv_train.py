import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List
from algo import Sorter
from d_data import *
from d_model import D_Model
from d_utils import *
from d_eventTimer import EventTimer
from d_algo import flatten_grad
from tqdm import tqdm

def d_cv_train(d_trainset: D_VisionData, 
          optimizer: torch.optim.Optimizer, 
          model: D_Model, 
          sorter: Sorter, criterion: nn.Module, 
          epoch, counter, args, eventTimer: EventTimer, grad_acc=8):
    model.train()
    # We obtain the order from sorter before each epoch
    with eventTimer("sorter_sort"): 
        perm_list = sorter.sort()

    cur_loss = 0
    acc_step = 0
    print_rank_0(args.rank, f"Number of batches: {len(d_trainset)}")
    for batch in range(len(d_trainset)):
        # Using the obtained order, we get the training examples
        X, Y = d_trainset[perm_list[batch]]
        
        optimizer.zero_grad()
        with eventTimer("everything"):
            with eventTimer("forward_pass"):
                Y_hat = model(X)
                loss = criterion(Y_hat, Y.to(torch.int64))

            with eventTimer("backward"):
                loss.backward()

            # Depending on the sorter, there might be online sorting steps
            with eventTimer("sorter_step"):
                sorter.step(optimizer, batch)

            # Perform gradient accumulation depending on the current step
            with eventTimer("SGD_step"):
                model.grad_copy_buffer.binary_op_(list(p.grad.data for p in model.parameters()), ADD_TO_LEFT)
                acc_step += 1
                if (batch > 0 and batch % grad_acc == 0) or (batch == len(d_trainset) - 1) or (grad_acc == 1):
                    # (reached a minibatch size) or (reached the end and have remainding microbatch) or (no grad_acc)
                    model.grad_copy_buffer.unary_op_(AVERAGE_BY_(acc_step))
                    model.grad_copy_buffer.binary_op_(list(p.grad for p in model.parameters()), RIGHT_COPY_)
                    optimizer.step()
                    model.grad_copy_buffer.unary_op_(ZERO_)
                    acc_step = 0
                    with eventTimer("communication"):
                        model.communicate_weight_inplace()

        if args.rank == 0:
            counter.update(1)
        cur_loss += loss.detach()
        if batch > 0 and batch % args.log_interval == 0 and args.rank == 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches | loss {:.3f}'.format(epoch, batch, len(d_trainset), cur_loss.item() / (batch+1)))
            
    return cur_loss / len(d_trainset)


@torch.no_grad()
def d_cv_test(testloader: DataLoader, d_cv_model: D_Model, epoch: int, rank, device=None, dtype=torch.float32):
    if rank == 0:
        d_cv_model.eval()
        global_score = torch.zeros(1, device=device)

        length = 0
        for data, targets in testloader:
            data, targets = data.to(dtype=dtype, device=device), targets.to(dtype=dtype, device=device)
            Y_hat = d_cv_model(data)
            preds = torch.max(Y_hat, dim=1)[1]
            global_score += (preds == targets).sum()
            length += len(targets)

        global_score = global_score / length * 100.0

        print(f'| epoch {epoch:3d} | global avg acc {global_score.item():.4f} %|')
        return global_score 


@torch.no_grad()
def d_cv_full_train_loss(rank, trainloader: DataLoader, d_cv_model, criterion, device=None, dtype=torch.float32):
    if rank == 0:
        d_cv_model.eval() 
        counter = tqdm(range(len(trainloader)))
        length = 0
        cur_loss = torch.zeros(1, device=device)
        cur_score = torch.zeros(1, device=device)
        d_cv_model.eval()
        for data, targets in trainloader:
            counter.update(1)
            data, targets = data.to(dtype=dtype, device=device), targets.to(dtype=dtype, device=device)
            Y_hat = d_cv_model(data)
            preds = torch.max(Y_hat, dim=1)[1]
            
            cur_score += (preds == targets).sum()
            loss = criterion(Y_hat, targets.to(torch.int64))
            cur_loss += loss * len(targets)
            length += len(targets)

        cur_epoch_full_train_loss = cur_loss / length
        cur_epoch_full_train_acc = cur_score / length * 100.0

        print(f"cur epoch full train loss is:   {cur_epoch_full_train_loss.item()}")
        print(f"cur epoch full train acc is:    {cur_epoch_full_train_acc.item():.4f} %")
        return cur_epoch_full_train_loss, cur_epoch_full_train_acc