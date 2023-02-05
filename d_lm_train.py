import torch
import torch.nn as nn
from typing import List
from d_lm_data import *
from d_algo import D_Sorter
from d_lstm_model import *
from torch.optim import Optimizer
from d_eventTimer import EventTimer


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def d_LM_train(c_data: D_LM_Dataset,
                optimizer: Optimizer,
                lm_model: D_Model,
                sorter: D_Sorter, 
                epoch: int, 
                counter, 
                args, 
                eventTimer: EventTimer,
                device, 
                criterion=nn.NLLLoss()):
    lm_model.train()
    cur_loss: torch.Tensor = torch.zeros(
        1, device=device)
    H = lm_model.model.init_hidden(1)
    cur_loss = 0
    acc_step = 0
    with eventTimer(f"epoch-{epoch}"):
        with eventTimer("sorter_sort"):
            perm_list = sorter.sort()

    for batch in range(len(c_data)):
        X, Y = c_data.trainset[perm_list[batch]]
        optimizer.zero_grad()

        H = repackage_hidden(H)
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("forward_pass"):
                Y_hat, H = lm_model.model(X, H)
                loss = criterion(Y_hat, Y)

        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("backward"):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(lm_model.parameters(), 0.25)

        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("sorter_step"):
                sorter.step(optimizer, batch)

        # Perform gradient accumulation depending on the current step
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("SGD_step"):
                lm_model.grad_copy_buffer.binary_op_(
                    [p.grad.data for p in lm_model.parameters() if p.requires_grad], ADD_TO_LEFT)
                acc_step += 1
                if (batch > 0 and batch % args.grad_acc == 0) or (batch == c_data.indices.individual_batch_cnt - 1) or (args.grad_acc == 1):
                    # (reached a minibatch size) or (reached the end and have remainding microbatch) or (no grad_acc)
                    lm_model.grad_copy_buffer.unary_op_(AVERAGE_BY_(acc_step))
                    lm_model.grad_copy_buffer.binary_op_(
                        [p.grad for p in lm_model.parameters() if p.requires_grad], RIGHT_COPY_)
                    optimizer.step()
                    lm_model.grad_copy_buffer.unary_op_(ZERO_)
                    acc_step = 0
            with eventTimer("communication"):
                lm_model.communicate_weight_inplace()

        if args.rank == 0:
            counter.update(1)
        cur_loss += loss.detach()

        if batch > 0 and batch % args.log_interval == 0 and args.rank == 0:
            print('| epoch {:3d} | {:5d}/{:50d} batches | node 0 loss {:.3f}'.format(
                epoch, batch, len(c_data), cur_loss.item() / batch))

    return cur_loss / c_data.indices.individual_batch_cnt


@torch.no_grad()
def evaluate_one_model(model: nn.Module, dataset: Dataset):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(dataset.data.shape[-1])
    criterion = nn.NLLLoss()
    for data, targets in dataset:
        output, hidden = model(data, hidden)
        hidden = repackage_hidden(hidden)
        total_loss += (len(data) * criterion(output, targets)).item()
        # total_loss += criterion(output, targets).item()
    return (total_loss / (len(dataset.data) - 1))


@torch.no_grad()
def d_LM_test(eval_dataset: Dataset, d_lm_model: D_Model, epoch: int):
    d_lm_model.eval()
    global_avg_model = d_lm_model.model
    global_avg_model.eval()
    global_test_loss = evaluate_one_model(global_avg_model, eval_dataset)
    global_test_ppl = torch.exp(torch.as_tensor(global_test_loss))
    print(f'| epoch {epoch:3d} | global avg ppl {global_test_ppl.item():.4f}|', flush=True)
    return global_test_ppl
