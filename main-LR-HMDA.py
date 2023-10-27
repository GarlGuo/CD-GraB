import torch
from tqdm.auto import tqdm
import functorch
import argparse
import torchopt
import pandas as pd
import os
import random
import torch.nn.functional as F
import sklearn.preprocessing
import numpy as np
import sklearn.model_selection
import torch.distributed as dist
from d_data import *
from d_algo import *
from d_utils import seed_everything
from tqdm.auto import tqdm
import argparse
import random
import os
import datetime
import warnings
from functorch import grad, make_functional_with_buffers
from d_utils import print_rank_0
from d_eventTimer import EventTimer
from d_hmda import *

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(
    description="distributed learning with CD-GraB on LR on HMDA dataset task")
parser.add_argument(
    "--node_cnt",
    type=int,
    default=4,
)
parser.add_argument(
    "--B",
    type=int,
    default=16,
    help="Batch size for the training dataloader.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-2,
    help="learning rate",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="momentum",
)
parser.add_argument(
    "--sorter",
    type=str,
    default="CD-GraB",
    choices=[
        "CD-GraB",
        "D-RR",
    ]
)
parser.add_argument("--epochs", type=int, default=50,
                    help="Total number of training epochs to perform.")
parser.add_argument("--seed", type=int, default=0,
                    help="A seed for reproducible training.")
parser.add_argument(
    "--n_cuda_per_process",
    default=1,
    type=int,
    help="# of subprocess for each mpi process.",
)
parser.add_argument("--local_rank", default=None, type=str)
# unused for now since n_cuda_per_process is 1
parser.add_argument("--world", default=None, type=str)
parser.add_argument("--backend", default="nccl", type=str)  # nccl

args = parser.parse_args()

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    timeout=datetime.timedelta(seconds=10000)
)
args.distributed = args.node_cnt > 1
cur_rank = dist.get_rank() if args.distributed else 0
args.rank = cur_rank

if args.node_cnt == torch.cuda.device_count():
    print_rank_0(cur_rank, "Running one process per GPU")
    args.dev_id = cur_rank
else:
    assert args.node_cnt % torch.cuda.device_count() == 0
    args.dev_id = cur_rank % torch.cuda.device_count()
device = torch.device(f'cuda:{args.dev_id}')
setattr(args, "use_cuda", device != torch.device("cpu"))

eventTimer = EventTimer(device=device)


torch.cuda.set_device(args.dev_id)
torch.cuda.empty_cache()

print_rank_0(cur_rank, vars(args))
seed_everything(args.seed)


random.seed(args.seed)
np.random.seed(args.seed)

dataset_torch_file_addr = f'data{os.sep}HMDA{os.sep}features-processed-NY-2017.pt'
target_torch_file_addr = f'data{os.sep}HMDA{os.sep}targets-processed-NY-2017.pt'

trainset_X, testset_X = torch.load(dataset_torch_file_addr, map_location='cpu')
trainset_y, testset_y = torch.load(target_torch_file_addr, map_location='cpu')

torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)


model = torch.nn.Linear(trainset_X.shape[1], 1).to(device)
fmodel, params, buffers = functorch.make_functional_with_buffers(model)


def last_even_number(num): return num if num % 2 == 0 else num - 1


n = args.node_cnt
B = args.B
microbatch = B // n
N = (last_even_number(len(trainset_X) // B)) * B
m = N // n
d = sum(p.numel() for p in model.parameters())
trainset_X, trainset_y = trainset_X[:N], trainset_y[:N]

trainset_X = trainset_X.view(n, m, trainset_X.shape[-1])
trainset_y = trainset_y.view(n, m)
trainset_X, trainset_y = trainset_X.to(device), trainset_y.to(device)

trainset_X_eval = trainset_X.view(N, trainset_X.shape[-1])
trainset_y_eval = trainset_y.view(-1)

sorter = {
    "CD-GraB": (lambda: CD_GraB(args.rank, args, n=n, m=m, d=d, microbatch=microbatch, device=device)),
    "D-RR": (lambda: D_RR(args.rank, n, m, device=device)),
}[args.sorter]()

exp_details = f"sorter-{args.sorter}-node-{args.node_cnt}-lr-{args.lr}-B-{args.B}-seed-{args.seed}"
print_rank_0(cur_rank, vars(args))
counter = tqdm(range(m * args.epochs), miniters=100)


model_name = 'LR'
exp_details = f"{model_name}-hmda-{args.sorter}-lr-{args.lr}-B-{args.B}-seed-{args.seed}"

result_path = f"results{os.sep}{model_name}-hmda{os.sep}{exp_details}"
if not os.path.exists(result_path) and cur_rank == 0:
    os.makedirs(result_path)


fmodel, params, buffers = functorch.make_functional_with_buffers(model)


def compute_loss_stateless_model(params, buffers, X, y):
    yhat = fmodel(params, buffers, X.view(1, *X.shape)).squeeze()
    return F.binary_cross_entropy_with_logits(yhat, y.squeeze())


func_per_example_grad = torch.vmap(functorch.grad(
    compute_loss_stateless_model), in_dims=(None, None, 0, 0))


max_train_steps = int(m * args.epochs)
with eventTimer('SGD'):
    optimizer = torchopt.sgd(lr=args.lr, momentum=0.9)
    opt_state = optimizer.init(params)


@torch.no_grad()
def HMDA_eval(dset_X, dset_y, model, params):
    total_correct = 0
    total_loss = 0
    for i, p in enumerate(model.parameters()):
        p.data.copy_(params[i])
    dset_X = dset_X.view(dset_X.numel() // dset_X.shape[-1], dset_X.shape[-1])
    dset_y = dset_y.view(-1)
    test_B = int(2 ** 10)
    for B_idx in range(0, len(dset_X), test_B):
        batch = torch.arange(B_idx, min(B_idx + test_B, len(dset_X)))
        X = dset_X[batch].to(device)
        y = dset_y[batch].to(device)
        logits = model(X).squeeze()
        predictions = (logits > 0).float()
        correct = (predictions == y).sum().item()
        total_correct += correct
        loss = F.binary_cross_entropy_with_logits(logits, y)
        total_loss += loss * len(batch)
    return total_loss / len(dset_X), total_correct / len(dset_X)


results = {
    'train': {'loss': [], 'acc': []},
    'test': {'loss': [], 'acc': []},
}
for e in range(1, args.epochs + 1):
    d_HMDA_train(cur_rank,
                trainset_X,
                trainset_y,
                func_per_example_grad,
                fmodel,
                params,
                buffers,
                optimizer,
                opt_state,
                sorter,
                counter,
                eventTimer,
                e,
                n,
                microbatch,
                d,
                device=device)
    torch.cuda.empty_cache()
    full_train_loss, train_acc = HMDA_eval(trainset_X_eval, trainset_y_eval, model, params)
    test_loss, test_acc = HMDA_eval(testset_X, testset_y, model, params)
    print_rank_0(
        cur_rank, f'Epoch {e} | full train loss {full_train_loss:.6f}')
    print_rank_0(
        cur_rank, f'Epoch {e} | test acc {100 * test_acc:.3f}%')

    results["train"]["acc"].append(train_acc)
    results["train"]["loss"].append(full_train_loss)

    results["test"]["acc"].append(test_acc)
    results["test"]["loss"].append(test_loss)


exp_folder = f"results{os.sep}LR-HMDA{os.sep}{exp_details}"
time_folder = f"{exp_folder}{os.sep}time{os.sep}"
if args.rank == 0:
    if not os.path.exists(time_folder):
        os.makedirs(time_folder)

dist.barrier()
eventTimer.save_results(f"{time_folder}time-{cur_rank}.pt")

if args.rank == 0:
    print('saving expDetails results')
    torch.save(results, f"{exp_folder}{os.sep}results.pt")
