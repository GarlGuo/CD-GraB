import torch
import torch.distributed as dist
import torch.nn.functional as F
from d_time_series_train import *
from d_data import *
import d_model
from d_algo import *
from d_utils import seed_everything
from tqdm.auto import tqdm
import argparse
import random
import os
import datetime
import warnings
import torchopt
from functorch import grad, make_functional_with_buffers
from d_utils import print_rank_0
from d_eventTimer import EventTimer
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(
    description="Distributed learning with CD-GraB on MLP on M4 weekly task")
parser.add_argument(
    "--node_cnt",
    type=int,
    default=32,  
    help="number of decentralized nodes",
)
parser.add_argument(
    "--B",
    type=int,
    default=32,
    help="Batch size for the training dataloader.",
)
parser.add_argument(
    "--test_B",
    type=int,
    default=128,
    help="Batch size for the evaluation dataloader.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--momentum",
    type=float,
    default=0.9,
    help="momentum",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0,
    help="weight decay",
)
parser.add_argument(
    "--nlayers",
    type=int,
    default=3,
    help="number of MLP layers",
    choices=[3, 4, 5]
)
parser.add_argument(
    "--nhid",
    type=int,
    default=64,
    help="hidden layer dimension",
    choices=[64, 128, 256]
)
parser.add_argument(
    "--in_seq",
    type=int,
    default=20,
    help="input sequence length",
    choices=[20, 30, 40]
)
parser.add_argument(
    "--out_seq",
    type=int,
    default=6,
    help="output sequence length",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=1,
    help="number of steps",
    choices=[1, 2, 3]
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
parser.add_argument("--epochs", type=int, default=30,
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


model = d_model.Auto_MLP(
    input_dim=1, output_dim=1, input_length=args.in_seq,
    num_steps=args.num_steps, hidden_dim=args.nhid,
    num_layers=args.nlayers, use_RevIN=False,
    seed=args.seed, device=device
).to(device)
fmodel, params, buffers = make_functional_with_buffers(model)


def compute_loss_stateless_model(params, buffers, inp, tgt):
    denorm_outs, norm_outs, norm_tgts = fmodel(
        params, buffers, inp.view(1, *inp.shape), tgt.view(1, *tgt.shape))
    return F.mse_loss(norm_outs, norm_tgts)


func_compute_sample_grad = torch.vmap(
    grad(compute_loss_stateless_model), in_dims=(None, None, 0, 0))


class PerEpochStepLR:
    def __init__(self, init_lr, step=10, gamma=0.25) -> None:
        self.lr = init_lr
        self.epoch_step = step
        self.epochs_count_down = step
        self.has_updated = False
        self.gamma = gamma

    def __call__(self, c):
        return self.lr

    def step(self):
        self.epochs_count_down -= 1
        if self.epochs_count_down == 0:
            self.lr *= self.gamma
            self.epochs_count_down = self.epoch_step  # reset


lr_scheduler = PerEpochStepLR(args.lr)
sgd = torchopt.sgd(lr=lr_scheduler, momentum=args.momentum,
                   weight_decay=args.weight_decay)
opt_state = sgd.init(params)


d_data = C_M4_Dataset(
    args,
    args.node_cnt,
    (args.B // args.node_cnt),
    input_length=args.in_seq,
    output_length=args.out_seq,
    freq='Weekly',
    device=device
)


m = len(d_data)
n = args.node_cnt
d = sum(p.numel() for p in model.parameters() if p.requires_grad)
B = args.B
microbatch = B // n
sorter = {
    "CD-GraB": (lambda: CD_GraB(args.rank, args, n=n, m=m, d=d, device=device)),
    "D-RR": (lambda: D_RR(args.rank, n, m, device=device)),
}[args.sorter]()

exp_details = f"sorter-{args.sorter}-node-{args.node_cnt}-lr-{args.lr}-B-{args.B}-seed-{args.seed}-nhid-{args.nhid}-nlayers-{args.nlayers}"
basic_dir = f'results{os.sep}M4'
print_rank_0(cur_rank, vars(args))
counter = tqdm(range(len(d_data) * args.epochs), miniters=100)

results = {
    'train': {
        'rmse': []
    },
    'val': {
        'rmse': []
    },
    'test': {
        'rmse': [],
        'smape': []
    }
}
torch.save((model, sorter), f'model{os.sep}M4{os.sep}{exp_details}-epoch-0.pt')
for e in range(1, args.epochs + 1):
    dist.barrier()
    d_time_series_train_epoch(
        cur_rank,
        d_data,
        func_compute_sample_grad,
        fmodel,
        params,
        buffers,
        sgd,
        opt_state,
        sorter,
        counter,
        eventTimer,
        e,
        n,
        microbatch,
        d,
        device=device
    )

    print_rank_0(cur_rank, f'Epoch {e} has finished!')
    lr_scheduler.step()

    dist.barrier()

    train_rmse, _, _ = d_time_series_eval_epoch(
        d_data.train_loader_eval, model, params, device=device)
    eval_rmse, _, _ = d_time_series_eval_epoch(
        d_data.valid_loader, model, params, device=device)
    test_rmse, test_preds, test_tgts = d_time_series_eval_epoch(
        d_data.test_loader, model, params, device=device)

    test_preds = (test_preds * torch.tensor(d_data.test_dataset.ts_stds, device=device).reshape(-1, 1, 1)) + \
        torch.tensor(d_data.test_dataset.ts_means,
                     device=device).reshape(-1, 1, 1)
    test_tgts = (test_tgts * torch.tensor(d_data.test_dataset.ts_stds, device=device).reshape(-1, 1, 1)) + \
        torch.tensor(d_data.test_dataset.ts_means,
                     device=device).reshape(-1, 1, 1)
    _, _, _, test_smape = sMAPE(test_preds, test_tgts)

    print_rank_0(
        cur_rank, f"Epoch {e} | Train RMSE: {train_rmse.item():0.3f} | Valid RMSE {eval_rmse.item():0.3f} | Test RMSE {test_rmse.item():0.3f} | Test SMAPE {test_smape.item():0.3f}")

    results["train"]['rmse'].append(train_rmse)
    results["val"]['rmse'].append(eval_rmse)
    results["test"]['rmse'].append(test_rmse)
    results["test"]['smape'].append(test_smape)


exp_folder = f"results{os.sep}M4{os.sep}{exp_details}"
time_folder = f"{exp_folder}{os.sep}time"
if args.rank == 0:
    if not os.path.exists(time_folder):
        os.makedirs(time_folder)

dist.barrier()
eventTimer.save_results(f"{time_folder}time-{cur_rank}.pt")

if args.rank == 0:
    print('saving expDetails results')
    torch.save(results, f"{exp_folder}{os.sep}results.pt")
