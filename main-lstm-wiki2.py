import torch
import torch.distributed as dist
from d_data import *
from d_lm_data import *
from d_lm_train import *
import d_model
from d_model import *
from d_algo import *
from tqdm.auto import tqdm
import argparse
import random
import os
import datetime
import warnings
import torchopt
from d_utils import print_rank_0
from d_eventTimer import EventTimer
import functorch


warnings.filterwarnings('ignore')
def seed_everything(seed):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


parser = argparse.ArgumentParser(
    description="distributed learning with CD-GraB on LSTM on WikText-2 task")
parser.add_argument(
    "--node_cnt",
    type=int,
    default=4,
    help="number of decentralized nodes",
)
parser.add_argument(
    "--B",
    type=int,
    default=32,
    help="Batch size for the training dataloader.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=5,
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
    "--bptt",
    type=int,
    default=35,
    help="sequence length",
)
parser.add_argument(
    "--sorter",
    type=str,
    default="D-RR",
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
)  # only support 1 for now
parser.add_argument("--local_rank", default=None, type=str)
# unused for now since n_cuda_per_process is 1
parser.add_argument("--world", default=None, type=str)
parser.add_argument("--backend", default="nccl", type=str) # nccl

args = parser.parse_args()

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    timeout=datetime.timedelta(seconds=5000)
)

args.distributed = True and args.node_cnt > 1
cur_rank = dist.get_rank() if args.distributed else 0
args.rank = cur_rank

epochs = args.epochs
seed = args.seed


if args.node_cnt == torch.cuda.device_count():
    print_rank_0(cur_rank, "Running one process per GPU")
    args.dev_id = cur_rank
else:
    assert args.node_cnt % torch.cuda.device_count() == 0
    args.dev_id = cur_rank % torch.cuda.device_count()
    print(f"Process {cur_rank} is running on cuda:{args.dev_id}")
device = torch.device(f'cuda:{args.dev_id}')
setattr(args, "use_cuda", device != torch.device("cpu"))

print_rank_0(cur_rank, vars(args))
seed_everything(args.seed)

torch.cuda.set_device(args.dev_id)
device = torch.device(f'cuda:{args.dev_id}')
eventTimer = EventTimer(device=device)


d_data = D_LM_Dataset(
    args, 
    args.node_cnt, 
    args.B,
    f'data{os.sep}wikitext-2', 
    device=device
)
model = d_model.LSTMModel(
    d_data.ntokens, ninp=32, nhid=32, nlayers=2
).to(device=device)

fmodel, params, buffers = functorch.make_functional_with_buffers(model)

class ReduceLROnPlateau:
    def __init__(self, init_lr, factor=0.1, patience=5, threshold=1) -> None:
        self.lr = init_lr
        self.factor = factor
        self.threshold = threshold
        self.num_bad_epochs = 0
        self.best = float('inf')
        self.patience = patience

    def step(self, metrics):
        if metrics < self.best - self.threshold:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.lr = self.lr * self.factor
            self.num_bad_epochs = 0

    def __call__(self, c):
        return self.lr


class PerEpochStepLR:
    def __init__(self, init_lr, step=10, gamma=0.1) -> None:
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

# func_compute_sample_grad = torch.vmap(functorch.grad(compute_loss_stateless_model, has_aux=True), in_dims=(None, None, 0, 0, 0))
# lr_scheduler = ReduceLROnPlateau(args.lr, factor=0.1, patience=5, threshold=1)
lr_scheduler = PerEpochStepLR(args.lr)
with eventTimer('SGD'):
    sgd = torchopt.sgd(lr=lr_scheduler, momentum=args.momentum,
                    weight_decay=args.weight_decay)
    opt_state = sgd.init(params)


m = len(d_data)
n = args.node_cnt
d = sum(p.numel() for p in model.parameters() if p.requires_grad)
B = args.B
microbatch = B // n
with eventTimer('sorter'):
    sorter = {
        "CD-GraB": (lambda: CD_GraB(args.rank, args, n=n, m=m, d=d, device=device)),
        "D-RR": (lambda: D_RR(args.rank, n, m, device=device)),
    }[args.sorter]()

exp_details = f"{args.sorter}-node-{args.node_cnt}-lr-{args.lr}-B-{args.B}-seed-{args.seed}"
counter = tqdm(range(m * args.epochs), miniters=100)
results = {
    'train': {'ppl': [], 'loss': []},
    'test': {'ppl': [], 'loss': []},
}
for e in range(1, args.epochs + 1):
    torch.cuda.empty_cache()
    dist.barrier()
    LM_train(cur_rank,
        d_data,
        model,
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

    train_val_ppl, full_train_loss = LM_test(cur_rank, d_data.trainset_eval, model, params)    
    # lr_scheduler.step(train_val_ppl)
    lr_scheduler.step()
    print_rank_0(cur_rank, f'epoch {e} | train ppl {train_val_ppl:.2f} | full train loss {full_train_loss:.3f} ')

    dist.barrier()
    test_ppl, test_loss = LM_test(cur_rank, d_data.test_dataset, model, params)

    results['train']['loss'].append(full_train_loss)
    results['train']['ppl'].append(train_val_ppl)

    results['test']['loss'].append(test_loss)
    results['test']['ppl'].append(test_ppl)
    print_rank_0(cur_rank, f'epoch {e} | test ppl {test_ppl:.2f} | ')


exp_folder = f"results{os.sep}lstm-wiki2{os.sep}{exp_details}"
time_folder = f"{exp_folder}{os.sep}time{os.sep}"
if cur_rank == 0:
    if not os.path.exists(time_folder):
        os.makedirs(time_folder)

dist.barrier()
eventTimer.save_results(f"{time_folder}time-{cur_rank}.pt")

if cur_rank == 0:
    print('saving expDetails results')
    torch.save(results, f"{exp_folder}{os.sep}results.pt")
