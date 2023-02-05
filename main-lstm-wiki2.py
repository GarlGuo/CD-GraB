import torch
import torch.distributed as dist
from d_data import *
from d_topology import *
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
from d_utils import print_rank_0
from d_eventTimer import EventTimer


warnings.filterwarnings('always')


def seed_everything(seed):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


parser = argparse.ArgumentParser(
    description="D-GraB with LSTM on WikiText-2")
parser.add_argument(
    "--log_interval",
    type=int,
    default=500,
    help="log train loss after {log_interval} steps",
)
parser.add_argument(
    "--node_cnt",
    type=int,
    default=16,
    help="number of decentralized nodes",
)
parser.add_argument(
    "--test_B",
    type=int,
    default=10,
    help="Batch size for the evaluation dataloader.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=5.0,
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
    "--clip",
    type=float,
    default=0.25,
    help="max grad norm",
)
parser.add_argument(
    "--sorter",
    type=str,
    default="D-GraB",
    choices=[
        "D-GraB",
        "I-B",
        "D-RR",
        "I-PB"
    ]
)
parser.add_argument("--epochs", type=int, default=50,
                    help="Total number of training epochs to perform.")
parser.add_argument("--seed", type=int, default=0,
                    help="A seed for reproducible training.")
parser.add_argument("--grad_acc", type=int, default=2,
                    help="grad acc is ignored")
parser.add_argument(
    "--n_cuda_per_process",
    default=1,
    type=int,
    help="# of subprocess for each mpi process.",
)  # only support 1 for now
parser.add_argument("--local_rank", default=None, type=str)
# unused for now since n_cuda_per_process is 1
parser.add_argument("--world", default=None, type=str)
parser.add_argument("--backend", default="nccl", type=str)

args = parser.parse_args()

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    timeout=datetime.timedelta(seconds=5000)
)

args.distributed = True and args.node_cnt > 1
cur_rank = dist.get_rank() if args.distributed else 0
args.rank = cur_rank

node_cnt = args.node_cnt
epochs = args.epochs
seed = args.seed
log_interval = args.log_interval

# set corresponding device id for each worker
if args.node_cnt == torch.cuda.device_count():
    args.dev_id = cur_rank
else:
    assert args.node_cnt % torch.cuda.device_count() == 0
    args.dev_id = cur_rank % torch.cuda.device_count()
device = torch.device(f'cuda:{args.dev_id}')


setattr(args, "use_cuda", device != torch.device("cpu"))

print_rank_0(cur_rank, vars(args))
seed_everything(args.seed)

graph: Graph = CentralizedGraph(args.node_cnt, cur_rank, args.world)
protocol = CentralizedProtocol(graph.rank, args.node_cnt)
torch.cuda.set_device(args.dev_id)
device = torch.device(f'cuda:{args.dev_id}')
dtype = torch.int64

eventTimer = EventTimer(device=device)

d_data = D_LM_Dataset(args, args.node_cnt, f'data{os.sep}wikitext-2', device=device)


def model_maker():
    return d_model.LSTMModel(
        d_data.ntokens, ninp=32, nhid=32, nlayers=2,
        dropout=0.2, seed=args.seed, device=device
    )


c_model = D_Model(graph.rank, args.node_cnt, protocol, model_maker)
sgd = torch.optim.SGD(c_model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    sgd, mode='min', factor=0.1, patience=5, threshold=1)

m = len(d_data)
n = node_cnt
d = sum(p.numel() for p in c_model.parameters() if p.requires_grad)

sorter = {
    "D-GraB": lambda: D_GraB_PairBalance(args.rank, n=n, m=m, d=d, device=device),
    "I-B": lambda: I_Balance(args.rank, n=n, m=m, d=d, device=device),
    "D-RR": lambda: D_RandomReshuffling(args.rank, n, m),
    "I-PB": lambda: I_PairBalance(args.rank, m=m, n=n, d=d, device=device)
}[args.sorter]()

counter = tqdm(range(len(d_data) * args.epochs), miniters=100)
global_test_ppls, global_val_ppls, global_train_val_ppls = [], [], []
local_train_losses = []

seed_everything(args.seed)
for e in range(1, args.epochs + 1):
    dist.barrier()
    d_LM_train(d_data, sgd, c_model, sorter, e, counter, args, eventTimer, device)

    print_rank_0(cur_rank, "validation on training dataset")
    dist.barrier()
    global_avg_train_ppl = d_LM_test(d_data.trainset_eval, c_model, e)

    print_rank_0(cur_rank, "validation on testing dataset")
    dist.barrier()
    global_avg_test_ppl = d_LM_test(d_data.test_dataset, c_model, e)

    lr_scheduler.step(global_avg_train_ppl)
