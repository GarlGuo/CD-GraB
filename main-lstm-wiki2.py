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
    description="decentralized learning with D-GraB")
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
    "--train_B",
    type=int,
    default=1,
    help="Batch size for the training dataloader.",
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
    default="d-onlinebalance",
    choices=[
        "d-onlinebalance",
        "d-grab",
        "d-rr",
        "d-with-r",
        "d-ind-pairb"
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

graph: Graph = CentralizedGraph(args.node_cnt, cur_rank, args.world)
protocol = CentralizedProtocol(graph.rank, args.node_cnt)
torch.cuda.set_device(args.dev_id)
device = torch.device(f'cuda:{args.dev_id}')
dtype = torch.int64

eventTimer = EventTimer(device=device)

d_data = DReal_LM_Dataset(
    args, args.node_cnt, f'data{os.sep}wikitext-2', device=device)


def model_maker():
    return d_model.LSTMModel(
        d_data.ntokens, ninp=32, nhid=32, nlayers=2,
        dropout=0.2, seed=args.seed, device=device
    )


c_model = DReal_Model(graph.rank, args.node_cnt, protocol, model_maker)
sgd = torch.optim.SGD(c_model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    sgd, mode='min', factor=0.1, patience=5, threshold=1)

m = len(d_data)
n = node_cnt
d = sum(p.numel() for p in c_model.parameters() if p.requires_grad)

sorter = {
    "d-onlinebalance": lambda: D_GraB_PairBalance(args.rank, args, n=args.node_cnt, m=len(d_data),
                                                            d=sum(p.numel() for p in c_model.parameters() if p.requires_grad), device=device),
    "d-grab": lambda: I_Balance(args.rank, args, n=args.node_cnt, m=len(d_data),
                                       d=sum(p.numel() for p in c_model.parameters() if p.requires_grad), device=device),
    "d-rr": lambda: D_RR(args.rank, args.node_cnt, len(d_data)),
    "d-with-r": lambda: CReal_WithR(args.rank, args.node_cnt, len(d_data)),
    "d-ind-pairb": lambda: D_PairBalance(args.rank, args, m=len(d_data), n=args.node_cnt,
                                           d=sum(p.numel() for p in c_model.parameters() if p.requires_grad), device=device)
}[args.sorter]()


exp_details = f"{args.sorter}-node-{args.node_cnt}-lr-{args.lr}-train-B-{args.train_B}-seed-{args.seed}"
basic_dir = f"real-results{os.sep}lstm-wiki2"

counter = tqdm(range(len(d_data) * args.epochs), miniters=100)
global_test_ppls, global_val_ppls, global_train_val_ppls = [], [], []
local_train_losses = []

seed_everything(args.seed)
for e in range(1, args.epochs + 1):
    dist.barrier()
    local_train_loss = cReal_LM_train(d_data, sgd, c_model, sorter, e,
                                      counter, args, eventTimer, device)
    local_train_losses.append(local_train_loss)

    print_rank_0(cur_rank, "validation on training dataset")
    dist.barrier()
    global_avg_train_val_ppl = dReal_LM_test(
        d_data.trainset_eval, c_model, e, args.rank)

    print_rank_0(cur_rank, "validation on testing dataset")
    dist.barrier()
    global_avg_test_ppl = dReal_LM_test(
        d_data.test_dataset, c_model, e, args.rank)

    global_train_val_ppls.append(global_avg_train_val_ppl)
    global_test_ppls.append(global_avg_test_ppl)
    lr_scheduler.step(global_avg_train_val_ppl)

if cur_rank == 0:
    global_test_ppls = torch.as_tensor(global_test_ppls)
    global_train_val_ppls = torch.as_tensor(global_train_val_ppls)

eventTimer.save_results(
    f'{basic_dir}{os.sep}{exp_details}__eventTimer{cur_rank}.pt')
torch.save(
    local_train_losses, f"real-results{os.sep}glue{os.sep}local-train-loss-rank-{cur_rank}-{exp_details}.pt"
)
if cur_rank == 0:
    torch.save(global_train_val_ppls,
               f"{basic_dir}{os.sep}global-train-ppl-{exp_details}.pt")
    torch.save(global_test_ppls,
               f"{basic_dir}{os.sep}global-test-ppl-{exp_details}.pt")

print_rank_0(cur_rank, f"{graph} - {args.sorter} finished")
