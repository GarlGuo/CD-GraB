import torch
import torch.distributed as dist
from d_cv_train import *
from d_data import *
from d_topology import *
import d_model
from d_model import *
from d_algo import *
from tqdm.auto import tqdm
import argparse
import random
import os
import datetime
from c_criterion import *
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


parser = argparse.ArgumentParser(description="all-reduce GraB")
parser.add_argument(
    "--log_interval",
    type=int,
    default=1000,
    help="log train loss after {log_interval} steps",
)
parser.add_argument(
    "--node_cnt",
    type=int,
    default=5,
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
    default=512,
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
    "--grad_acc",
    type=int,
    default=2,
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
parser.add_argument("--epochs", type=int, default=3, help="Total number of training epochs to perform.")
parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
parser.add_argument(
    "--n_cuda_per_process",
    default=1,
    type=int,
    help="# of subprocess for each mpi process.",
) # only support 1 for now
parser.add_argument("--local_rank", default=None, type=str)
parser.add_argument("--world", default=None, type=str) # unused for now since n_cuda_per_process is 1
parser.add_argument("--backend", default="nccl", type=str)

args = parser.parse_args()

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    timeout=datetime.timedelta(seconds=200)
)

args.distributed = True and args.node_cnt > 1
cur_rank = dist.get_rank() if args.distributed else 0
args.rank = cur_rank

dtype = torch.float32
if args.node_cnt == torch.cuda.device_count():
    print_rank_0(cur_rank, "Running one process per GPU")
    args.dev_id = cur_rank
else:
    assert args.node_cnt % torch.cuda.device_count() == 0
    args.dev_id = cur_rank % torch.cuda.device_count()
    print(f"Process {cur_rank} is running on cuda:{args.dev_id}")
print(f'device id: {args.dev_id}')
device = torch.device(f'cuda:{args.dev_id}')
setattr(args, "use_cuda", device != torch.device("cpu"))

eventTimer = EventTimer(device=device)

graph: Graph = CentralizedGraph(args.node_cnt, cur_rank, args.world)
W = graph.get_W()
P = graph.get_P()
protocol = CentralizedProtocol(graph.rank, args.node_cnt)
torch.cuda.set_device(args.dev_id)
print_rank_0(cur_rank, f'W: {W}')
seed_everything(args.seed)

d_data = D_CIFAR10(args.node_cnt, train_B=args.train_B, test_B=args.test_B, device=device, d_dataset_format=partitioned_dset_maker, args=args)
model_maker = lambda: d_model.LeNet(seed=args.seed).to(device)

d_model = D_Model(graph.rank, args.node_cnt, protocol, model_maker)
sgd = torch.optim.SGD(d_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
criterion = ConvexMultiClassification_Loss(d_model)

args.d = sum(p.numel() for p in d_model.parameters() if p.requires_grad)
sorter = {
    "D-GraB": lambda: D_GraB_PairBalance(args.rank, n=args.node_cnt, m=len(d_data), d=args.d, device=device),
    "I-B": lambda: I_Balance(args.rank, n=args.node_cnt, m=len(d_data), d=args.d, device=device),
    "D-RR": lambda: D_RandomReshuffling(args.rank, args.node_cnt, len(d_data)),
    "I-PB": lambda: D_PairBalance(args.rank, m=len(d_data), n=args.node_cnt,
        d=sum(p.numel() for p in d_model.parameters() if p.requires_grad), device=device)
}[args.sorter]()

exp_details = f"{args.sorter}-node-{args.node_cnt}-backend-{args.backend}-lr-{args.lr}-epoch-{args.epochs}-train-B-{args.train_B}-grad_acc-{args.grad_acc}-seed-{args.seed}"
print_rank_0(exp_details)
counter = tqdm(range(len(d_data) * args.epochs), miniters=100)

global_test_acc = []
global_full_train_loss = []
global_full_train_acc = []
inidvidual_LOSS = []

for e in range(args.epochs):
    dist.barrier()
    individual_loss = d_cv_train(d_data, sgd, d_model, sorter, criterion, e, counter, args, eventTimer, grad_acc=args.grad_acc)
    inidvidual_LOSS.append(individual_loss)
    
    dist.barrier()
    global_avg_test_score = d_cv_test(d_data.testloader, d_model, e, cur_rank, device=device)
    global_test_acc.append(global_avg_test_score)
    
    dist.barrier()
    if args.rank == 0:
        cur_e_trainLoss, cur_e_trainAcc = d_cv_full_train_loss(args.rank, d_data.trainloader,d_model, criterion, device=device)
        global_full_train_loss.append(cur_e_trainLoss)
        global_full_train_acc.append(cur_e_trainAcc)
