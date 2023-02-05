import torch
import torch.distributed as dist
from d_glue_train import *
from d_data import *
from d_topology import *
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


task_name = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
parser = argparse.ArgumentParser(
    description="all-reduce GraB. Tune linear head on QQP")
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
    default=128,
    help="Batch size for the evaluation dataloader.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-4,
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
    default=8,
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
parser.add_argument("--task_name", default='qqp', choices=task_name, type=str)
parser.add_argument("--local_rank", default=None, type=str)
# unused for now since n_cuda_per_process is 1
parser.add_argument("--world", default=None, type=str)
parser.add_argument("--backend", default="nccl", type=str)

args = parser.parse_args()

dist.init_process_group(
    backend=args.backend,
    init_method="env://",
    timeout=datetime.timedelta(seconds=60000)
)

args.distributed = args.node_cnt > 1
cur_rank = dist.get_rank() if args.distributed else 0
args.rank = cur_rank

dtype = torch.float32
if args.node_cnt == torch.cuda.device_count():
    print_rank_0(cur_rank, "Running one process per GPU")
    args.dev_id = cur_rank
else:
    assert (args.node_cnt % torch.cuda.device_count() == 0) or args.node_cnt <= torch.cuda.device_count()
    args.dev_id = cur_rank % torch.cuda.device_count()
    print(f"Process {cur_rank} is running on cuda:{args.dev_id}")
device = torch.device(f'cuda:{args.dev_id}')
setattr(args, "use_cuda", device != torch.device("cpu"))

eventTimer = EventTimer(device=device)

graph: Graph = CentralizedGraph(args.node_cnt, cur_rank, args.world)
protocol = CentralizedProtocol(graph.rank, args.node_cnt)

torch.cuda.set_device(args.dev_id)
torch.cuda.empty_cache()
print_rank_0(vars(args))
seed_everything(args.seed)

exp_config = get_glue_config(args, args.seed)
bert_model = lambda: BERT_model_maker(args, exp_config, args.seed, device)[0]

d_data = D_GLUE_Embeddings(
    args, exp_config, args.node_cnt, bert_model, device=device)
d_model = D_Model(graph.rank, args.node_cnt, protocol,
                      (lambda: BERT_LinearHead(exp_config['num_labels'], device, args.seed)))
sgd = torch.optim.SGD(d_model.model.classifier.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
del bert_model

args.d = sum(p.numel() for p in d_model.parameters() if p.requires_grad)
sorter = {
    "D-GraB": lambda: D_GraB_PairBalance(args.rank, n=args.node_cnt, m=len(d_data), d=args.d, device=device),
    "I-B": lambda: I_Balance(args.rank, n=args.node_cnt, m=len(d_data), d=args.d, device=device),
    "D-RR": lambda: D_RandomReshuffling(args.rank, args.node_cnt, len(d_data)),
    "I-PB": lambda: D_PairBalance(args.rank, m=len(d_data), n=args.node_cnt,
                                  d=sum(p.numel() for p in d_model.parameters() if p.requires_grad), device=device)
}[args.sorter]()

exp_details = f"{args.task_name}-sorter-{args.sorter}-node-{args.node_cnt}-lr-{args.lr}-train-B-{args.train_B}-grad_acc-{args.grad_acc}-seed-{args.seed}"

print_rank_0(exp_details)
counter = tqdm(range(d_data.indices.individual_batch_cnt *
               args.epochs), miniters=100)

local_train_loss = []
global_test_acc = []
global_train_acc = []
full_train_losses = []
for e in range(args.epochs):
    dist.barrier()
    print_rank_0(cur_rank, vars(args))
    local_train_loss.append(
        d_bert_train(d_data, sgd, d_model, sorter, e, counter,
                        args, eventTimer, grad_acc=args.grad_acc)
    )
    dist.barrier()

    full_train_loss, train_acc = d_full_train_loss(
        exp_config, d_data.trainloader, d_model, cur_rank, device
    )
    full_train_losses.append(full_train_loss)
    global_train_acc.append(train_acc)
    dist.barrier()
    global_avg_test_score = d_bert_test(
        d_data.testloader, d_model, e, exp_config, cur_rank, device=device)
    dist.barrier()
    global_test_acc.append(global_avg_test_score)
