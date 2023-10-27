import torch
from d_cv_train import *
from d_algo import *
from tqdm.auto import tqdm
import argparse
import random
import os
import datetime
import warnings
from collections import OrderedDict
from functorch import make_functional_with_buffers, grad
from copy import deepcopy

warnings.filterwarnings('ignore')


def seed_everything(seed):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


parser = argparse.ArgumentParser(description="decentralized learning")
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
    "--update_B",
    type=int,
    default=64,
    help="Batch size for the training dataloader.",
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
    default=1e-2,
    help="weight decay",
)
parser.add_argument(
    "--sorter",
    type=str,
    default="CD-GraB",
    choices=[
        "CD-GraB",
        "D-RR",
        "I-B",
        "I-PB"
    ]
)
parser.add_argument("--epochs", type=int, default=100,
                    help="Total number of training epochs to perform.")
parser.add_argument("--seed", type=int, default=0,
                    help="A seed for reproducible training.")

args = parser.parse_args()
args.node_cnt = args.B

device = torch.device(f'cuda:0')
setattr(args, "use_cuda", device != torch.device("cpu"))


torch.cuda.set_device(0)
seed_everything(args.seed)


class LeNet(nn.Module):
    """
    Input - 3x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    ReLU
    F7 - 10 (Output)
    """

    def __init__(self, seed=0):
        super(LeNet, self).__init__()
        seed_everything(seed)
        self.convnet = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", nn.Conv2d(3, 6, kernel_size=(5, 5))),
                    ("relu1", nn.ReLU()),
                    ("s2", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("conv3", nn.Conv2d(6, 16, kernel_size=(5, 5))),
                    ("relu3", nn.ReLU()),
                    ("s4", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ("conv5", nn.Conv2d(16, 120, kernel_size=(5, 5))),
                    ("relu5", nn.ReLU()),
                ]
            )
        )
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("fc6", nn.Linear(120, 84)),
                    ("relu6", nn.ReLU()),
                    ("fc7", nn.Linear(84, 10)),
                ]
            )
        )

    def forward(self, x):
        out = self.convnet(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out

    def pred(self, x):
        y_scores = self(x)
        return torch.max(y_scores, dim=1)[1]


data_path = f'data{os.sep}cifar10'
if not os.path.exists(data_path):
    os.makedirs(data_path)

(trainset_X, trainset_Y), (testset_X, testset_Y) = torch.load(
    f'data{os.sep}cifar10{os.sep}lenet-raw-data.pt', map_location=torch.device('cuda'))



model = LeNet(seed=args.seed).to(device)


fmodel, params, buffers = make_functional_with_buffers(model)


def compute_loss_stateless_model(params, buffers, X, Y):
    return F.cross_entropy(fmodel(params, buffers, X.view(1, *X.shape)), Y.view(1, *Y.shape).long())


N = (last_even_num(len(trainset_X) // args.update_B)) * args.update_B
trainset_X, trainset_Y = trainset_X[:N], trainset_Y[:N]

n = args.node_cnt
m = N // n
trainset_X, trainset_Y = trainset_X.view(n, m, 3, 32, 32), trainset_Y.view(n, m)

d = sum(p.numel() for p in model.parameters() if p.requires_grad)
B = args.B
if args.sorter == 'CD-GraB':
    sorter = CReal_PairBalance_Simulated(args, n=n, m=m, d=d, device=device)
elif args.sorter == 'D-RR':
    sorter = [RandomShuffle(m, device=device) for _ in range(n)]
elif args.sorter == 'I-B':
    sorter = [GraB_Single(m, d, device=device) for _ in range(n)]
elif args.sorter == 'I-PB':
    sorter = [PairBalance_Single(m, d, device=device) for _ in range(n)]
else:
    raise NotImplementedError()


exp_details = f"{args.sorter}-node-{args.node_cnt}-lr-{args.lr}-B-{args.B}-seed-{args.seed}"

print(exp_details)
counter = tqdm(range(2 * m * args.epochs), miniters=100)


func_compute_sample_grad = torch.vmap(
    grad(compute_loss_stateless_model), in_dims=(None, None, 0, 0))
sgd = torchopt.sgd(lr=args.lr, momentum=args.momentum,
                   weight_decay=args.weight_decay)
opt_state = sgd.init(params)

results = {
    'train': {
        'acc': [], 'loss': []
    },
    'test': {
        'acc': [], 'loss': []
    },
    'parallel_herding_bounds': [],
}
for e in range(1, args.epochs + 1):
    opt_state_copy = copy.deepcopy(opt_state)
    params_copy = copy.deepcopy(params)
    avg_grad: torch.Tensor = d_cv_train_functorch(trainset_X,
                                                trainset_Y,
                                                func_compute_sample_grad,
                                                fmodel,
                                                params,
                                                buffers,
                                                sgd,
                                                opt_state,
                                                sorter,
                                                counter,
                                                e,
                                                n,
                                                B,
                                                args.update_B,
                                                d,
                                                device=device)

    if type(sorter) == list and (isinstance(sorter[0], GraB_Single) or isinstance(sorter[0], PairBalance_Single)):
        perm_list = torch.vstack([s.next_orders for s in sorter])
    elif isinstance(sorter, CReal_PairBalance_Simulated):
        perm_list = sorter.next_orders
    elif type(sorter) == list and isinstance(sorter[0], RandomShuffle):
        perm_list: torch.Tensor = torch.vstack(
            [torch.randperm(m) for _ in range(n)]).cuda()
    else:
        raise NotImplementedError()

    p_herding_bound, avg_grad_error = \
        empirical_parallel_herding_bound(trainset_X,
                                         trainset_Y,
                                         func_compute_sample_grad,
                                         fmodel,
                                         params_copy,
                                         buffers,
                                         sgd,
                                         opt_state_copy,
                                         counter,
                                         e,
                                         n,
                                         B,
                                         args.update_B,
                                         d,
                                         perm_list,
                                         avg_grad,
                                         device=device)

    test_acc, test_loss = c_cv_test(
        testset_X, testset_Y, model, params, device=device)
    results['test']['acc'].append(test_acc)
    results['test']['loss'].append(test_loss)
    print(f'epoch {e} | test acc {test_acc:.1f}% | ')

    train_acc, train_loss = c_cv_test(
        trainset_X.view(trainset_X.numel() // (3 * 32 * 32), 3, 32, 32), 
        trainset_Y.view(-1), model, params, device=device)
    results['train']['acc'].append(train_acc)
    results['train']['loss'].append(train_loss)
    print(f'epoch {e} | train loss {train_loss:.3f} | ', flush=True)

    results['parallel_herding_bounds'].append(p_herding_bound)
    print(f'epoch {e} | train loss {p_herding_bound:.3f} | avg gradient error {avg_grad_error.mean().item():.3f} ', flush=True)


exp_folder = f"results{os.sep}lenet-cifar10{os.sep}{exp_details}"
if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)
torch.save(results, f"{exp_folder}{os.sep}results.pt")
