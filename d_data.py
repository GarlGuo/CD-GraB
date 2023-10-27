import torch
import torchvision
from torch.utils.data import *
import torch.distributed as dist
import os
import random
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
import numpy as np
from collections.abc import Callable
import torch
from datasets import load_dataset
from torch.utils.data import *
import numpy as np
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    default_data_collator,
)
from tqdm.auto import tqdm


def last_even_num(odd_or_even_num):
    if odd_or_even_num % 2 == 0:
        return odd_or_even_num
    else:
        return odd_or_even_num - 1


class D_Dataset_Indices:
    def __init__(self, node_cnt, node_idx_map, args) -> None:
        super().__init__()
        self.node_cnt = node_cnt
        self.individual_batch_cnt = node_idx_map.shape[1]
        self.local_indices = node_idx_map[args.rank]


class D_Dataset_Partitioned(D_Dataset_Indices):
    def __init__(self, dataset, node_cnt, args, device=None) -> None:
        # must shuffle before dividing
        # shuffle, split
        # shuffle before reshape, instead of simply calling, arange => random.shuffle => reshape
        data_idx = np.arange(len(dataset))
        random.shuffle(data_idx)
        total_batch_cnt = last_even_num(len(dataset) // node_cnt) * node_cnt
        node_idx_map = torch.tensor(data_idx[:total_batch_cnt], dtype=torch.int64, device=device).reshape(
            (node_cnt, total_batch_cnt // node_cnt))
        super().__init__(node_cnt, node_idx_map, args)


def partitioned_dReal_dset_maker(
    dset, nodes, args, device=None): return D_Dataset_Partitioned(dset, nodes, args, device=device)


class D_VisionData(Dataset):
    def __init__(self, node_cnt, dset_maker: VisionDataset,
                 dset_addr, train_transform, test_transform, d_dataset_format=partitioned_dReal_dset_maker,
                 download=False, test_B=128, device=None, dtype=torch.float32, args=None, **kw) -> None:
        super().__init__()
        self.node_cnt = node_cnt
        self.trainset: VisionDataset = dset_maker(
            root=dset_addr, train=True, download=download, transform=train_transform)
        self.testset: VisionDataset = dset_maker(
            root=dset_addr, train=False, download=download, transform=test_transform)

        self.indices: D_Dataset_Indices = d_dataset_format(
            self.trainset, self.node_cnt, args=args, **kw)

        self.device = device
        self.dtype = dtype

        self.trainloader = DataLoader(self.trainset, batch_size=test_B)
        self.testloader = DataLoader(self.testset, batch_size=test_B)

        self.images = torch.stack([self.trainset[idx][0] for idx in self.indices.local_indices.view(-1)]).to(
            device=self.device, dtype=self.dtype)
        self.targets = torch.tensor(
            [self.trainset[idx][1] for idx in self.indices.local_indices.view(-1)], device=self.device, dtype=self.dtype)

        self.images = self.images.reshape(
            self.indices.individual_batch_cnt, *self.images[0].shape)
        self.targets = self.targets.reshape(
            self.indices.individual_batch_cnt, *self.targets[0].shape)

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class D_CIFAR10(D_VisionData):
    def __init__(self, node_cnt, train_B=16, test_B=64,
                 dset_addr=f'data{os.sep}cifar10-data', d_dataset_format=partitioned_dReal_dset_maker, download=False, device=None, args=None, **kw) -> None:
        cifar10_normalize_transform = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            cifar10_normalize_transform,
        ]
        )
        test_transform = transforms.Compose([
            transforms.ToTensor(), cifar10_normalize_transform
        ]
        )
        super(D_CIFAR10, self).__init__(
            node_cnt, torchvision.datasets.CIFAR10,
            dset_addr, train_transform, test_transform,
            download=download, test_B=test_B,
            d_dataset_format=d_dataset_format, device=device, args=args,
            **kw
        )
        self.figure_size_flatten = 3 * 32 * 32
        self.num_classes = 10


class Dataset_M4(Dataset):
    def __init__(self,
                 input_length,  # num of input steps
                 output_length,  # forecasting horizon
                 freq,  # The frequency of time series
                 train_data_addr="data/M4/train.npy",  # path to numpy data files
                 # for testing mode, we need to load both train and test data
                 test_data_addr="data/M4/test.npy",
                 mode="train",  # train, validation or test
                 expand_dim=False,  # whether expand last dimension
                 seed=0,
                 device=None
                 ):
        self.input_length = input_length
        self.output_length = output_length
        self.mode = mode
        self.expand_dim = expand_dim
        self.device = device
        # Load training set
        self.train_data = np.load(train_data_addr, allow_pickle=True)
        self.data_lsts = self.train_data.item().get(freq)

        # First do global standardization
        self.ts_means, self.ts_stds = [], []
        for i in range(len(self.data_lsts)):
            avg, std = np.mean(self.data_lsts[i]), np.std(self.data_lsts[i])
            self.ts_means.append(avg)
            self.ts_stds.append(std)
            self.data_lsts[i] = (self.data_lsts[i] - avg) / std

        if mode == "test":
            self.test_lsts = np.load(
                test_data_addr, allow_pickle=True).item().get(freq)
            for i in range(len(self.test_lsts)):
                self.test_lsts[i] = (self.test_lsts[i] -
                                     self.ts_means[i])/self.ts_stds[i]
            self.ts_indices = [i for i in range(len(self.test_lsts))]

        elif mode == "train" or "valid":
            # shuffle slices before split
            self.ts_indices = [(i, j) for i in range(len(self.data_lsts))
                               for j in range(0, len(self.data_lsts[i]) - input_length - output_length, 3)]
            np.random.RandomState(0).shuffle(self.ts_indices)

            # 80%-20% train-validation split
            if mode == "train":
                self.ts_indices = self.ts_indices[:int(
                    len(self.ts_indices)*0.9)]
            elif mode == "valid":
                self.ts_indices = self.ts_indices[int(
                    len(self.ts_indices)*0.9):]
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.ts_indices)

    def __getitem__(self, index):
        if self.mode == "test":
            x = self.data_lsts[index][-self.input_length:]
            y = self.test_lsts[index]
        else:
            i, j = self.ts_indices[index]
            x = self.data_lsts[i][j:j+self.input_length]
            y = self.data_lsts[i][j+self.input_length: j +
                                  self.input_length+self.output_length]

        if self.expand_dim:
            return torch.from_numpy(x).float().unsqueeze(-1).to(self.device),  torch.from_numpy(y).float().unsqueeze(-1).to(self.device)
        return torch.from_numpy(x).float().to(self.device), torch.from_numpy(y).float().to(self.device)


class C_M4_Dataset(Dataset):
    def __init__(self, args, node_cnt, microbatch, input_length, output_length, freq, device=None, d_dataset_format=partitioned_dReal_dset_maker) -> None:
        self.train_dataset = Dataset_M4(input_length=input_length, output_length=output_length,
                                        freq=freq, mode="train", expand_dim=False, device=device)
        self.val_dataset = Dataset_M4(input_length=input_length, output_length=output_length,
                                      freq=freq, mode="valid", expand_dim=False, device=device)
        self.test_dataset = Dataset_M4(
            input_length=input_length, output_length=13, freq=freq, mode="test", expand_dim=False, device=device)
        self.args = args
        self.microbatch = microbatch

        self.indices = d_dataset_format(
            self.train_dataset, node_cnt, args, device)
        self.train_loader_eval = DataLoader(
            self.train_dataset, batch_size=1024, shuffle=False)
        self.valid_loader = DataLoader(
            self.val_dataset, batch_size=1024, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1024, shuffle=False)
        self.device = device

    def __len__(self):
        return (self.indices.individual_batch_cnt // self.microbatch) * self.microbatch

    def __getitem__(self, idx):
        if type(idx) == int or (isinstance(idx, torch.Tensor) and idx.dim() == 0):
            mapped_idx = self.indices.local_indices[idx]
            inps, tgts = self.train_dataset[mapped_idx]
            inps = inps.to(self.device).unsqueeze(0)
            tgts = tgts.to(self.device).unsqueeze(0)
        elif isinstance(idx, torch.Tensor) and idx.dim() == 1:
            inps, tgts = [], []
            for i in self.indices.local_indices[idx]:
                inp, tgt = self.train_dataset[i]
                inps.append(inp)
                tgts.append(tgt)
            inps = torch.vstack(inps).to(self.device)
            tgts = torch.vstack(tgts).to(self.device)
        else:
            raise NotImplementedError()
        if inps.dim() > 2:
            return inps, tgts
        else:
            return inps.unsqueeze(-1), tgts.unsqueeze(-1)


def last_even_num(N): return N if N % 2 == 0 else N - 1


class C_M4_New_Dataset(Dataset):
    def __init__(self, args, node_cnt, microbatch, input_length, output_length, freq, device=None) -> None:
        self.train_dataset = Dataset_M4(input_length=input_length, output_length=output_length,
                                        freq=freq, mode="train", expand_dim=False, device=device)
        self.val_dataset = Dataset_M4(input_length=input_length, output_length=output_length,
                                      freq=freq, mode="valid", expand_dim=False, device=device)
        self.test_dataset = Dataset_M4(
            input_length=input_length, output_length=13, freq=freq, mode="test", expand_dim=False, device=device)
        self.args = args
        self.microbatch = microbatch

        B = node_cnt * microbatch
        N = last_even_num(len(self.train_dataset) // B) * B
        self.indices = torch.arange(N, device=device, dtype=torch.int64).reshape(
            node_cnt, N // B, microbatch)[self.args.rank]

        self.train_loader_eval = DataLoader(
            self.train_dataset, batch_size=1024, shuffle=False)
        self.valid_loader = DataLoader(
            self.val_dataset, batch_size=1024, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1024, shuffle=False)
        self.device = device

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if type(idx) == int or (isinstance(idx, torch.Tensor) and idx.dim() == 0):
            inps, tgts = [], []
            for i in self.indices[idx]:
                inp, tgt = self.train_dataset[i]
                inps.append(inp)
                tgts.append(tgt)
            inps = torch.vstack(inps).to(self.device)
            tgts = torch.vstack(tgts).to(self.device)
        else:
            raise NotImplementedError()
        if inps.dim() > 2:
            return inps, tgts
        else:
            return inps.unsqueeze(-1), tgts.unsqueeze(-1)


class C_M4_New_Simulated_Dataset(Dataset):
    def __init__(self, args, node_cnt, microbatch, input_length, output_length, freq, device=None) -> None:
        self.train_dataset = Dataset_M4(input_length=input_length, output_length=output_length,
                                        freq=freq, mode="train", expand_dim=False, device=device)
        self.val_dataset = Dataset_M4(input_length=input_length, output_length=output_length,
                                      freq=freq, mode="valid", expand_dim=False, device=device)
        self.test_dataset = Dataset_M4(
            input_length=input_length, output_length=13, freq=freq, mode="test", expand_dim=False, device=device)
        self.args = args
        self.microbatch = microbatch
        self.node_cnt = node_cnt

        B = node_cnt * microbatch
        N = last_even_num(len(self.train_dataset) // B) * B
        self.indices = torch.arange(
            N, device=device, dtype=torch.int64).reshape(node_cnt, N // node_cnt)

        self.train_loader_eval = DataLoader(
            self.train_dataset, batch_size=1024, shuffle=False)
        self.valid_loader = DataLoader(
            self.val_dataset, batch_size=1024, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1024, shuffle=False)
        self.device = device

    def __len__(self):
        return self.indices.shape[1]

    def __getitem__(self, batch):
        inps, tgts = [], []
        for i in self.indices[torch.arange(self.node_cnt, device=self.device), batch]:
            inp, tgt = self.train_dataset[i]
            inps.append(inp)
            tgts.append(tgt)
        inps = torch.vstack(inps).to(self.device)
        tgts = torch.vstack(tgts).to(self.device)
        if inps.dim() > 2:
            return inps, tgts
        else:
            return inps.unsqueeze(-1), tgts.unsqueeze(-1)
