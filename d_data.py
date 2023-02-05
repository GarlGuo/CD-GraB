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
from transformers import AutoTokenizer, default_data_collator
from tqdm.auto import tqdm


def last_even_num(odd_or_even_num):
    if odd_or_even_num % 2 == 0:
        return odd_or_even_num
    else:
        return odd_or_even_num - 1


class D_Dataset_Indices:
    def __init__(self, microbatch: int, node_cnt: int, node_idx_map, args=None) -> None:
        super().__init__()
        self.B = microbatch
        self.node_cnt = node_cnt
        self.individual_batch_cnt = node_idx_map.shape[1]
        if microbatch == 1:
            self.local_indices = node_idx_map[args.rank].flatten()
        else:
            self.local_indices = node_idx_map[args.rank]


class D_Dataset_Partitioned(D_Dataset_Indices):
    def __init__(self, dataset, microbatch, node_cnt, args) -> None:
        # must shuffle before dividing
        # shuffle, split
        # shuffle before reshape, instead of simply calling, arange => random.shuffle => reshape
        data_idx = np.arange(len(dataset))
        random.shuffle(data_idx)
        total_batch_cnt = last_even_num(len(dataset) // (microbatch * node_cnt)) * node_cnt
        target_len = total_batch_cnt * microbatch
        node_idx_map = torch.tensor(data_idx[:target_len], dtype=torch.int64).reshape(
            (node_cnt, total_batch_cnt // node_cnt, microbatch))
        super().__init__(microbatch, node_cnt, node_idx_map, args)


def partitioned_dset_maker(
    dset, B, nodes, args): return D_Dataset_Partitioned(dset, B, nodes, args)


class D_Dataset_Plain(D_Dataset_Indices):
    def __init__(self, dataset, microbatch, node_cnt, args) -> None:
        total_datapoint_cnt = last_even_num(len(dataset) // microbatch) * microbatch
        node_idx_map = torch.stack(
            [torch.arange(total_datapoint_cnt).reshape((len(dataset) // microbatch), microbatch)] * node_cnt)
        super().__init__(microbatch, node_cnt, node_idx_map, args)


def plain_dset_maker(dset, microbatch, nodes, args): return D_Dataset_Plain(dset, microbatch, nodes, args)


class D_VisionData(Dataset):
    def __init__(self, node_cnt, dset_maker: VisionDataset,
                 dset_addr, train_transform, test_transform, d_dataset_format=partitioned_dset_maker,
                 download=False, train_B=16, test_B=128, device=None, dtype=torch.float32, args=None, **kw) -> None:
        super().__init__()
        self.node_cnt = node_cnt
        self.trainset: VisionDataset = dset_maker(
            root=dset_addr, train=True, download=download, transform=train_transform)
        self.testset: VisionDataset = dset_maker(
            root=dset_addr, train=False, download=download, transform=test_transform)

        self.indices: D_Dataset_Indices = d_dataset_format(
            self.trainset, train_B, self.node_cnt, args=args, **kw)

        # print(f'{args.rank}: {self.indices.local_indices}')
        self.device = device
        self.dtype = dtype

        self.trainloader = DataLoader(self.trainset, batch_size=test_B)
        self.testloader = DataLoader(self.testset, batch_size=test_B)

        self.images = torch.stack([self.trainset[idx][0] for idx in self.indices.local_indices.view(-1)]).to(
            device=self.device, dtype=self.dtype)
        self.targets = torch.tensor(
            [self.trainset[idx][1] for idx in self.indices.local_indices.view(-1)], device=self.device, dtype=self.dtype)

        self.images = self.images.reshape(
            self.indices.individual_batch_cnt, train_B, *self.images[0].shape)
        self.targets = self.targets.reshape(
            self.indices.individual_batch_cnt, train_B, *self.targets[0].shape)

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


class D_CIFAR10(D_VisionData):
    def __init__(self, node_cnt, train_B=16, test_B=64,
                 dset_addr=f'data{os.sep}cifar10-data', d_dataset_format=partitioned_dset_maker, download=False, device=None, args=None, **kw) -> None:
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
            download=download, train_B=train_B, test_B=test_B,
            d_dataset_format=d_dataset_format, device=device, args=args,
            **kw
        )
        self.figure_size_flatten = 3 * 32 * 32
        self.num_classes = 10


class GLUE:
    glue_task_to_keys = {
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

    def __init__(self, args, exp_config) -> None:
        super().__init__()

        self.raw_datasets = exp_config["raw_datasets"]
        sentence1_key, sentence2_key = self.glue_task_to_keys[args.task_name]

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        padding = "max_length"

        label_to_id = exp_config["label_to_id"]

        def preprocess_function(examples):
            # Tokenize the texts
            texts = (
                (examples[sentence1_key],) if sentence2_key is None else (
                    examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*texts, padding=padding,
                               max_length=128, truncation=True)

            if "label" in examples:
                if label_to_id is not None:
                    # Map labels to IDs (not necessary for GLUE tasks)
                    result["labels"] = [label_to_id[l]
                                        for l in examples["label"]]
                else:
                    # In all cases, rename the column to labels because the model will expect that.
                    result["labels"] = examples["label"]
            return result

        processed_datasets = self.raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=self.raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

        self.trainset = processed_datasets["train"]
        self.testset = processed_datasets["validation_matched" if args.task_name ==
                                          "mnli" else "validation"]
        # DataLoaders creation:
        self.data_collator = default_data_collator


class D_GLUE:
    def __init__(self, args, exp_config, node_cnt, device=None, d_dataset_format=partitioned_dset_maker, **kw) -> None:
        self.node_cnt = node_cnt
        self.device = device

        self.glue = GLUE(args, exp_config)

        self.trainset = list(DataLoader(
            self.glue.trainset, shuffle=False, collate_fn=self.glue.data_collator, batch_size=args.train_B))
        self.indices: D_Dataset_Indices = d_dataset_format(
            self.trainset, args.train_B, self.node_cnt, args=args, **kw)

        self.trainloader = DataLoader(
            self.glue.trainset, shuffle=False, collate_fn=self.glue.data_collator, batch_size=args.test_B)
        self.testloader = DataLoader(
            self.glue.testset, shuffle=False, collate_fn=self.glue.data_collator, batch_size=args.test_B)

        self.node_cnt = node_cnt
        self.device = device

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, index):
        return self.to_device(dict(self.trainset[self.indices.local_indices[index]]))

    def to_device(self, data):
        new_data = dict()
        for k, v in data.items():
            new_data[k] = v.to(self.device)
        return new_data


class D_GLUE_Embeddings:
    @torch.no_grad()
    def __init__(self, args, exp_config, node_cnt, model, device=None, d_dataset_format=partitioned_dset_maker, **kw) -> None:
        self.node_cnt = node_cnt
        self.device = device
        self.glue = GLUE(args, exp_config)
        train_embeddings_addr = f'data{os.sep}glue-data{os.sep}{args.task_name}-bertbase-train-embedding.pt'
        test_embeddings_addr = f'data{os.sep}glue-data{os.sep}{args.task_name}-bertbase-test-embedding.pt'

        if not (os.path.exists(train_embeddings_addr) and os.path.exists(test_embeddings_addr)):
            model = model()
            model.eval()
            trainset_raw = list(DataLoader(
                self.glue.trainset, shuffle=False, collate_fn=self.glue.data_collator, batch_size=128))
            testset_raw = list(DataLoader(
                self.glue.testset, shuffle=False, collate_fn=self.glue.data_collator, batch_size=32))
            self.trainset_embeddings = []
            for batch in tqdm(trainset_raw):
                embeddings = model(
                    input_ids=batch['input_ids'].to(device=device),
                    token_type_ids=batch['token_type_ids'].to(device=device),
                    attention_mask=batch['attention_mask'].to(device=device),
                )[1]
                self.trainset_embeddings.append(embeddings)
            self.trainset_embeddings = torch.vstack(self.trainset_embeddings)
            torch.save(self.trainset_embeddings, train_embeddings_addr)
            self.testset_embeddings = []
            for batch in tqdm(testset_raw):
                embeddings = model(
                    input_ids=batch['input_ids'].to(device=device),
                    token_type_ids=batch['token_type_ids'].to(device=device),
                    attention_mask=batch['attention_mask'].to(device=device),
                )[1]
                self.testset_embeddings.append(embeddings)
            self.testset_embeddings = torch.vstack(self.testset_embeddings)
            torch.save(self.testset_embeddings, test_embeddings_addr)
        else:
            self.trainset_embeddings = torch.load(
                train_embeddings_addr, map_location=torch.device('cpu'))
            self.testset_embeddings = torch.load(
                test_embeddings_addr, map_location=torch.device('cpu'))

        self.trainset_labels = torch.tensor(
            [self.glue.trainset[i]['labels']
                for i in range(len(self.trainset_embeddings))],
            device=self.device
        )
        self.testset_labels = torch.tensor(
            [self.glue.testset[i]['labels']
                for i in range(len(self.testset_embeddings))],
            device=self.device
        )
        self.trainset = [(self.trainset_embeddings[i], self.trainset_labels[i])
                         for i in range(len(self.trainset_embeddings))]
        self.testset = [(self.testset_embeddings[i], self.testset_labels[i])
                        for i in range(len(self.testset_embeddings))]
        self.indices: D_Dataset_Indices = d_dataset_format(
            self.trainset_embeddings, args.train_B, self.node_cnt, args=args, **kw)
        self.trainloader = DataLoader(self.trainset, batch_size=args.test_B)
        self.testloader = DataLoader(self.testset, batch_size=args.test_B)

        self.node_cnt = node_cnt
        self.device = device

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, index):
        embeddings = self.trainset_embeddings[self.indices.local_indices[index]].to(
            device=self.device)
        labels = self.trainset_labels[self.indices.local_indices[index]].to(
            device=self.device)
        return embeddings, labels

