import os
import torch
from torch.utils.data import Dataset
from dReal_data import *


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, train_path, valid_path, test_path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(train_path)
        self.valid = self.tokenize(valid_path)
        self.test = self.tokenize(test_path)
    
    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids, dtype=torch.int64))
            ids = torch.cat(idss)

        return ids


def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data[:nbatch * bsz]
    data = data.view(bsz, -1).t().contiguous()
    return data


class LMDataset(Dataset):
    def __init__(self, args, data: torch.Tensor, device=None) -> None:
        super().__init__()
        self.args = args
        self.data = data
        self.device = device

    def __getitem__(self, i):
        i = i * self.args.bptt
        seq_len = min(self.args.bptt, self.data.shape[0] - 1 - i)
        data = self.data[i:i + seq_len]
        target = self.data[i + 1:i + 1 + seq_len]
        return data.to(self.device), target.view(-1).to(self.device)

    def __len__(self):
        return (self.data.shape[0] - 1) // self.args.bptt


# class DReal_LM_Dataset:
#     def __init__(self, args, node_cnt, B, dir_addr:str, d_dataset_format=partitioned_dReal_dset_maker, device=None, **kw) -> None:
#         self.device = device
#         self.args = args
#         self.B = B
#         train_path = os.path.join(dir_addr, 'train.txt')
#         valid_path = os.path.join(dir_addr, 'valid.txt')
#         test_path = os.path.join(dir_addr, 'test.txt')        

#         self.corpus = Corpus(train_path, valid_path, test_path)
#         self.ntokens = len(self.corpus.dictionary)

#         self.node_cnt = node_cnt
#         self.trainset = LMDataset(args, batchify(self.corpus.train, B), device=self.device)

#         self.trainset_eval = LMDataset(args, batchify(self.corpus.train, B), device=self.device)  
#         self.val_dataset = LMDataset(args, batchify(self.corpus.valid, B), device=self.device)
#         self.test_dataset = LMDataset(args, batchify(self.corpus.test, B), device=self.device)

#         self.microbatch = microbatch
#         self.indices = d_dataset_format(self.trainset, node_cnt, args=args, device=device)

#     def __len__(self):
#         return (self.indices.individual_batch_cnt // self.microbatch) * self.microbatch

#     def __getitem__(self, idx):
#         if type(idx) == int or (isinstance(idx, torch.Tensor) and idx.dim() == 0):
#             mapped_idx = self.indices.local_indices[idx]
#             X, Y = self.trainset[mapped_idx]
#             return X.unsqueeze(-1), Y.unsqueeze(-1)
#         elif isinstance(idx, torch.Tensor) and idx.dim() == 1:
#             mapped_idx = self.indices.local_indices[idx]
#             X, Y = [], []
#             for i in mapped_idx:
#                 x, y = self.trainset[i // self.B]
#                 X.append(x[:, i % self.B])
#                 Y.append(y.view(x.shape)[:, i % self.B])
#             return torch.vstack(X).T, torch.cat(Y)
#         else:
#             raise NotImplementedError(idx)


class DReal_LM_Dataset: 
    def __init__(self, args, node_cnt, B, dir_addr: str, device=None, **kw) -> None:
        self.device = device
        self.args = args
        self.B = B
        self.microbatch = B // node_cnt
        train_path = os.path.join(dir_addr, 'train.txt')
        valid_path = os.path.join(dir_addr, 'valid.txt')
        test_path = os.path.join(dir_addr, 'test.txt')

        self.corpus = Corpus(train_path, valid_path, test_path)
        self.ntokens = len(self.corpus.dictionary)

        self.node_cnt = node_cnt
        self.trainset = LMDataset(args, batchify(
            self.corpus.train, B), device=self.device)

        self.trainset_eval = LMDataset(args, batchify(
            self.corpus.train, B), device=self.device)
        self.val_dataset = LMDataset(args, batchify(
            self.corpus.valid, B), device=self.device)
        self.test_dataset = LMDataset(args, batchify(
            self.corpus.test, B), device=self.device)
        
        if node_cnt == B:
            self.index = self.args.rank
        else:
            assert B % node_cnt == 0
            self.index = torch.arange(B, device=device).reshape(node_cnt, B // node_cnt)[self.args.rank]            

    def __len__(self):
        return (len(self.trainset) // 2 * 2)

    def __getitem__(self, idx):
        if type(idx) == int or (isinstance(idx, torch.Tensor) and idx.dim() == 0):
            X, Y = self.trainset[idx]
            Y = Y.view(X.shape)
            X, Y = X[:, self.index], Y[:, self.index].flatten()
            if X.dim() == 1:
                return X.unsqueeze(-1), Y
            else:
                return X, Y
        elif isinstance(idx, torch.Tensor) and idx.dim() == 1:
            X, Y = [], []
            for i in idx:
                x, y = self.trainset[i]
                y = y.view(x.shape)
                X.append(x[:, self.index])
                Y.append(y[:, self.index])
            return torch.stack(X, dim=-1), torch.cat(Y)
        else:
            raise NotImplementedError(idx)
