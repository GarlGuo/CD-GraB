import os
import torch
import torch.utils.data
from d_data import *


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
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, args, data: torch.Tensor, device=None) -> None:
        super().__init__()
        self.args = args
        self.data = data
        self.device = device

    def __getitem__(self, i):
        if i >= len(self): raise IndexError(f'index {i} out of range')
        i = i * self.args.bptt
        seq_len = min(self.args.bptt, self.data.shape[0] - 1 - i)
        data = self.data[i:i + seq_len]
        target = self.data[i + 1:i + 1 + seq_len]
        return data.to(self.device), target.view(-1).to(self.device)

    def __len__(self):
        return (self.data.shape[0] // self.args.bptt)


class D_LM_Dataset:
    def __init__(self, args, node_cnt, dir_addr:str, d_dataset_format=partitioned_dset_maker, device=None, **kw) -> None:
        self.device = device
        train_path = os.path.join(dir_addr, 'train.txt')
        valid_path = os.path.join(dir_addr, 'valid.txt')
        test_path = os.path.join(dir_addr, 'test.txt')        

        self.corpus = Corpus(train_path, valid_path, test_path)
        self.ntokens = len(self.corpus.dictionary)

        self.node_cnt = node_cnt
        self.trainset_eval = LMDataset(args, batchify(self.corpus.train, args.test_B), device=self.device)  
        self.val_dataset = LMDataset(args, batchify(self.corpus.valid, args.test_B), device=self.device)
        self.test_dataset = LMDataset(args, batchify(self.corpus.test, args.test_B), device=self.device)

        self.trainset = LMDataset(args, batchify(self.corpus.train, 1).to(self.device))
        self.indices = d_dataset_format(self.trainset, 1, node_cnt, args=args)

    def __len__(self):
        return self.indices.individual_batch_cnt

    def __getitem__(self, idx):
        mapped_idx = self.indices.local_indices[idx]
        return self.trainset[mapped_idx]
