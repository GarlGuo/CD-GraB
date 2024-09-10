import os
import torch
from torch.utils.data import Dataset
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
