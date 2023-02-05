import torch.nn as nn
import torch.nn.functional as F
from d_model import *
import time
from d_utils import *


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp=32, nhid=32, nlayers=2, dropout=0.2, tie_weights=False, seed=0, device=None):
        super(LSTMModel, self).__init__()
        if seed is not None: 
            seed_everything(seed)
            self.deterministic = True
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout, device=device)
        self.encoder = nn.Embedding(ntoken, ninp, device=device)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, device=device)
        self.decoder = nn.Linear(nhid, ntoken, device=device)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        nn.init.uniform_(self.encoder.weight, -0.1, 0.1, device=device)
        nn.init.zeros_(self.decoder.bias, device=device)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1, device=device)

        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        emb = self.encoder(input)
        # emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))
