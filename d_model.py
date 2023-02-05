import torch
import torch.distributed as dist
import torch.nn as nn
from collections.abc import Callable
from typing import Dict, List
from d_tensorbuffer import *
import torch.nn.functional as F
from d_utils import seed_everything
from transformers import (
    AutoModelForSequenceClassification, PretrainedConfig, AutoConfig)
from collections import OrderedDict


def sec_norm_sq(tensor) -> torch.Tensor:
    return torch.norm(tensor, 2) ** 2


def fro_norm_sq(tensor) -> torch.Tensor:
    return torch.norm(tensor, 'fro') ** 2


class D_Model(nn.Module):
    def __init__(self, rank, node_cnt, protocol, model_maker: Callable[[], nn.Module]) -> None:
        super(D_Model, self).__init__()
        self.model = model_maker()
        self.model_maker = model_maker
        self.node_cnt = node_cnt
        self.layer_cnt = len(list(self.model.parameters()))
        self.rank = rank
        self.protocol = protocol
        self.weight_buffer = D_TensorBuffer(
            rank, node_cnt, [p for p in self.model.parameters() if p.requires_grad], protocol)
        self.grad_copy_buffer = D_TensorBuffer(rank, node_cnt, [torch.zeros_like(
            p) for p in self.model.parameters() if p.requires_grad], protocol)
        # Stats-collecting fields below
        if rank == 0:
            self.X_tl = None
            self.grad_Xtl = None
            self.stats = {
                # Consensus error per step: epoch t, step l, m number of data, n number of nodes
                "|| X_tl - X_tl * 11^T/n ||_F^2": [],
                # Herding bound: node i, step j; like max{np.cumsum()}
                'max || sum_j sum_i (grad_ij - avg(grad_j)) ||^2': [],
                'train_loss': []
            }

    def parameters(self):
        return self.model.parameters()

    # p * n tensor
    def gather_weights(self):
        Xtl = self.weight_buffer.gather()
        if self.rank == 0:
            self.X_tl = torch.as_tensor(Xtl)

    # p * n tensor
    def gather_grad(self):
        grad_buffer = self.get_grad_communication_buffer()
        grad_Xtl = grad_buffer.gather()
        if self.rank == 0:
            self.grad_Xtl = grad_Xtl

    def communicate_weight_inplace(self):
        self.weight_buffer.communicate_inplace()

    def get_global_averaged_model(self):
        global_avg_model = self.model_maker()
        avg_weight_vector = self.weight_buffer.global_average()
        ptr = 0
        for p in global_avg_model.parameters():
            if not p.requires_grad:
                continue
            length = len(p.data.view(-1))
            p.view(-1).data.copy_(avg_weight_vector[ptr: ptr + length])
            ptr += length
        return global_avg_model

    def eval(self, *args, **kw):
        self.model.eval(*args, **kw)

    def train(self, *args, **kw):
        self.model.train(*args, **kw)

    def forward(self, *args, **kw):
        return self.model(*args, **kw)

    def get_grad_acc_buffer(self):
        return torch.cat([p.grad.view(-1).detach().clone() for p in self.model.parameters() if p.requires_grad])

    def get_grad_communication_buffer(self):
        return D_TensorBuffer(
            self.rank,
            self.node_cnt,
            torch.cat([p.grad.view(-1).detach().clone()
                      for p in self.model.parameters() if p.requires_grad]),
            self.protocol
        )

    # Stats-collecting helper below
    @torch.no_grad()
    def get_Xtl_minus_avg_Xtl_fro_sq(self):  # || X_tl - X_tl * 11^T/n ||_F^2
        diff = self.X_tl - self.X_tl @ (torch.ones(self.node_cnt, self.node_cnt,
                                        device=self.X_tl.device, dtype=self.X_tl.dtype) / self.node_cnt)
        return fro_norm_sq(diff).cpu()

    @torch.no_grad()
    def record_runtime_stats_in_epoch(self):
        self.stats['|| X_tl - X_tl * 11^T/n ||_F^2'].append(
            self.get_Xtl_minus_avg_Xtl_fro_sq())

    @torch.no_grad()
    def record_train_loss(self, loss):
        self.stats['train_loss'].append(loss.data)

    def save_results(self, addr):
        torch.save(self.stats, addr)


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


class LogisticRegression(nn.Module):
    def __init__(self, figure_size_flatten, num_classes, device=None, seed=0) -> None:
        seed_everything(seed)
        super(LogisticRegression, self).__init__()
        self.figure_size_flatten = figure_size_flatten
        self.linear = nn.Linear(
            figure_size_flatten, num_classes, device=device, dtype=torch.float32)

    def forward(self, x):
        return self.linear(x.view(-1, self.figure_size_flatten).to(torch.float32))

    def pred(self, x):
        y_scores = self(x)
        return torch.max(y_scores, dim=1)[1]


class LSTMModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp=32, nhid=32, nlayers=2, dropout=0.2, seed=0, device=None):
        super(LSTMModel, self).__init__()
        seed_everything(seed)
        self.ntoken = ntoken
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, device=device)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, device=device)
        self.decoder = nn.Linear(nhid, ntoken, device=device)
        seed_everything(seed)
        nn.init.uniform_(self.encoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        emb = self.encoder(input)
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.ntoken)
        return F.log_softmax(decoded, dim=1), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))


def BERT_model_maker(args, exp_config, seed, device):
    num_labels = exp_config["num_labels"]
    label_list = exp_config["label_list"]
    is_regression = exp_config["is_regression"]

    seed_everything(seed)
    model_config = AutoConfig.from_pretrained(
        'bert-base-uncased', num_labels=num_labels, finetuning_task=args.task_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased', config=model_config).to(device=device)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(
            num_labels=num_labels).label2id and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: label_name_to_id[label_list[i]] for i in range(num_labels)}

    elif args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in model_config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label for label, id in model_config.label2id.items()}
    for n, p in model.named_parameters():
        if not n.startswith('classifier'):
            p.requires_grad_(False)
    return model.bert, model.classifier, model.config


class BERT_LinearHead(nn.Module):
    def __init__(self, num_labels, device, seed=0) -> None:
        super(BERT_LinearHead, self).__init__()
        seed_everything(seed)
        self.classifier = nn.Linear(768, num_labels).to(device=device)
        self.num_labels = num_labels

    def forward(self, embeddings, labels=None, eval_model=False):
        logits = self.classifier(embeddings)
        if eval_model:
            return logits
        if labels is not None:
            return nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False, axis=1):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.axis = axis
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params:
        self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features))

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=self.axis, keepdim=True)
        self.stdev = torch.sqrt(
            torch.std(x, dim=self.axis, keepdim=True) + self.eps)

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class Auto_MLP(nn.Module):
    def __init__(self,
                 input_dim,  # number of features
                 input_length,  # input length of ts
                 output_dim,  # number of output features
                 num_steps,  # number of prediction steps every forward pass
                 hidden_dim,  # hidden dimension
                 num_layers,  # number of layers
                 use_RevIN=True,  # whether to use reversible normalization
                 seed=0,
                 device=None
                 ):

        super(Auto_MLP, self).__init__()
        seed_everything(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_length = input_length
        self.num_steps = num_steps
        self.use_RevIN = use_RevIN
        if use_RevIN:
            self.normalizer = RevIN(
                num_features=self.output_dim, axis=(1, 2), affine=False)

        model = [nn.Linear(input_length*input_dim,
                           hidden_dim).to(device=device), nn.ReLU()]
        for i in range(num_layers-2):
            model += [nn.Linear(hidden_dim,
                                hidden_dim).to(device=device), nn.ReLU()]
        model += [nn.Linear(hidden_dim, output_dim *
                            num_steps).to(device=device)]

        self.model = nn.Sequential(*model)

    def forward(self, inps, tgts):

        if self.use_RevIN:
            # number of autoregreesive steps given the number of predictions output by the model
            auto_steps = tgts.shape[1]//self.num_steps
            if tgts.shape[1] % self.num_steps > 0:
                auto_steps += 1

            denorm_outs = []
            norm_tgts = []
            norm_outs = []
            for i in range(auto_steps):
                # normalize input ts
                norm_inp = self.normalizer.forward(inps, mode="norm")
                pred = self.model(norm_inp.reshape(norm_inp.shape[0], -1))
                pred = pred.reshape(
                    inps.shape[0], self.num_steps, self.output_dim)
                norm_outs.append(pred)

                # normalize tgts
                norm_tgts.append(self.normalizer._normalize(
                    tgts[:, i*self.num_steps: (i+1)*self.num_steps]))

                # denormalize prediction and add back to the input
                denorm_outs.append(
                    self.normalizer.forward(pred, mode="denorm"))
                # print(inps.shape, denorm_outs[-1].shape)
                inps = torch.cat(
                    [inps[:, self.num_steps:], denorm_outs[-1]], dim=1)

            norm_outs = torch.cat(norm_outs, dim=1)
            norm_tgts = torch.cat(norm_tgts, dim=1)
            denorm_outs = torch.cat(denorm_outs, dim=1)

            return denorm_outs[:, :norm_tgts.shape[1]], norm_outs[:, :norm_tgts.shape[1]], norm_tgts
        else:
            # number of autoregreesive steps given the number of predictions output by the model
            auto_steps = tgts.shape[1]//self.num_steps
            if tgts.shape[1] % self.num_steps > 0:
                auto_steps += 1
            outs = []
            for i in range(auto_steps):
                pred = self.model(inps.reshape(inps.shape[0], -1))
                pred = pred.reshape(
                    inps.shape[0], self.num_steps, self.output_dim)
                outs.append(pred)
                inps = torch.cat([inps[:, self.num_steps:], outs[-1]], dim=1)

            outs = torch.cat(outs, dim=1)
            return outs, outs, tgts
