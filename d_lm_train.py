import os
import torch
from torch.utils.data import Dataset
from d_data import *
from d_lm_data import *
from d_eventTimer import EventTimer
from d_algo import *
import torchopt
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
import torch.nn.functional as F
import math


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


class D_LM_Dataset:
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
            self.index = torch.arange(B, device=device).reshape(
                node_cnt, B // node_cnt)[self.args.rank]

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


@torch.no_grad()
def clip_grad_norm_(
        grads, max_norm: float, norm_type: float = 2.0,
        error_if_nonfinite: bool = False, foreach=None) -> torch.Tensor:
    r"""Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)
    first_device = grads[0].device
    grouped_grads = _group_tensors_by_device_and_dtype(
        [[g.detach() for g in grads]])  # type: ignore[assignment]

    if norm_type == torch.inf:
        norms = [g.detach().abs().max().to(first_device) for g in grads]
        total_norm = norms[0] if len(
            norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for ((device, _), [grads]) in grouped_grads.items():
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, norm_type))
            elif foreach:
                raise RuntimeError(
                    f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
            else:
                norms.extend([torch.norm(g, norm_type) for g in grads])

        total_norm = torch.norm(torch.stack(
            [norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-6)
    # Note: multiplying by the clamped coef is redundant when the coef is clamped to 1, but doing so
    # avoids a `if clip_coef < 1:` conditional which can require a CPU <=> device synchronization
    # when the gradients do not reside in CPU memory.
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for ((device, _), [grads]) in grouped_grads.items():
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
            torch._foreach_mul_(grads, clip_coef_clamped.to(
                device))  # type: ignore[call-overload]
        elif foreach:
            raise RuntimeError(
                f'foreach=True was passed, but can\'t use the foreach API on {device.type} tensors')
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in grads:
                g.detach().mul_(clip_coef_clamped_device)

    return total_norm


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def LM_train(cur_rank,
             d_trainset,
             model,
             fmodel,
             params,
             buffers,
             optimizer,
             opt_state,
             sorter,
             counter,
             eventTimer: EventTimer,
             epoch,
             node_cnt,
             microbatch,
             d,
             device=None):

    H = model.init_hidden(microbatch)    
    with eventTimer(f"epoch-{epoch}"):
        with eventTimer("sorter_sort"):
            perm_list = sorter.sort()

    if isinstance(sorter, CD_GraB_SingleGrad):
        with eventTimer(f'epoch-{epoch}'):
            with eventTimer("communication"):
                gathered_grads = torch.empty(node_cnt, d, device=device)

    for batch in range(0, len(d_trainset), 1):
        with eventTimer(f'epoch-{epoch}'):
            with eventTimer("dataset"):
                X, Y = d_trainset[perm_list[batch]]
                H = repackage_hidden(H)
        if isinstance(sorter, D_RR):
            with eventTimer(f'epoch-{epoch}'):
                with eventTimer("forward-backward"):
                    Y_hat, H = fmodel(params, buffers, X, H)
                    loss = F.nll_loss(Y_hat, Y.long())
                    avg_grads = torch.autograd.grad(loss, params)
                    with torch.no_grad():
                        avg_grads = torch.cat([g.view(-1) for g in avg_grads])
                
                with torch.no_grad():                   
                    dist.all_reduce(avg_grads, op=dist.ReduceOp.SUM)
                    avg_grads /= node_cnt

                dist.barrier()

        elif isinstance(sorter, CD_GraB_SingleGrad):
            with eventTimer(f'epoch-{epoch}'):
                with eventTimer("forward-backward"):
                    Y_hat, H = fmodel(params, buffers, X, H)
                    loss = F.nll_loss(Y_hat, Y.long())
                    avg_grads = torch.autograd.grad(loss, params)                
                    with torch.no_grad():
                        avg_grads = torch.cat([g.view(-1) for g in avg_grads])

            with torch.no_grad():
                with eventTimer(f'epoch-{epoch}'):
                    with eventTimer("communication"):
                        dist.all_gather_into_tensor(gathered_grads, avg_grads, async_op=False)
                        avg_grads = gathered_grads.mean(dim=0)

                    with eventTimer("sorter"):
                        sorter.step(gathered_grads, batch)

        else:
            raise NotImplementedError()

        with torch.no_grad():
            with eventTimer(f'epoch-{epoch}'):
                with eventTimer("SGD"):
                    # compute gradient and do SGD step
                    avg_grad_list = []
                    grad_cnt = 0
                    for p in params:
                        avg_grad_list.append(
                            avg_grads[grad_cnt: p.numel() + grad_cnt].view(p.shape))
                        grad_cnt += p.numel()
                    # this line doesn't work on PyTorch 2.2, and I am still trying to figure out why
                    # we can still get a similar performance without gradient clipping
                    # clip_grad_norm_(avg_grad_list, 0.25) #
                    updates, opt_state = optimizer.update(
                        avg_grad_list, opt_state, params=params)
                    torchopt.apply_updates(
                        params, tuple(updates), inplace=True)
            if cur_rank == 0:
                counter.update(1)


@torch.no_grad()
def LM_test(rank, eval_dataset, model, params):
    for i, p in enumerate(model.parameters()):
        p.data.copy_(params[i])
    model.eval()
    total_loss = 0
    hidden = model.init_hidden(eval_dataset.data.shape[-1])
    for i in range(len(eval_dataset)):
        data, targets = eval_dataset[i]
        output, hidden = model(data, hidden)
        hidden = repackage_hidden(hidden)
        total_loss += F.nll_loss(output, targets).item()
    ppl = torch.exp(torch.as_tensor(total_loss / (len(eval_dataset) - 1)))
    avg_loss = total_loss / len(eval_dataset)
    return ppl, avg_loss



def LM_train_single_transformer(
        node_idx_map,
        d_trainset,
        func_compute_sample_grad,
        fmodel,
        params,
        buffers,
        optimizer,
        opt_state,
        sorter,
        counter,
        epoch,
        n,
        m,
        d,
        device=None,
        is_bert=False):

    if isinstance(sorter, CD_GraB_Simulated):
        perm_list = sorter.sort()
    else:
        perm_list = torch.vstack([s.sort() for s in sorter])
    for batch in range(0, m, 1):
        data_dict = d_trainset[torch.vstack([node_idx_map[i, x] for i, x in enumerate(perm_list[:, batch])]).view(-1)]
        for k, v in data_dict.items():
            if v.is_cpu: data_dict[k] = v.cuda()
        if type(sorter) == list and isinstance(sorter[0], RandomShuffle):
            avg_grads = torch.autograd.grad(fmodel(params, buffers, data_dict), params)
        elif isinstance(sorter, CD_GraB_Simulated):
            if is_bert:
                per_sample_grads_to_balance = \
                    func_compute_sample_grad(
                        params,
                        buffers,
                        data_dict['input_ids'],
                        data_dict['token_type_ids'],
                        data_dict['attention_mask'],
                        data_dict['labels'],
                    )
            else:
                per_sample_grads_to_balance = \
                    func_compute_sample_grad(
                        params,
                        buffers,
                        data_dict['input_ids'],
                        data_dict['attention_mask'],
                        data_dict['labels'],
                    )
            with torch.no_grad():
                sorter.step(torch.hstack([g.reshape(g.shape[0], -1) for g in per_sample_grads_to_balance]), batch)
                avg_grads = tuple(g.mean(dim=0)
                                  for g in per_sample_grads_to_balance)
                del per_sample_grads_to_balance
        else:
            raise NotImplementedError()

        with torch.no_grad():
            updates, opt_state = optimizer.update(avg_grads, opt_state, params=params)
            torchopt.apply_updates(params, tuple(updates), inplace=True)
            counter.update(1)
            del updates, avg_grads


@torch.no_grad()
def LM_test_transformer_transformer_library(d_data, model, params, device):
    cur_loss = 0
    test_datasize = len(d_data)
    counter = tqdm(range(test_datasize))
    for i, p in enumerate(model.parameters()):
        p.data.copy_(params[i])
    for index in DataLoader(torch.arange(test_datasize, device=device), batch_size=256):
        counter.update(len(index))
        data_dict = d_data[index]
        if data_dict['input_ids'].is_cpu:
            data_dict = {k : v.cuda() for k, v in data_dict.items()}
        loss = model(data_dict)
        cur_loss += (loss * len(index))
    return cur_loss / test_datasize, math.exp(cur_loss / (test_datasize - 1))
