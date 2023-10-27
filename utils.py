import torch


def flatten_grad(optimizer):
    t = []
    for _, param_group in enumerate(optimizer.param_groups):
        for p in param_group['params']:
            if p.grad is not None: t.append(p.grad.data.view(-1))
    return torch.concat(t)

def flatten_params(model):
    t = []
    for _, param in enumerate(model.parameters()):
        if param is not None: t.append(param.data.view(-1))
    return torch.concat(t)