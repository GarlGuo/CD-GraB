import torch
from torch.utils.data import *
from datasets import load_dataset, load_metric
from d_data import *
from torch.utils.data import DataLoader
from d_eventTimer import EventTimer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PretrainedConfig,
)
from torch.utils.data import DataLoader
from d_model import *
from d_utils import *
from tqdm.auto import tqdm
from d_algo import flatten_grad, D_Sorter



def get_glue_config(args, seed=0):
    raw_datasets = load_dataset("glue", args.task_name)
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1
    seed_everything(seed)
    config = AutoConfig.from_pretrained(
        "bert-base-uncased", num_labels=num_labels, finetuning_task=args.task_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        config=config,
    )
    is_regression = args.task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(
            num_labels=num_labels).label2id
        and args.task_name is not None
        and not is_regression
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
            id: label for label, id in config.label2id.items()}
    elif args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}
    del model
    return {
        "task_name": args.task_name,
        "label_to_id": label_to_id,
        "label_list": label_list,
        "raw_datasets": raw_datasets,
        "is_regression": is_regression,
        "num_labels": num_labels
    }


def d_bert_train(glue_data: D_GLUE_Embeddings,
                     optimizer: torch.optim.Optimizer,
                     c_bert: D_Model,
                     sorter: D_Sorter, 
                     epoch: int, 
                     counter, 
                     args, 
                     eventTimer: EventTimer, 
                     grad_acc: int):
    c_bert.eval()  # disable dropout
    with eventTimer(f"epoch-{epoch}"):
        with eventTimer("sorter_sort"):
            perm_list = sorter.sort()
    cur_loss = 0
    acc_step = 0
    for batch in range(glue_data.indices.individual_batch_cnt):
        embeddings, labels = glue_data[perm_list[batch]]
        optimizer.zero_grad()
        
        # forward pass
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("forward_pass"):
                loss = c_bert(embeddings, labels=labels)

        # backward pass
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("backward"):
                loss.backward()
        
        # sorter step
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("sorter_step"):
                sorter.step(optimizer, batch)

        # Perform gradient accumulation depending on the current step
        with eventTimer(f"epoch-{epoch}"):
            with eventTimer("SGD_step"):
                c_bert.grad_copy_buffer.binary_op_(
                    [p.grad.data for p in c_bert.parameters() if p.requires_grad], ADD_TO_LEFT)
                acc_step += 1
                if (batch > 0 and batch % grad_acc == 0) or (batch == glue_data.indices.individual_batch_cnt - 1) or (grad_acc == 1):
                    # (reached a minibatch size) or (reached the end and have remainding microbatch) or (no grad_acc)
                    c_bert.grad_copy_buffer.unary_op_(AVERAGE_BY_(acc_step))
                    c_bert.grad_copy_buffer.binary_op_(
                        [p.grad for p in c_bert.parameters() if p.requires_grad], RIGHT_COPY_)
                    optimizer.step()
                    c_bert.grad_copy_buffer.unary_op_(ZERO_)
                    acc_step = 0
                    with eventTimer("communication"):
                        c_bert.communicate_weight_inplace()

        if args.rank == 0:
            counter.update(1)
        cur_loss += loss.detach()

        if batch > 0 and batch % args.log_interval == 0 and args.rank == 0:
            print('| epoch {:3d} | {:5d}/{:5d} batches | loss {:.3f}'.format(epoch, batch,
                  glue_data.indices.individual_batch_cnt, cur_loss.item() / batch))
    return cur_loss / glue_data.indices.individual_batch_cnt


@torch.no_grad()
def evaluate_one_model(model: nn.Module, testloader: DataLoader, exp_config, device=None):
    model.eval()
    metric = load_metric("glue", exp_config["task_name"])
    for embeddings, labels in testloader:
        logits = model(embeddings.to(device=device),
                       eval_model=True, labels=labels.to(device=device))
        predictions = logits.argmax(
            dim=-1) if not exp_config["is_regression"] else logits.squeeze()
        metric.add_batch(
            predictions=predictions,
            references=labels.to(device=device),
        )

    return metric.compute()


@torch.no_grad()
def d_bert_test(testloader: DataLoader, c_bert: D_Model, epoch: int, exp_config, rank, device=None):
    c_bert.eval()
    if rank == 0:
        avg_bert_model = c_bert.model
        global_score = evaluate_one_model(
            avg_bert_model, testloader, exp_config, device=device)
        print(f'| epoch {epoch} | global avg score {global_score}', flush=True)
        return global_score


@torch.no_grad()
def d_full_train_loss(exp_config, trainloader: DataLoader, c_bert: D_Model, rank, device=None):
    c_bert.eval()  # disable dropout
    cur_loss = 0
    metric = load_metric("glue", exp_config["task_name"])
    counter = tqdm(range(len(trainloader)))
    if rank == 0:
        avg_bert_model = c_bert.model
        for embeddings, labels in trainloader:
            counter.update(1)
            loss = avg_bert_model(embeddings.to(device=device), 
                                    labels=labels.to(device=device))
            logits = avg_bert_model(embeddings.to(device=device),
                                 labels=labels.to(device=device), 
                                 eval_model=True)
            cur_loss += loss * len(embeddings)
            predictions = logits.argmax(dim=-1) if not exp_config["is_regression"] else logits.squeeze()
            metric.add_batch(
                predictions=predictions,
                references=labels.to(device=device),
            )

        return cur_loss / len(trainloader.dataset), metric.compute()
    return None, None
