#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import math
from algo import *
import os
from itertools import chain

import datasets
import torch
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchopt
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
)
import sys
import huggingface_pt
import functorch
from functorch import make_functional_with_buffers, grad
from d_lm_train import *
import warnings

warnings.filterwarnings('ignore')

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


parser = argparse.ArgumentParser(
    description="Finetune a transformers model on a causal language modeling task")
parser.add_argument(
    "--dataset_name",
    type=str,
    default='wikitext',
    help="The name of the dataset to use (via the datasets library).",
)
parser.add_argument(
    "--dataset_config_name",
    type=str,
    default='wikitext-103-raw-v1',
    help="The configuration name of the dataset to use (via the datasets library).",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    default='hf-internal-testing/tiny-random-gpt2'
)
parser.add_argument(
    "--B",
    type=int,
    default=64,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--lr",
    type=float,
    default=5e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0,
)
parser.add_argument("--epochs", type=int, default=50,
                    help="Total number of training epochs to perform.")
parser.add_argument("--seed", type=int, default=0,
                    help="A seed for reproducible training.")
parser.add_argument(
    "--model_type",
    type=str,
    default=None,
    help="Model type to use if training from scratch.",
    choices=MODEL_TYPES,
)
parser.add_argument(
    "--node_cnt",
    type=int,
    default=64,
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=128,
)
parser.add_argument(
    "--sorter",
    type=str,
    default="CD-GraB",
    choices=[
        "CD-GraB",
        "D-RR",
    ]
)
args = parser.parse_args()
args.node_cnt = args.B

exp_details = f"sorter-{args.sorter}-lr-{args.lr}-B-{args.B}-seed-{args.seed}"

set_seed(args.seed)
raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

config = AutoConfig.from_pretrained('gpt2')
tokenizer = AutoTokenizer.from_pretrained('gpt2')

model_name = args.model_name_or_path
block_size = args.max_seq_length

# Preprocessing the datasets.
# First we tokenize all the texts.
column_names = raw_datasets["train"].column_names
text_column_name = "text" if "text" in column_names else column_names[0]

def tokenize_function(examples):
    return tokenizer(examples[text_column_name], max_length=128, truncation=True)

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True,
    num_proc=32,
    remove_columns=column_names,
    desc="Running tokenizer on dataset",
)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i: i + block_size]
            for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
# for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
# to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    num_proc=32,
    desc=f"Grouping texts in chunks of {block_size}",
)

train_dataset = lm_datasets["train"]
eval_dataset = lm_datasets["validation"]


train_dataset = train_dataset.with_format('torch')
eval_dataset = eval_dataset.with_format('torch')


config.n_embd = 128
config.n_ctx = 128
config.n_layer = 2
config.n_head = 2
config.n_positions = 128
config.summary_first_dropout = 0
config.attn_pdrop = 0
config.resid_pdrop = 0
model = huggingface_pt.GPT2LMHeadModel(config).cuda()
model.eval()

fmodel, params, buffers = make_functional_with_buffers(model)

def compute_loss_stateless_model(params, buffers, input_ids, attention_mask, labels):
    data_dict = {
        'input_ids': input_ids.view(1, *input_ids.shape),
        'attention_mask': attention_mask.view(1, *attention_mask.shape),
        'labels': labels.view(1, *labels.shape),
    }
    return fmodel(params, buffers, data_dict)

def last_even_num(n): return n if n % 2 == 0 else n - 1

ft_compute_sample_grad = torch.vmap(functorch.grad(compute_loss_stateless_model), in_dims=(None, None, 0, 0, 0))

epochs = args.epochs

n = args.node_cnt
B = args.B
N = len(train_dataset)
d = sum(p.numel() for p in model.parameters() if p.requires_grad)
device = torch.device('cuda')

m = last_even_num(N // n)
N = m * n

node_idx_map = torch.arange(N).reshape(n, m).cuda()

max_train_steps = int(math.ceil(N / B) * epochs)

no_decay = ["bias", "LayerNorm.weight"]
wd_mask_list = [any(nd in n for nd in no_decay) for n, p in model.named_parameters() if p.requires_grad]


optimizer = torchopt.adamw(lr=args.lr, weight_decay=args.weight_decay, mask=(lambda params: wd_mask_list))
opt_state = optimizer.init(params)

counter = tqdm(range(m * args.epochs))

if args.sorter == 'CD-GraB':
    sorter = CD_GraB_Simulated(args, n=n, m=m, d=d, device=device)
elif args.sorter == 'D-RR':
    sorter = [RandomShuffle(m, device=device) for _ in range(n)]
else:
    raise NotImplementedError()


results = {
    'train': {'ppl': [], 'loss': []},
    'test': {'ppl': [], 'loss': []},
}
for e in range(1, args.epochs + 1):
    LM_train_single_transformer(
        node_idx_map,
        train_dataset,
        ft_compute_sample_grad,
        fmodel,
        params,
        buffers,
        optimizer,
        opt_state,
        sorter,
        counter,
        e,
        n,
        m,
        d,
        device=device,
        is_bert=False
    )

    full_train_loss, full_train_ppl = LM_test_transformer_transformer_library(train_dataset, model, params, device=device)
    print(f'| epoch {e} | full train loss {full_train_loss:.4f} |', flush=True)
    print(f'| epoch {e} | train ppl {full_train_ppl:.4f} |', flush=True)

    test_loss, test_ppl = LM_test_transformer_transformer_library(eval_dataset, model, params, device=device)
    print(f'| epoch {e} | test ppl {test_ppl} |', flush=True)

    results['train']['loss'].append(full_train_loss)
    results['train']['ppl'].append(full_train_ppl)

    results['test']['loss'].append(test_loss)
    results['test']['ppl'].append(test_ppl)

    torch.save((model, sorter), f'model{os.sep}gpt2-wiki103{os.sep}{exp_details}-epoch-{e}.pt')
    torch.save(results, f"{exp_folder}{os.sep}results-{e}.pt")

torch.save(results, f"{exp_folder}{os.sep}results.pt")
torch.save((model, sorter),
           f'model{os.sep}gpt2-wiki103{os.sep}{exp_details}-final.pt')
