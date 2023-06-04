import argparse
import os
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataloader import WosDataModule
from transformers import AutoConfig, AutoTokenizer
from utils import set_seed
from tqdm import tqdm
from deepspeed.comm import comm
import torch.distributed as dist
import deepspeed

## parser setting
parser = argparse.ArgumentParser()
parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
parser.add_argument("--local_rank", type=int)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--data_dir", type=str, default="/data")
parser.add_argument("--model_name",type=str, default="skt/kogpt2-base-v2",)
parser.add_argument("--max_seq_length", default=768,type=int)
parser.add_argument("--seed", default=42, type=int,)
args = parser.parse_args()

# set seed
set_seed(args.seed)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
            '<eos_a>', '<sos_context>', '<eos_context>']
tokenizer.add_tokens(SPECIAL_TOKENS)

# load dataset
train_filepath = 'train_english.json'
ontology_filepath = 'ontology_english.json'

data_module = WosDataModule(args, tokenizer)
train_data_loader = data_module.get_dataloader(
    train_filepath, ontology_filepath, args.batch_size, seed=args.seed
)