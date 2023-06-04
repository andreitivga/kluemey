import argparse
import os
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataloader import WosDataModule
from transformers import AutoConfig, AutoTokenizer
from utils import set_seed
from tqdm import tqdm

## parser setting
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--data_dir", type=str, default="/data")
parser.add_argument("--model_name", type=str, default="gpt2")
parser.add_argument("--max_seq_length", default=768, type=int)
parser.add_argument("--seed", default=42, type=int)
args = parser.parse_args()

# set seed
set_seed(args.seed)

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2", bos_token='</s>', eos_token='</s>', unk_token='<unk>',
  pad_token='<pad>', mask_token='<mask>')
SPECIAL_TOKENS = ['<sos_u>', '<sos_r>', '<sos_b>', '<sos_a>', '<eos_u>', '<eos_r>', '<eos_b>', 
            '<eos_a>', '<sos_context>', '<eos_context>']
tokenizer.add_tokens(SPECIAL_TOKENS)

# load dataset
train_filepath = 'banking_data.json'
# dev_filepath = 'data/wos-v1.1/wos_dev.json'
ontology_filepath = 'data/wos-v1.1/ontology.json'

data_module = WosDataModule(args, tokenizer)
train_data_loader = data_module.get_dataloader(
    train_filepath, ontology_filepath, args.batch_size, seed=args.seed
)
# dev_data_loader = data_module.get_dataloader(
#     dev_filepath, ontology_filepath, args.batch_size, seed=args.seed
# )
args.processor = data_module.processor

# load model
model = AutoModelForCausalLM.from_pretrained(args.model_name)
model.resize_token_embeddings(len(tokenizer)) 
model.cuda()

# model train
epochs = 25
for epoch in range(epochs):
    print('Epoca')
    for batch in tqdm(train_data_loader):
        print('batch')
        model.train()
        model.zero_grad()
        train_input_ids, train_input_masks, train_target_ids = [b for b in batch[:-1]]
        output = model(
            input_ids=train_input_ids.cuda(),
            attention_mask=train_input_masks.cuda(),
            labels=train_input_ids.cuda(),
        )
        loss = output.loss
        loss.backward()

    # wandb logging
    print("loss", loss.item())
    print("epoch", epoch+1)

    # # model eval step
    # with torch.no_grad():
    #     model.eval()
    #     for batch in tqdm(dev_data_loader):
    #         dev_input_ids, dev_input_masks, dev_target_ids = [b for b in batch[:-1]]
    #         eval_out = model(
    #             input_ids=dev_input_ids.cuda(),
    #             attention_mask=dev_input_masks.cuda(),
    #             labels=dev_input_ids.cuda()
    #         )
    #         eval_loss = eval_out.loss

    # model save
    ckpt_dir = f"model_save/{args.model_name.replace('/', '-')}_split-{epoch}-final"
    model.save_pretrained(ckpt_dir)