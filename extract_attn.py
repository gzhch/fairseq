import torch
import importlib
import sys
#importlib.reload(sys.modules['fairseq.models.roberta'])
from fairseq.models.roberta import RobertaModel
import random


pretrain = RobertaModel.from_pretrained(
    '/new_home/zhuocheng/transformer/models/roberta.large',
    checkpoint_file='model.pt',
).cuda()
pretrain.eval()

train_data = []
with open('/new_home/zhuocheng/transformer/datasets/glue_data/CoLA/train.tsv', 'r', encoding='utf-8') as fin:
    for line in fin:
        train_data.append(line)

train_attn = []
for i in train_data:
    sent = i.strip().split('\t')[-1]
    tokens = pretrain.encode(sent).cuda()
    attn = pretrain.model.encoder.sentence_encoder.forward_scriptable(tokens.unsqueeze(0))['attn_weights']
    train_attn.append(torch.stack(attn).squeeze().cpu())

    
torch.save(train_attn, 'train_attn.pt')
