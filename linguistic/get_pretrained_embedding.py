
from transformers import BertTokenizer, BertModel
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")

test = pd.read_csv('test.csv')
# test = pd.read_csv('test.csv')

test_emb = []

text = test['text']
for t in text:

    encoded_input = tokenizer(t, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded_input)
    test_emb.append(output[1].detach().squeeze(0))


test_emb = torch.stack(test_emb)
torch.save(test_emb, 'test_emb.pt')
print(train_emb)