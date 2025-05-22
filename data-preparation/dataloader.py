#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
                    # tokenizes the entire text
        token_ids = tokenizer.encode(txt) 

        # using a sliding window to chunk the boook into overlapping sequences of max length
        for i in range(0, len(token_ids) -max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1 ]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # returns total no. of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    # returns a single row from the dataset
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # initialize tokenizer
    tokenizzer =tiktoken.get_encoding("gpt2")

    # creates dataset
    dataset = GPTDatasetV1(txt, tokenizzer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
     raw_text = f.read()

vocab_size=50257
output_dim=256
context_length=1024 # represents the supported input size of LLM

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

batch_szie = 8
max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=batch_szie, max_length=max_length, stride=max_length)


# In[2]:


for batch in dataloader:
    x, y = batch
    token_embeddings = token_embedding_layer(x)
                                        # torch.arange contains sequence of numbers upto max. length -1
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    input_embeddings= token_embeddings + pos_embeddings

    break


# In[5]:


print(input_embeddings.shape)

