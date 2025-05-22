#!/usr/bin/env python
# coding: utf-8

# In[233]:


from importlib.metadata import version

print("torch version: ", version("torch"))


# In[234]:


import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], 
   [0.55, 0.87, 0.66], 
   [0.57, 0.85, 0.64],
   [0.22, 0.58, 0.33],
   [0.77, 0.25, 0.10],
   [0.05, 0.80, 0.55]]
)


# In[235]:


query = inputs[1] # 2nd input token is query
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)


# In[236]:


res = 0.

for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]

print(res)
print(torch.dot(inputs[0], query))


# In[237]:


attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())


# In[238]:


def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention weights: ", attn_weights_2_naive)
print("SUm: ", attn_weights_2_naive.sum())


# In[239]:


attn_weights_2 = torch.softmax(attn_scores_2, dim = 0)
print("Attention weights: ", attn_weights_2)
print("Sum: ", attn_weights_2.sum())


# In[240]:


query = inputs[1]
context_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec_2  += attn_weights_2[i]*x_i

print(context_vec_2)


# In[241]:


attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)


# In[242]:


attn_scores = inputs @ inputs.T
print(attn_scores)


# In[243]:


attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)


# In[244]:


row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum: ", row_2_sum)
print("All row sums: ", attn_weights.sum(dim=-1))


# In[245]:


all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
print(context_vec_2)


# In[246]:


x_2 = inputs[1] # 2nd input elememt
d_in= inputs.shape[1]
d_out = 2   #output embedding size


# In[247]:


torch.manual_seed(123)

# requires_grad is used to reduce the clutter in outputs
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)


# In[248]:


query_2 = x_2 @ W_query
key_2= x_2 @ W_key
value_2 = x_2 @W_value

print(query_2, key_2, value_2)


# In[249]:


keys = inputs @ W_key
values = inputs @ W_value

print("keys.shape", keys.shape)
print("values.shape", values.shape)


# In[250]:


# now computing the unnormalized attention scores by computing dot product between query  & each key vector

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)


# In[251]:


attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)


# In[252]:


d_k  = keys.shape[1]
                # softmax turns attention scores into attention weights
attn_weights_2= torch.softmax(attn_scores_2 / d_k**0.5, dim = -1)
print(attn_weights_2)


# In[253]:


context_vec_2 = attn_weights_2 @ values
print(context_vec_2)


# In[254]:


import torch.nn as nn

# selfattention_v1 provides neccessary functionalities for model layer creation and management
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @self.W_key
        queries = x @self.W_query
        values = x @self.W_value
        attn_scores = queries @keys.T 
        attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim = -1)

        context_vec = attn_weights @ values
        return context_vec


# In[255]:


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))


# In[256]:


class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec =attn_weights @values

        return context_vec


# In[257]:


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2)


# In[258]:


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5 , dim=-1)
print(attn_weights)


# In[259]:


context_length= attn_scores.shape[0]
mask_simple =torch.tril(torch.ones(context_length, context_length))
print(mask_simple)


# In[260]:


# zeroing out values above diagonal
masked_simple = attn_weights*mask_simple
print(masked_simple)


# In[261]:


# renormalizing attention weights to sum up to 1 again in each row
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple/row_sums

print(masked_simple_norm)


# In[262]:


mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked  =  attn_scores.masked_fill(mask.bool(), -torch.inf)

print(masked)


# In[263]:


# softmax converts its input into probability distribution
attn_weights= torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print(attn_weights)


# In[264]:


# dropout helps in preventing overfitting by ensuring that model doesnt become overly reliant on any specific set of hiddenb layer units
# only used during training and disabled afterward
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # choosing a dropout rate of 50%

#to compensate for reduction in active elements, values of remainign elements in matrix aare scaled by 1/0.5

example = torch.ones(6, 6)
print(dropout(example))


# In[265]:


torch.manual_seed(123)
print(dropout(attn_weights))


# In[266]:


batch = torch.stack((inputs, inputs), 0)
print(batch.shape)


# In[267]:


# in causal attention a token is allowed to attend to previos tokens and itself but not to any future tokens

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        return context_vec


# In[268]:


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs shape: ", context_vecs.shape)


# In[269]:


# combining multple single head attention modules
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias
                                                    )
                                                    for _ in range(num_heads) ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# In[270]:


torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3, 2

mha= MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs shape: ", context_vecs.shape)


# In[271]:


# integratin multi head functionality within a single class
# splits input into mutilple heads by reshaping the projected query, key and value
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
        "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out //num_heads   #reduces projection dim to match desrired output dim
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj= nn.Linear(d_out, d_out)
        self.dropout =nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # tensor shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) 
        queries = queries.transpose(1, 2) 
        values = values.transpose(1, 2) 
        attn_scores = queries @ keys.transpose(2, 3) 
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens] 

        attn_scores.masked_fill_(mask_bool, -torch.inf) 
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2) 

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) 

        return context_vec


# In[272]:


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

