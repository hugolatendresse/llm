"""
TODO 

Send to device everywhere!

"""



import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import os
import math
import time
torch.manual_seed(137)


class AttentionModule(nn.Module):

    def __init__(self, num_head, embedding_size):
        super().__init__()
        self.num_head = num_head
        self.embedding_size = embedding_size
        assert self.embedding_size % self.num_head == 0
        self.qkv_matrices = nn.Linear(self.embedding_size, 3 * self.embedding_size)
        self.qkv_projections = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x):
        batch_size, seq_len, q_dim = x.size()
        qkv = self.qkv_matrices(x)
        q, k, v = qkv.split(self.embedding_size, dim=2)
        k = k.view(batch_size, seq_len, self.num_head, q_dim // self.num_head).transpose(1, 2) 
        q = q.view(batch_size, seq_len, self.num_head, q_dim // self.num_head).transpose(1, 2) 
        v = v.view(batch_size, seq_len, self.num_head, q_dim // self.num_head).transpose(1, 2) 
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch_size, seq_len, q_dim) 
        return self.qkv_projections(attn)

class MultiLayerPerceptron(nn.Module):

    def __init__(self, embedding_size):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, 4 * embedding_size)
        self.activation = nn.GELU(approximate='tanh')
        self.linear2 = nn.Linear(4 * embedding_size, embedding_size)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

class TransformerBlock(nn.Module):

    def __init__(self, vocab_size, block_cnt, seq_len, num_head, embedding_size):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_size)
        self.multi_head_attention = AttentionModule( 
                                        num_head=num_head, 
                                        embedding_size=embedding_size
                                        )
        self.layer_norm2 = nn.LayerNorm(embedding_size)
        self.multi_layer_perceptron = MultiLayerPerceptron(
                                        embedding_size=embedding_size
                                        )

    def forward(self, x):
        x = x + self.multi_head_attention(self.layer_norm1(x))
        x = x + self.multi_layer_perceptron(self.layer_norm2(x))
        return x


class Transformer(nn.Module):

    def __init__(self, seq_len, vocab_size, block_cnt, num_head, embedding_size, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.block_cnt = block_cnt
        self.num_head = num_head
        self.embedding_size = embedding_size
        self.hyperparams = dict(
            seq_len=seq_len, vocab_size=vocab_size, block_cnt=block_cnt, num_head=num_head, embedding_size=embedding_size)

        self.positional_embeddings = nn.Embedding(self.seq_len, self.embedding_size)
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.blocks = nn.Sequential(*[TransformerBlock(**self.hyperparams) for _ in range(self.block_cnt)]) 
        self.layer_norm = nn.LayerNorm(self.embedding_size)
        self.unembed = nn.Linear(self.embedding_size, self.vocab_size, bias=False)
        self.word_embeddings.weight = self.unembed.weight

        self.apply(self.initialize_weights) # initialize weights

    def initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / ((2 * self.block_cnt) ** 0.5)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, labels=None):
        batch_size, seq_len = idx.size()
        position_idxs = torch.arange(0, seq_len, dtype=torch.long, device=idx.device) 
        x = self.word_embeddings(idx) + self.positional_embeddings(position_idxs)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.unembed(x) 
        loss = None
        if labels is not None:
            input = logits.view(-1, logits.size(-1))
            target = labels.view(-1)
            loss = F.cross_entropy(input, target)
        return logits, loss