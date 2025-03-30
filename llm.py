import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    max_seq_len: int = 1024
    vocab_len: int = 50257
    layer_cnt: int = 12
    head_cnt: int = 12 
    model_dim: int = 768


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.head_cnt = config.head_cnt
        self.model_dim = config.model_dim
        self.qkv_mat = nn.Linear(config.model_dim, 3 * config.model_dim)
        self.linear = nn.Linear(config.model_dim, config.model_dim)
        self.linear.linear_layer = 1 

    def forward(self, x):
        # TODO compare with Needle
        batch, seq_len, model_dim = x.size()
        q, k, v = self.qkv_mat(x).split(self.model_dim, dim=2)

        # k, q, v dimensions should be (batch, head, seq, model_dim)
        assert self.model_dim % self.head_cnt == 0
        k = k.view(batch, seq_len, self.head_cnt, model_dim // self.head_cnt)
        q = q.view(batch, seq_len, self.head_cnt, model_dim // self.head_cnt)
        v = v.view(batch, seq_len, self.head_cnt, model_dim // self.head_cnt)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, model_dim) 
        y = self.linear(y)
        return y
