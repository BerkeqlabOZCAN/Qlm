import math

import torch
import torch.nn as nn


from .attention import MultiHeadAttention


class LayerNorm(nn.Module):
  def __init__(self, embedding_dim, device, eps=1e-5):
    super().__init__()
    self.gamma = nn.Parameter(torch.ones(embedding_dim, device=device))
    self.beta = nn.Parameter(torch.zeros(embedding_dim, device=device))
    self.eps = eps

  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + self.eps)
    return self.gamma * x_norm + self.beta



class GELU(nn.Module):
  def forward(self, x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))



class GatedMLP(nn.Module):
  def __init__(self, embedding_dim, device, dropout=0.1):
    super().__init__()
    hidden_dim = int(2 * 4 * embedding_dim / 3)
    self.gate_proj = nn.Linear(embedding_dim, hidden_dim, bias=False, device=device)
    self.up_proj = nn.Linear(embedding_dim, hidden_dim, bias=False, device=device)
    self.down_proj = nn.Linear(hidden_dim, embedding_dim, bias=False, device=device)
    self.gelu = GELU()
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    gate = self.gelu(self.gate_proj(x))
    up = self.up_proj(x)
    return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):

  def __init__(self, embedding_dim, num_heads, context_length, device, dropout=0.1,
               rope=False):
    super().__init__()
    self.norm1 = LayerNorm(embedding_dim, device=device)
    self.attention = MultiHeadAttention(embedding_dim, num_heads, context_length,
                                        device, dropout, rope=rope)
    self.norm2 = LayerNorm(embedding_dim, device=device)
    self.feed_forward = GatedMLP(embedding_dim, device, dropout)

  def forward(self, x):
    attention_output, _ = self.attention(self.norm1(x))
    x = x + attention_output
    x = x + self.feed_forward(self.norm2(x))
    return x
