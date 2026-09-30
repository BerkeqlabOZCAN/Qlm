import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import rope_tablosu, rope_uygula

class MultiHeadAttention(nn.Module):
  def __init__(self, embedding_dim, num_heads, context_length, device, dropout=0.1,
               rope=False):
    super().__init__()
    assert embedding_dim % num_heads == 0, f"embedding_dim ({embedding_dim}) num_heads'e ({num_heads}) bolunmeli"

    self.num_heads = num_heads
    self.head_dim = embedding_dim // num_heads

    self.w_query = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
    self.w_key = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
    self.w_value = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)
    self.w_output = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device)

    self.dropout = nn.Dropout(dropout)   # kullanilmiyor eski ckpt'ler icın
    self.dropout_p = dropout


    # bu da kullanilmiyor (is_causal=True) eski state_dict'ler yuklensin diye duruyor
    causal_mask = torch.triu(torch.ones(context_length, context_length, device=device), diagonal=1).bool()
    self.register_buffer("causal_mask", causal_mask)

    # rope=False eski modeller rotasyon sadece embedding'de bir kez yapiliyordu
    self.rope = rope
    if rope:
      cos, sin = rope_tablosu(context_length, self.head_dim, device=device)
      self.register_buffer("rope_cos", cos, persistent=False)
      self.register_buffer("rope_sin", sin, persistent=False)
    self.embedding_dim = embedding_dim
    self.context_length = context_length
    self.device = device

  def forward(self, x):
    assert x.dim() == 3, f"girdi (batch, context, dim) olmali, gelen: {tuple(x.shape)}"
    batch_size, context_length, dimension = x.shape
    assert dimension == self.embedding_dim, f"embedding_dim uyusmuyor: {self.embedding_dim} vs {dimension}"

    query = self.w_query(x)
    key = self.w_key(x)
    value = self.w_value(x)

    query = query.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.view(batch_size, context_length, self.num_heads, self.head_dim).transpose(1, 2)


    if self.rope:                    # v'ye uyğulanmaz
      query = rope_uygula(query, self.rope_cos, self.rope_sin)
      key = rope_uygula(key, self.rope_cos, self.rope_sin)

    output = F.scaled_dot_product_attention(
        query, key, value,
        is_causal=True,
        dropout_p=self.dropout_p if self.training else 0.0,
    )
    output = output.transpose(1, 2).contiguous().view(batch_size, context_length, self.embedding_dim)
    output = self.w_output(output)
    return output, None
