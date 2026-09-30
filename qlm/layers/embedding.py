import torch
import torch.nn as nn


def get_rotary_position_encoding(x: torch.Tensor, base=10000, device="cpu"):
  batch_size, context_length, dimension = x.shape
  assert dimension % 2 == 0, "dimension must be even"

  half_dimension = dimension // 2
  freqs_indices = torch.arange(0, half_dimension, device=device, dtype=torch.float32)
  freqs = 1.0 / (base ** (freqs_indices / dimension))
  positions = torch.arange(0, context_length, device=device, dtype=torch.float32).unsqueeze(1)
  angles = positions * freqs
  sin_angles = torch.sin(angles)
  cos_angles = torch.cos(angles)

  x_even = x[:, :, :dimension // 2]
  x_odd = x[:, :, dimension // 2:]

  x_even_rotated = x_even * cos_angles - x_odd * sin_angles
  x_odd_rotated = x_even * sin_angles + x_odd * cos_angles

  x_rotated = torch.empty_like(x, device=device)
  x_rotated[:, :, :dimension // 2] = x_even_rotated
  x_rotated[:, :, dimension // 2:] = x_odd_rotated
  return x_rotated
def rope_tablosu(context_length, head_dim, base=10000, device="cpu"):
  yari = head_dim // 2
  freqs = 1.0 / (base ** (torch.arange(0, yari, device=device,
                                       dtype=torch.float32) * 2 / head_dim))
  konum = torch.arange(context_length, device=device,
                       dtype=torch.float32).unsqueeze(1)
  aci = konum * freqs
  return torch.cos(aci), torch.sin(aci)



def rope_uygula(x, cos, sin):
  T = x.shape[-2]
  c = cos[:T].to(x.dtype).unsqueeze(0).unsqueeze(0)     # (1,1,T,yari)
  s = sin[:T].to(x.dtype).unsqueeze(0).unsqueeze(0)
  yari = x.shape[-1] // 2
  x1, x2 = x[..., :yari], x[..., yari:]
  return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

class ModelEmbedding(nn.Module):
  def __init__(self, vocab_size, embedding_dim, context_length, device, rope=True):
    super().__init__()
    assert embedding_dim % 2 == 0, "rotary encoding icin embedding_dim cift olmali"
    self.embedding = nn.Embedding(vocab_size, embedding_dim, device=device)
    nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    self.get_pos = get_rotary_position_encoding
    # rope=True eski modeller icin yenilerde rotasyon katmanlarda
    self.rope = rope
    self.vocab_size = vocab_size
    self.context_length = context_length
    self.device = device


  def forward(self, x):
    if x.dim() == 1:
      x = x.unsqueeze(0)
    assert x.dim() == 2, f"girdi (batch, context) olmali, gelen: {tuple(x.shape)}"
    if x.max() >= self.vocab_size or x.min() < 0:
      raise ValueError(f"token id'leri 0-{self.vocab_size - 1} araliginda olmali")
    if x.shape[1] > self.context_length:
      x = x[:, -self.context_length:]
    x = self.embedding(x)
    if self.rope:
      x = self.get_pos(x, device=self.device)
    return x
