import torch
import torch.nn as nn


from .config import ModelConfig
from .layers import ModelEmbedding, TransformerBlock, LayerNorm


class LanguageModel(nn.Module):


  def __init__(self, config: ModelConfig):
    super().__init__()
    self.config = config

    katman_rope = getattr(config, "rope_katman", False)
    self.embedding = ModelEmbedding(
        config.vocab_size, config.embedding_dim, config.context_length,
        config.device, rope=not katman_rope
    )
    self.blocks = nn.ModuleList([
        TransformerBlock(
            config.embedding_dim, config.num_heads, config.context_length,
            config.device, config.dropout, rope=katman_rope,
        )
        for _ in range(config.num_layers)
    ])
    self.final_norm = LayerNorm(config.embedding_dim, device=config.device)
    self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False, device=config.device)
    self.lm_head.weight = self.embedding.embedding.weight  # weight tying

    self.context_length = config.context_length
    self.device = config.device

  def forward(self, x):
    x = self.embedding(x)
    for block in self.blocks:
      x = block(x)
    x = self.final_norm(x)
    return self.lm_head(x)

  def num_parameters(self):
    return sum(p.numel() for p in self.parameters())

  def save(self, path):
    torch.save({"config": self.config.to_dict(), "state_dict": self.state_dict()}, path)

  @classmethod
  def load(cls, path, device="cpu"):
    checkpoint = torch.load(path, weights_only=False, map_location=device)
    config = ModelConfig(**checkpoint["config"])
    config.device = device
    model = cls(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
