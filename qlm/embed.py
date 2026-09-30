import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingModel(nn.Module):

  def __init__(self, model, tokenizer=None, pooling="last"):
    super().__init__()
    assert pooling in ("last", "mean"), f"pooling 'last' ya da 'mean' olmali, gelen: {pooling}"

    self.embedding = model.embedding
    self.blocks = model.blocks
    self.final_norm = model.final_norm

    self.tokenizer = tokenizer
    self.pooling = pooling
    self.context_length = model.context_length
    self.device = model.device
    self.embedding_dim = model.config.embedding_dim
    self.pad_id = getattr(tokenizer, "pad_id", None)

  def forward_hidden(self, input_ids):
    x = self.embedding(input_ids)
    for block in self.blocks:
      x = block(x)
    return self.final_norm(x)

  def pool(self, hidden, attention_mask):
    if self.pooling == "last":
      lengths = attention_mask.sum(dim=1)                      # (B,)
      last_idx = (lengths - 1).clamp(min=0).long()            # (B,)
      batch_idx = torch.arange(hidden.size(0), device=hidden.device)
      return hidden[batch_idx, last_idx]                       # (B D)
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)      # (B T 1)
    summed = (hidden * mask).sum(dim=1)                        # (B D)
    counts = mask.sum(dim=1).clamp(min=1e-9)                   # (B 1)
    return summed / counts

  @torch.no_grad()
  def encode(self, texts, batch_size=32, normalize=True, show_progress=False):
    if self.tokenizer is None:
      raise ValueError("encode() icin tokenizer gerekli; EmbeddingModel(model, tokenizer) ver.")
    if isinstance(texts, str):
      texts = [texts]

    was_training = self.training
    self.eval()
    pad_id = self.pad_id if self.pad_id is not None else 0

    out = []
    ranger = range(0, len(texts), batch_size)
    if show_progress:
      try:
        from tqdm import tqdm
        ranger = tqdm(ranger, desc="Encoding")
      except ImportError:
        pass

    for start in ranger:
      batch = texts[start:start + batch_size]

      seqs = []
      for t in batch:
        ids = self.tokenizer.encode(t)
        if len(ids) == 0:
          ids = [pad_id]                      # bos metin
        ids = ids[:self.context_length]
        seqs.append(ids)

      max_len = max(len(s) for s in seqs)
      input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=self.device)
      attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)
      for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long, device=self.device)
        attention_mask[i, :len(s)] = 1

      hidden = self.forward_hidden(input_ids)
      vecs = self.pool(hidden, attention_mask)                # (B D)
      if normalize:
        vecs = F.normalize(vecs, p=2, dim=1)
      out.append(vecs.float().cpu().numpy())

    if was_training:
      self.train()
    return np.vstack(out).astype(np.float32)
  @classmethod
  def from_pretrained(cls, model_path, tokenizer=None, device="cpu", pooling="last"):
    from .model import LanguageModel
    lm = LanguageModel.load(model_path, device=device)
    return cls(lm, tokenizer=tokenizer, pooling=pooling)



if __name__ == "__main__":
  # kucuk rastgele modelle sekil testi
  from .config import ModelConfig
  from .model import LanguageModel

  cfg = ModelConfig(vocab_size=50, embedding_dim=32, context_length=16, num_heads=4, num_layers=2)
  lm = LanguageModel(cfg)
  class DummyTok:
    pad_id = 0
    def encode(self, text):
      return [(ord(c) % 49) + 1 for c in text][:16] or [0]

  emb = EmbeddingModel(lm, DummyTok(), pooling="last")
  vecs = emb.encode(["merhaba dunya", "kisa", "cok daha uzun bir ornek cumle burada"])
  norms = np.linalg.norm(vecs, axis=1)
  print("cikti sekli:", vecs.shape)                 # (3 32)
  print("L2 normlar (hepsi ~1 olmali):", norms.round(4))
  print("self-benzerlik (~1):", round(float(vecs[0] @ vecs[0]), 4))
  print("OK" if vecs.shape == (3, 32) and np.allclose(norms, 1.0, atol=1e-4) else "HATA")
