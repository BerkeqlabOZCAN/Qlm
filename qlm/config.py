from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
  vocab_size: int
  embedding_dim: int = 64
  context_length: int = 32
  num_heads: int = 4
  num_layers: int = 4
  dropout: float = 0.1
  device: str = "cpu"
  # eski modelde degistirme
  rope_katman: bool = False

  def to_dict(self):
    return asdict(self)


@dataclass
class TrainConfig:
  epochs: int = 100
  batch_size: int = 8
  stride: int = 8
  lr: float = 3e-3
  val_split: float = 0.1
  eval_every: int = 10         # epoch
  seed: int = 1
  warmup_ratio: float = 0.03
  min_lr_ratio: float = 0.1    # cosine'in inecegi taban
  use_amp: bool = True         # sadece cuda'da etkin
  save_every_steps: int = 0    # ara kayit 0 = kapali
  betas: tuple = (0.9, 0.95)
  weight_decay: float = 0.1    # sadece 2+ boyutlu agirliklara
  grad_clip: float = 1.0
  grad_accum: int = 1          # efektif batch = batch_size * grad_accum
  eval_every_steps: int = 0
  eval_batches: int = 50       # 0 = tum val setı
  num_workers: int = 0
  max_hours: float = 0.0
