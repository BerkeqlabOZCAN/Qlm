import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from .config import ModelConfig, TrainConfig
from .model import LanguageModel
from .tokenizer import BPETokenizer
from .trainer import train



def gpu_sayisi():
  return torch.cuda.device_count() if torch.cuda.is_available() else 0


def _surec(rank, dunya, bin_yolu, n_token, bpe_yolu, mcfg, tcfg,
           save_path, resume_from, dtype, port, veri_adim,
           program_sifirla):
  os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
  os.environ.setdefault("MASTER_PORT", str(port))
  # cocuk surecte stdout blok tamponlu yoksa log'lar ancak sonda gorunuyor
  try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
  except Exception:
    pass

  cuda = torch.cuda.is_available()
  dist.init_process_group("nccl" if cuda else "gloo", rank=rank, world_size=dunya)
  if cuda:
    torch.cuda.set_device(rank)
  try:
    tokens = np.memmap(bin_yolu, dtype=dtype, mode="r", shape=(n_token,))
    tok = BPETokenizer(bpe_yolu)
    if mcfg is not None:
      mcfg = ModelConfig(**{**mcfg.to_dict(),
                            "device": f"cuda:{rank}" if cuda else "cpu"})
    train(tokens, tok, mcfg, tcfg, save_path=save_path, resume_from=resume_from,
          veri_adim=veri_adim, program_sifirla=program_sifirla)
  finally:
    dist.barrier()              # rank 0 kaydi bitirsin
    dist.destroy_process_group()



def train_ddp(bin_yolu, n_token, bpe_yolu, model_config: ModelConfig = None,
              train_config: TrainConfig = None, save_path="model_weights.pt",
              resume_from=None, dtype="uint16", port=29500, nprocs=None,
              veri_adim=None, program_sifirla=False):
  dunya = nprocs or gpu_sayisi()
  if dunya < 2:
    tokens = np.memmap(bin_yolu, dtype=dtype, mode="r", shape=(n_token,))
    return train(tokens, BPETokenizer(bpe_yolu), model_config, train_config,
                 save_path=save_path, resume_from=resume_from, veri_adim=veri_adim,
                 program_sifirla=program_sifirla)

  ad = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU/gloo"
  print(f"DDP: {dunya} surec ile egitim basliyor ({ad})")
  mp.spawn(_surec,
           args=(dunya, bin_yolu, n_token, bpe_yolu, model_config, train_config,
                 save_path, resume_from, dtype, port, veri_adim,
                 program_sifirla),
           nprocs=dunya, join=True)

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  if not os.path.exists(save_path):
    raise RuntimeError(f"DDP egitimi model yazmadi: {save_path}")
  return LanguageModel.load(save_path, device=device)
