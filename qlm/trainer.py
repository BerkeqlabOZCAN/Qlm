import contextlib
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from .config import ModelConfig, TrainConfig
from .model import LanguageModel
from .data import create_data_loader


class _AtlamaliSampler(Sampler):

  def __init__(self, temel, atla):
    self.temel = temel
    self.atla = max(0, int(atla))

  def __iter__(self):
    it = iter(self.temel)
    for _ in range(self.atla):
      if next(it, None) is None:
        break
    return it

  def __len__(self):
    return max(0, len(self.temel) - self.atla)
def _make_scheduler(optimizer, total_steps, warmup_ratio, min_lr_ratio):
  warmup = max(1, int(warmup_ratio * total_steps))

  def lr_lambda(step):
    if step < warmup:
      return (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr_ratio + (1 - min_lr_ratio) * cosine


  return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def train(tokens, tokenizer, model_config: ModelConfig = None,
          train_config: TrainConfig = None, save_path="model_weights.pt",
          verbose=True, resume_from=None, veri_adim=None,
          program_sifirla=False):
  tcfg = train_config or TrainConfig()
  ddp = dist.is_available() and dist.is_initialized()
  rank = dist.get_rank() if ddp else 0
  dunya = dist.get_world_size() if ddp else 1
  ana = (rank == 0)
  verbose = verbose and ana
  if resume_from is not None:
    ck = torch.load(resume_from, weights_only=False, map_location="cpu")
    model_config = ModelConfig(**ck["config"])
  if model_config is None:
    model_config = ModelConfig(vocab_size=len(tokenizer))
  if ddp and torch.cuda.is_available():
    # ckpt'deki "cuda:0" kalirsa tum rank'lar ayni karta biniyor
    model_config.device = f"cuda:{torch.cuda.current_device()}"
  elif ddp:
    model_config.device = "cpu"   # gloo yerel test

  torch.manual_seed(tcfg.seed)

  pad_id = tokenizer.pad_id if tokenizer.pad_id is not None else 0
  ctx = model_config.context_length

  split = int(len(tokens) * (1 - tcfg.val_split))
  train_tokens, val_tokens = tokens[:split], tokens[split:]

  train_sampler = None
  if ddp:
    # sampler sadece sayiyi istiyor
    pencere = max(0, len(train_tokens) - ctx - 1) // tcfg.stride + 1
    train_sampler = DistributedSampler(range(pencere), num_replicas=dunya,
                                       rank=rank, shuffle=True, drop_last=True)
  train_loader = create_data_loader(train_tokens, ctx, tcfg.stride, tcfg.batch_size,
                                    shuffle=True, pad_id=pad_id, device=model_config.device,
                                    sampler=train_sampler, num_workers=tcfg.num_workers)
  # val'i sadece rank 0 olcuyor bolmeye gerek yok
  val_loader = create_data_loader(val_tokens, ctx, ctx, tcfg.batch_size,
                                  shuffle=False, pad_id=pad_id, device=model_config.device,
                                  num_workers=tcfg.num_workers)
  device = model_config.device
  model = LanguageModel(model_config)
  if resume_from is not None:
    model.load_state_dict(ck["state_dict"])
    if verbose:
      print(f"resume: {resume_from} yuklendi, egitime devam")

  ham = model                            # dDP'siz hali kayit icin
  if ddp:
    ind = torch.device(device).index
    model = DDP(model, device_ids=[ind] if ind is not None else None)

  # layernorm ve bias'a weight decay yok
  kararli = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
  ceza_yok = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
  gruplar = [{"params": kararli, "weight_decay": tcfg.weight_decay},
             {"params": ceza_yok, "weight_decay": 0.0}]


  try:
    optimizer = torch.optim.AdamW(gruplar, lr=tcfg.lr, betas=tuple(tcfg.betas),
                                  fused=str(device).startswith("cuda"))
  except (RuntimeError, TypeError):
    optimizer = torch.optim.AdamW(gruplar, lr=tcfg.lr, betas=tuple(tcfg.betas))

  accum = max(1, tcfg.grad_accum)
  # scheduler optimizer adimini sayiyor mikro-batch'i degil
  adim_per_epoch = max(1, math.ceil(len(train_loader) / accum))
  total_steps = max(1, tcfg.epochs) * adim_per_epoch
  scheduler = _make_scheduler(optimizer, total_steps, tcfg.warmup_ratio, tcfg.min_lr_ratio)

  use_amp = tcfg.use_amp and str(device).startswith("cuda")
  scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

  gecmis_best = float("inf")
  gecmis_veri = None
  if resume_from is not None and os.path.exists(resume_from + ".ckpt"):
    ts = torch.load(resume_from + ".ckpt", map_location=device, weights_only=False)
    try:
      optimizer.load_state_dict(ts["optimizer"])
      if program_sifirla:
        # scaler'i da almiyoruz kendini birkac adimda ayarliyor
        if verbose:
          print("program sifirlandi: agirlik + optimizer momentleri korundu, "
                "LR programi/veri konumu/best_val BASTAN")
      else:
        scheduler.load_state_dict(ts["scheduler"])
        scaler.load_state_dict(ts["scaler"])
        gecmis_best = ts.get("best_val", float("inf"))
        gecmis_veri = ts.get("veri_adim")   # eski ckpt'lerde yok
        if verbose:
          print(f"tam durum yuklendi (.ckpt): scheduler adim {scheduler.last_epoch}, "
                f"best_val {gecmis_best:.4f}")
    except Exception as e:
      if verbose:
        print(f"tam durum yuklenemedi ({type(e).__name__}), sadece agirliklarla devam")

  # resume'da veriyi bastan okumasin diye kaldigi yeri atliyoruz
  veri_baslangic = 0
  gecmis_adim = 0 if program_sifirla else (
                 veri_adim if veri_adim is not None else
                 gecmis_veri if gecmis_veri is not None else
                 (scheduler.last_epoch if resume_from is not None else 0))
  atlanacak_ornek = 0
  if gecmis_adim and adim_per_epoch:
    gecis = gecmis_adim // adim_per_epoch
    kalan_adim = gecmis_adim % adim_per_epoch
    atlanacak_ornek = kalan_adim * accum * tcfg.batch_size
    if train_sampler is not None:
      train_sampler.set_epoch(gecis)
      if atlanacak_ornek:
        train_loader = DataLoader(
            train_loader.dataset, batch_size=tcfg.batch_size,
            sampler=_AtlamaliSampler(train_sampler, atlanacak_ornek),
            num_workers=tcfg.num_workers,
            pin_memory=str(device).startswith("cuda"))
    veri_baslangic = kalan_adim
    if verbose:
      print(f"veri konumu: gecis {gecis}, bu geciste {kalan_adim:,} adim "
            f"({atlanacak_ornek:,} pencere) atlandi -> TAZE veriden devam")

  def kaydet_tam():
    if not ana:
      return
    ham.save(save_path)
    torch.save({"optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_val": best_val,
                "veri_adim": veri_baslangic + adim}, save_path + ".ckpt")

  def loss_fn(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=pad_id)

  @torch.no_grad()
  def val_loss(max_batches=0):
    ham.eval()
    toplam, sayi = 0.0, 0
    for x, y in val_loader:
      x, y = x.to(device), y.to(device)
      with torch.amp.autocast("cuda", enabled=use_amp):
        toplam += loss_fn(ham(x), y).item()
      sayi += 1
      if max_batches and sayi >= max_batches:
        break
    ham.train()
    return toplam / max(1, sayi)

  if verbose:
    print(f"parametre: {ham.num_parameters():,} | train {len(train_tokens):,} / "
          f"val {len(val_tokens):,} token | device: {device} | AMP: {use_amp}"
          + (f" | DDP: {dunya} surec" if ddp else ""))
    print(f"AdamW: lr {tcfg.lr:.1e} | betas {tuple(tcfg.betas)} | wd {tcfg.weight_decay} "
          f"({sum(p.numel() for p in kararli):,} parametreye, "
          f"{sum(p.numel() for p in ceza_yok):,} parametre muaf)")
    print(f"adim/epoch: {adim_per_epoch:,} | efektif batch: {tcfg.batch_size * accum * dunya} "
          f"(batch {tcfg.batch_size} x accum {accum}"
          + (f" x {dunya} surec)" if ddp else ")")
          + (f" | zaman butcesi: {tcfg.max_hours} saat" if tcfg.max_hours else ""))

  model.train()
  best_val = gecmis_best
  adim = 0                 # bu kosudaki optimizer adimi
  baslangic = time.time()
  sure_doldu = False
  optimizer.zero_grad(set_to_none=True)
  for epoch in range(tcfg.epochs):
    if train_sampler is not None:
      train_sampler.set_epoch(epoch)
    toplam, mikro = 0.0, 0
    for inputs, targets in train_loader:
      inputs, targets = inputs.to(device), targets.to(device)
      # accum bitene kadar gradyanlari senkronlama
      son_mikro = (mikro + 1) % accum == 0
      senk = contextlib.nullcontext() if (not ddp or son_mikro) else model.no_sync()
      with senk:
        with torch.amp.autocast("cuda", enabled=use_amp):
          loss = loss_fn(model(inputs), targets)
        scaler.scale(loss / accum).backward()
      toplam += loss.item()
      mikro += 1

      if mikro % accum:
        continue
      if tcfg.grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
      scaler.step(optimizer)
      scaler.update()
      scheduler.step()
      optimizer.zero_grad(set_to_none=True)
      adim += 1

      if tcfg.save_every_steps and adim % tcfg.save_every_steps == 0:
        kaydet_tam()
        if verbose:
          gecen = (time.time() - baslangic) / 3600
          print(f"  adim {adim:,} | loss {loss.item():.4f} | "
                f"lr {scheduler.get_last_lr()[0]:.2e} | {gecen:.2f} saat | ara kayit",
                flush=True)

      if ana and tcfg.eval_every_steps and adim % tcfg.eval_every_steps == 0:
        vl = val_loss(tcfg.eval_batches)
        mark = ""
        if vl < best_val:
          best_val = vl
          kaydet_tam()
          mark = "  <- en iyi, kaydedildi"
        if verbose:
          print(f"  adim {adim:,} | val: {vl:.4f}{mark}", flush=True)
      # ranklar beraber durmali
      if tcfg.max_hours:
        dur = float((time.time() - baslangic) >= tcfg.max_hours * 3600)
        if ddp:
          bayrak = torch.tensor([dur], device=device)
          dist.all_reduce(bayrak, op=dist.ReduceOp.MAX)
          dur = bayrak.item()
        if dur:
          kaydet_tam()
          sure_doldu = True
          if verbose:
            print(f"zaman butcesi doldu ({tcfg.max_hours} saat, adim {adim:,}) -> "
                  f"{save_path} kaydedildi, temiz cikis")
          break

    if sure_doldu:
      break

    if ana and (epoch % tcfg.eval_every == 0 or epoch == tcfg.epochs - 1):
      tr = toplam / max(1, len(train_loader))
      vl = val_loss()
      mark = ""
      if vl < best_val:
        best_val = vl
        kaydet_tam()
        mark = "  <- kaydedildi"
      if verbose:
        print(f"epoch {epoch:3d} | train: {tr:.4f} | val: {vl:.4f}{mark}", flush=True)

  if verbose:
    print(f"en iyi validation loss: {best_val:.4f} ({save_path})")
  if not ana:
    return None
  return LanguageModel.load(save_path, device=model_config.device)
