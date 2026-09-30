"""Egitilmis bir modeli katman ekleyerek buyutur (agirliklar korunur)"""
import argparse
import os
import sys

import torch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from qlm import LanguageModel, ModelConfig   # noqa: E402



def yerlesim(eski_n, yeni_n):
  """Eski katmanlarin yeni dizideki konumlari (yeniler esit araliklarla arada)"""
  assert yeni_n >= eski_n, "kucultme desteklenmiyor"
  konum = [round(i * (yeni_n - 1) / (eski_n - 1)) for i in range(eski_n)]
  # cakismalari ac (round ayni degeri iki kez verebilir)
  for i in range(1, eski_n):
    if konum[i] <= konum[i - 1]:
      konum[i] = konum[i - 1] + 1
  assert konum[-1] <= yeni_n - 1, f"yerlesim tasti: {konum}"
  return konum
def buyut(girdi, cikti, yeni_katman, verbose=True):
  ck = torch.load(girdi, weights_only=False, map_location="cpu")
  cfg_d = dict(ck["config"])
  eski_katman = cfg_d["num_layers"]
  cfg_d["num_layers"] = yeni_katman
  cfg_d["device"] = "cpu"

  eski_cfg = ModelConfig(**{**ck["config"], "device": "cpu"})
  eski = LanguageModel(eski_cfg)
  eski.load_state_dict(ck["state_dict"])
  eski.eval()
  yeni = LanguageModel(ModelConfig(**cfg_d))
  yeni.eval()

  # bagli agirliklari koru
  yeni.embedding.load_state_dict(eski.embedding.state_dict())
  yeni.final_norm.load_state_dict(eski.final_norm.state_dict())

  konum = yerlesim(eski_katman, yeni_katman)
  for i, k in enumerate(konum):
    yeni.blocks[k].load_state_dict(eski.blocks[i].state_dict())

  yeniler = [k for k in range(yeni_katman) if k not in set(konum)]
  for k in yeniler:
    b = yeni.blocks[k]
    torch.nn.init.zeros_(b.attention.w_output.weight)      # residual dali = 0
    torch.nn.init.zeros_(b.feed_forward.down_proj.weight)  # residual dali = 0

  if verbose:
    print(f"katman {eski_katman} -> {yeni_katman}")
    print(f"  tasinan konumlar : {konum}")
    print(f"  YENI (sifir) : {yeniler}")
    print(f"  parametre {eski.num_parameters():,} -> {yeni.num_parameters():,} "
          f"(+%{(yeni.num_parameters()/eski.num_parameters()-1)*100:.0f})")


  # dogrulama ciktı birebir ayni mi
  g = torch.Generator().manual_seed(0)
  x = torch.randint(0, cfg_d["vocab_size"], (2, 64), generator=g)
  with torch.no_grad():
    a, b_ = eski(x), yeni(x)
  fark = (a - b_).abs().max().item()
  print(f"  DOGRULAMA azami fark: {fark:.3e}", end=" ")
  assert fark < 1e-4, f"BUYUTME BOZUK: cikti degisti ({fark})"
  print("-> birim fonksiyon, cikti korundu")

  yeni.save(cikti)
  print(f"kaydedildi: {cikti} ({os.path.getsize(cikti)/1e6:.0f} MB)")
  print("NOT: .ckpt uretilmedi — optimizer/scheduler bastan kurulacak.")
  return yeni


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("girdi")
  ap.add_argument("cikti")
  ap.add_argument("--katman", type=int, required=True)
  a = ap.parse_args()
  buyut(a.girdi, a.cikti, a.katman)


if __name__ == "__main__":
  main()
