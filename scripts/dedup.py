"""Korpustan tekrar eden belgeleri atar (tam kopya + yakin kopya)"""
import argparse
import hashlib
import os
import re
from multiprocessing import Pool


import numpy as np


SEP = "\n\n"
_BOSLUK = re.compile(r"\s+")
MASKE64 = (1 << 64) - 1

def _normalize(metin):
  return _BOSLUK.sub(" ", metin.strip().lower())

def _tam_hash(metin):
  return int.from_bytes(
      hashlib.blake2b(_normalize(metin).encode("utf-8", "ignore"),
                      digest_size=8).digest(), "big")


def _simhash(metin, azami_kelime=200):
  """64-bit SimHash kelime hash'lerinin bit bazinda agirlikli oyu"""
  kelimeler = _normalize(metin).split()[:azami_kelime]
  if not kelimeler:
    return 0
  sayac = [0] * 64
  for k in kelimeler:
    h = int.from_bytes(hashlib.blake2b(k.encode("utf-8", "ignore"),
                                       digest_size=8).digest(), "big")
    for b in range(64):
      sayac[b] += 1 if (h >> b) & 1 else -1
  imza = 0
  for b in range(64):
    if sayac[b] > 0:
      imza |= (1 << b)
  return imza


def _imzala(belgeler):
  return [(_tam_hash(b), _simhash(b)) for b in belgeler]


def _belgeler(yol, sep=SEP):
  tampon = ""
  with open(yol, encoding="utf-8", errors="ignore") as f:
    for satir in f:
      tampon += satir
      while sep in tampon:
        belge, tampon = tampon.split(sep, 1)
        if belge.strip():
          yield belge.strip()
  if tampon.strip():
    yield tampon.strip()

def _yigin(uretec, n):
  yig = []
  for x in uretec:
    yig.append(x)
    if len(yig) >= n:
      yield yig
      yig = []
  if yig:
    yield yig



def _hamming(a, b):
  return bin(int(a) ^ int(b)).count("1")

def yakin_kopyalari_bul(imzalar, esik=3, tur=4):
  """SimHash imzalarindan yakin kopya maskesi uretir (True = at)"""
  n = len(imzalar)
  at = np.zeros(n, dtype=bool)
  for t in range(tur):
    kaydir = t * (64 // max(1, tur))
    if kaydir:
      dondurulmus = ((imzalar >> np.uint64(kaydir)) |
                     (imzalar << np.uint64(64 - kaydir)))
    else:
      dondurulmus = imzalar
    sira = np.argsort(dondurulmus, kind="stable")
    for i in range(1, n):
      a, b = sira[i - 1], sira[i]
      if at[b] or at[a]:
        continue
      if _hamming(imzalar[a], imzalar[b]) <= esik:
        at[max(a, b)] = True          # sonra geleni at ilkini tut
  return at

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("korpus")
  ap.add_argument("--cikti", default=None, help="varsayilan: <korpus>.dedup")
  # 8 olcerek bulundu kopyalar <=6 farkli belgeler >=12 cikiyordu
  ap.add_argument("--esik", type=int, default=8,
                  help="SimHash Hamming esigi (olculen guvenli aralik 6-10)")
  ap.add_argument("--tur", type=int, default=4, help="siralama turu sayisi")
  ap.add_argument("--surec", type=int, default=None)
  ap.add_argument("--yigin", type=int, default=2000)
  ap.add_argument("--sadece-tam", action="store_true",
                  help="yakin kopya taramasini atla (sadece birebir ayni)")
  args = ap.parse_args()

  cikti = args.cikti or args.korpus + ".dedup"
  surec = args.surec or max(1, (os.cpu_count() or 2) - 1)
  print(f"korpus: {args.korpus} ({os.path.getsize(args.korpus)/1e9:.1f} GB)")
  print(f"imzalar hesaplaniyor ({surec} surec)...")

  tam, sim = [], []
  with Pool(processes=surec) as havuz:
    for parca in havuz.imap(_imzala, _yigin(_belgeler(args.korpus), args.yigin),
                            chunksize=1):
      for t, s in parca:
        tam.append(t)
        sim.append(s)
      if len(tam) % 200000 < args.yigin:
        print(f"  {len(tam):,} belge imzalandi")

  n = len(tam)
  tam = np.array(tam, dtype=np.uint64)
  sim = np.array(sim, dtype=np.uint64)
  print(f"toplam {n:,} belge")

  # 1) tam kopya ilk gorulen kalir
  _, ilk = np.unique(tam, return_index=True)
  at = np.ones(n, dtype=bool)
  at[ilk] = False
  tam_kopya = int(at.sum())
  print(f"  tam kopya : {tam_kopya:,} (%{tam_kopya/n*100:.1f})")

  # 2) yakin kopya (tam kopyalar disinda kalanlar uzerinde)
  yakin_kopya = 0
  if not args.sadece_tam:
    kalan = np.where(~at)[0]
    yakin = yakin_kopyalari_bul(sim[kalan], esik=args.esik, tur=args.tur)
    at[kalan[yakin]] = True
    yakin_kopya = int(yakin.sum())
    print(f"  yakin kopya: {yakin_kopya:,} (%{yakin_kopya/n*100:.1f})")

  kalan_sayi = n - int(at.sum())
  print(f"kalan     : {kalan_sayi:,} belge (%{kalan_sayi/n*100:.1f})")

  print("temiz korpus yaziliyor...")
  yazilan = 0
  with open(cikti, "w", encoding="utf-8", newline="\n") as f:
    for i, belge in enumerate(_belgeler(args.korpus)):
      if not at[i]:
        f.write(belge + SEP)
        yazilan += 1
  print(f"HAZIR: {cikti} | {yazilan:,} belge | "
        f"{os.path.getsize(cikti)/1e9:.1f} GB")
if __name__ == "__main__":
  main()
