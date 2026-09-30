"""Korpusu yerel makinede hazirlar"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlm import (train_bpe, tokenize_corpus, korpus_yaz, ornek_al,  # noqa: E402
                     VARSAYILAN_KAYNAKLAR, TAZE_KAYNAKLAR, BPETokenizer)
from qlm.corpus import KALITE_KAYNAKLAR, KALITE_BELGE_SAYISI      # noqa: E402
from qlm.paralel import korpus_yaz_paralel                                     # noqa: E402

VARSAYILAN_BELGE = 3_800_000
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--belge", type=int, default=VARSAYILAN_BELGE,
                  help="hedef belge sayisi")
  ap.add_argument("--vocab", type=int, default=32000)
  ap.add_argument("--cikti", default="veri", help="cikti klasoru")
  ap.add_argument("--bpe-ornek-gb", type=float, default=2.0,
                  help="BPE'nin egitilecegi ornek buyuklugu (GB)")
  ap.add_argument("--surec", type=int, default=None,
                  help="akis surec sayisi (varsayilan: cekirdek-1)")
  ap.add_argument("--tek-cekirdek", action="store_true",
                  help="paralelligi kapat (hata ayiklama)")
  ap.add_argument("--taze", action="store_true",
                  help="IKINCI TUR: tuketilmemis shard'lardan cek (Wikipedia yok)")
  ap.add_argument("--kalite", action="store_true",
                  help="UCUNCU TUR: BellaTurca kuratorlu derlemleri. Ham web "
                       "yerine duz nesir; --bpe-hazir ile birlikte kullan "
                       "(mevcut checkpoint'i surdurmek icin BPE ayni kalmali).")
  ap.add_argument("--bpe-hazir", default=None, metavar="YOL",
                  help="var olan bpe.json'i kullan (YENI BPE EGITME). Mevcut "
                       "modele veri ekliyorsan ZORUNLU: tokenizer degisirse "
                       "checkpoint kullanilamaz.")
  args = ap.parse_args()

  T0 = time.time()
  os.makedirs(args.cikti, exist_ok=True)
  korpus = os.path.join(args.cikti, "korpus.txt")
  bpe = os.path.join(args.cikti, "bpe.json")
  binp = os.path.join(args.cikti, "tokens.bin")

  def gecen():
    return f"{(time.time() - T0) / 3600:.2f} saat"


  # 1) korpus karisim + kalite fıltresi
  if not os.path.exists(korpus):
    if args.kalite:
      kaynaklar = KALITE_KAYNAKLAR
      # belğe yoksa hepsini al
      if args.belge == VARSAYILAN_BELGE:
        args.belge = KALITE_BELGE_SAYISI
      print(f"KALITE MOD: BellaTurca kuratorlu derlemler | "
            f"hedef {args.belge:,} belge")
      if not args.bpe_hazir:
        print("  UYARI: --bpe-hazir verilmedi. Yeni BPE egitilecek ve mevcut "
              "124M checkpoint KULLANILAMAZ hale gelecek. Devam etmek icin "
              "--bpe-hazir veri_taze/bpe.json ver.")
    elif args.taze:
      kaynaklar = TAZE_KAYNAKLAR
      print("TAZE MOD: tuketilmemis shard'lardan cekiliyor (Wikipedia haric)")
    else:
      kaynaklar = VARSAYILAN_KAYNAKLAR
    if args.tek_cekirdek:
      korpus_yaz(kaynaklar, korpus, hedef_belge=args.belge)
    else:
      korpus_yaz_paralel(kaynaklar, korpus, hedef_belge=args.belge,
                         surec=args.surec)
    print(f"korpus yazildi | {gecen()}")
  else:
    print(f"korpus zaten var, atlaniyor: {korpus}")

  # 2) BPE
  if args.bpe_hazir:
    import shutil
    if os.path.abspath(args.bpe_hazir) != os.path.abspath(bpe):
      shutil.copy(args.bpe_hazir, bpe)
    tok = BPETokenizer(bpe)
    print(f"HAZIR BPE kullaniliyor: {args.bpe_hazir} | vocab {len(tok)}")
  elif not os.path.exists(bpe):
    ornek = os.path.join(args.cikti, "ornek.txt")
    boyut = ornek_al(korpus, ornek, int(args.bpe_ornek_gb * 1e9))
    print(f"BPE ornegi: {boyut/1e9:.2f} GB (belge sinirina kirpildi)")
    tok = train_bpe(ornek, bpe, vocab_size=args.vocab)
    os.remove(ornek)
    print(f"BPE hazir: vocab {len(tok)} | {gecen()}")
  else:
    tok = BPETokenizer(bpe)          # modul basinda import edildi buradaki
    print(f"BPE zaten var: vocab {len(tok)}")   # yerel import onu golgeliyordu

  # 3) tokenize -> uint16 memmap
  if not os.path.exists(binp):
    # encode batch zaten paralel
    n = len(tokenize_corpus(tok, korpus, binp, add_eos=True,
                            toplu=1 if args.tek_cekirdek else 1000))
    with open(os.path.join(args.cikti, "meta.txt"), "w") as f:
      f.write(str(n))
    print(f"tokenize edildi: {n:,} token | {gecen()}")
  else:
    n = int(open(os.path.join(args.cikti, "meta.txt")).read())
    print(f"tokens.bin zaten var: {n:,} token")


  gb = os.path.getsize(binp) / 1e9
  print(f"\nHAZIR: {args.cikti}/ | tokens.bin {gb:.1f} GB | toplam {gecen()}")
  print(f"korpus.txt'yi silebilirsin: {os.path.getsize(korpus) / 1e9:.1f} GB")

if __name__ == "__main__":
  main()
