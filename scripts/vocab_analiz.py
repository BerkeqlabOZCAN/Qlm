"""Vocab boyutu analizi Turkce icin kac BPE token'i optimal"""
import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from qlm import train_bpe   # noqa: E402



def blok_parametre(d):
  """Bir transformer blogunun parametre sayisi (attention + gated MLP + 2 norm)"""
  gizli = int(2 * 4 * d / 3)
  return 4 * d * d + 3 * d * gizli + 4 * d

def analiz(metin, vocab, dosyalar, d, katman, gecici, baglam=512):
  yol = os.path.join(gecici, f"bpe_{vocab}.json")
  tok = train_bpe(dosyalar, yol, vocab_size=vocab)
  ids = tok.encode(metin)

  karakter = len(metin)
  kelime = len(metin.split())
  tk = len(ids) / karakter          # token / karakter
  gercek_vocab = len(tok)

  emb = gercek_vocab * d
  govde = katman * blok_parametre(d) + 2 * d
  toplam = emb + govde

  # token basina flop govde vs lm_head
  head_flop = 2 * d * gercek_vocab
  govde_flop = 2 * govde + 4 * katman * baglam * d
  return {
      "vocab": vocab,
      "gercek": gercek_vocab,
      "dolu": gercek_vocab >= vocab * 0.99,
      "tok_kar": tk,
      "tok_kel": len(ids) / max(1, kelime),
      "emb_m": emb / 1e6,
      "toplam_m": toplam / 1e6,
      "emb_pay": emb / toplam * 100,
      "head_pay": head_flop / (head_flop + govde_flop) * 100,
      "baglam_kar": int(512 / tk),
  }

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("korpus", help="analiz edilecek metin dosyasi")
  ap.add_argument("--vocablar", type=int, nargs="+",
                  default=[8000, 16000, 24000, 32000, 48000])
  ap.add_argument("--dim", type=int, default=640, help="embedding_dim")
  ap.add_argument("--katman", type=int, default=8, help="num_layers")
  ap.add_argument("--ornek", type=int, default=0,
                  help="olcum icin dosyanin ilk N MB'i (0 = tamami)")
  args = ap.parse_args()

  boyut = os.path.getsize(args.korpus) / 1e6
  print(f"korpus: {args.korpus} ({boyut:.1f} MB)")
  if boyut < 50:
    print("UYARI: korpus kucuk (<50MB); buyuk vocab'lar dolmayabilir, "
          "sonuclari yon gostergesi olarak oku.\n")
  with open(args.korpus, encoding="utf-8", errors="ignore") as f:
    metin = f.read(int(args.ornek * 1e6)) if args.ornek else f.read()

  gecici = tempfile.mkdtemp()
  satirlar = []
  for v in args.vocablar:
    s = analiz(metin, v, [args.korpus], args.dim, args.katman, gecici)
    satirlar.append(s)
    print(f"  vocab {v:>6,} olculdu (gercek {s['gercek']:,})")

  taban = satirlar[-1]["tok_kar"]
  print(f"\nd={args.dim}, {args.katman} katman | govde "
        f"{args.katman * blok_parametre(args.dim) / 1e6:.1f}M parametre (vocab'tan bagimsiz)\n")
  print(f"{'vocab':>7} {'tok/kar':>8} {'tok/kel':>8} {'emb':>8} {'toplam':>8} "
        f"{'emb%':>6} {'head%':>6} {'512tok=kar':>11} {'token+':>7}")
  print("-" * 78)
  for s in satirlar:
    fark = (s["tok_kar"] / taban - 1) * 100   # en buyuk vocab'a gore fazladan token
    isaret = "" if s["dolu"] else "  (dolmadi!)"
    print(f"{s['vocab']:>7,} {s['tok_kar']:>8.3f} {s['tok_kel']:>8.2f} "
          f"{s['emb_m']:>7.1f}M {s['toplam_m']:>7.1f}M {s['emb_pay']:>5.1f}% "
          f"{s['head_pay']:>5.1f}% {s['baglam_kar']:>11,} {fark:>+6.1f}%{isaret}")


  print("\nOkuma notlari:")
  print("  tok/kar dusuk = ayni metin daha az token = daha ucuz egitim, daha genis baglam")
  print("  emb%  yuksek = parametrelerin cogu sozluk tablosunda, govdede degil")
  print("  head% yuksek = hesabin cogu son softmax matmul'unde (kucuk modelde ciddi)")
  print("  token+ = en buyuk vocab'a kiyasla ayni metin icin fazladan token orani")
  print("  ZORUNLU: vocab < 65.536 (tokens.bin uint16 memmap)")



if __name__ == "__main__":
  main()
