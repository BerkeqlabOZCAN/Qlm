"""Yerel sohbet ve uretim testi"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qlm import BPETokenizer, LanguageModel, generate   # noqa: E402

# sabit sizinti testleri
TEST_PROMPTLARI = [
    "turkiye", "ankara", "yapay zeka", "saglik bakanligi",
    "en onemli", "bu konuda", "dun aksam", "bilim insanlari",
]
def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="modeller/llm_tr.pt")
  ap.add_argument("--bpe", default="modeller/bpe.json")
  ap.add_argument("--temperature", type=float, default=0.8)
  ap.add_argument("--top-p", type=float, default=0.9)
  ap.add_argument("--repetition-penalty", type=float, default=1.3)
  ap.add_argument("--max-tokens", type=int, default=60)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--test", action="store_true", help="sabit promptlarla kalite testi")
  args = ap.parse_args()

  for yol in (args.model, args.bpe):
    if not os.path.exists(yol):
      print(f"bulunamadi: {yol}\nmodel dosyalarini modeller/ altina koy")
      return 1
  tok = BPETokenizer(args.bpe)
  model = LanguageModel.load(args.model, device=args.device)
  print(f"model: {args.model} | parametre: {model.num_parameters():,} | vocab: {len(tok)}")


  def uret(p):
    return generate(model, tok, p, max_tokens=args.max_tokens,
                    temperature=args.temperature, top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty)
  if args.test:
    for p in TEST_PROMPTLARI:
      print(f"\n[{p}] -> {uret(p)}")
    return 0

  print("sohbet — cikmak icin 'cik'")
  while True:
    try:
      p = input("\nSiz : ").strip()
    except (EOFError, KeyboardInterrupt):
      break
    if p.lower() in ("cik", "çık", "exit", "quit"):
      break
    if p:
      print("Bot :", uret(p))
  return 0


if __name__ == "__main__":
  sys.exit(main())
