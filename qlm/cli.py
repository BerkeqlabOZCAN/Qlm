import argparse
import sys

from .config import ModelConfig, TrainConfig
from .tokenizer import BPETokenizer, train_bpe
from .model import LanguageModel
from .trainer import train
from .generate import generate


def _read(path):
  with open(path, "r", encoding="utf-8") as f:
    return f.read()


def main(argv=None):
  parser = argparse.ArgumentParser(prog="qlm", description="qlm dil modeli araclari")
  sub = parser.add_subparsers(dest="komut", required=True)

  p_bpe = sub.add_parser("train-bpe", help="metin dosyalarindan BPE tokenizer egit")
  p_bpe.add_argument("files", nargs="+")
  p_bpe.add_argument("--out", default="bpe.json")
  p_bpe.add_argument("--vocab-size", type=int, default=4000)


  p_tr = sub.add_parser("train", help="model egit")
  p_tr.add_argument("corpus")
  p_tr.add_argument("--bpe", default="bpe.json")
  p_tr.add_argument("--out", default="model_weights.pt")
  p_tr.add_argument("--epochs", type=int, default=100)
  p_tr.add_argument("--embedding-dim", type=int, default=64)
  p_tr.add_argument("--layers", type=int, default=4)
  p_tr.add_argument("--heads", type=int, default=4)

  p_ch = sub.add_parser("chat", help="egitilmis modelle sohbet")
  p_ch.add_argument("--bpe", default="bpe.json")
  p_ch.add_argument("--model", default="model_weights.pt")
  p_ch.add_argument("--temperature", type=float, default=0.3)

  args = parser.parse_args(argv)

  if args.komut == "train-bpe":
    tok = train_bpe(args.files, args.out, vocab_size=args.vocab_size)
    print(f"BPE egitildi -> {args.out} | vocab: {len(tok)}")

  elif args.komut == "train":
    tok = BPETokenizer(args.bpe)
    tokens = tok.encode(_read(args.corpus), add_eos=True)
    cfg = ModelConfig(vocab_size=len(tok), embedding_dim=args.embedding_dim,
                      num_layers=args.layers, num_heads=args.heads)
    train(tokens, tok, cfg, TrainConfig(epochs=args.epochs), save_path=args.out)

  elif args.komut == "chat":
    tok = BPETokenizer(args.bpe)
    model = LanguageModel.load(args.model)
    print("qlm sohbet — cikmak icin 'cik'")
    while True:
      try:
        prompt = input("Siz : ").strip()
      except (EOFError, KeyboardInterrupt):
        break
      if prompt.lower() in ("cik", "çık", "exit", "quit"):
        break
      if prompt:
        print("Bot :", generate(model, tok, prompt, temperature=args.temperature))
if __name__ == "__main__":
  main(sys.argv[1:])
