# Qlm

Sifirdan yazdigim kucuk bir Turkce dil modeli. Tokenizer disinda hazir transformer kutuphanesi kullanmadim, attention, RoPE, egitim dongusu vs hepsi elle yazildi.

- ByteLevel BPE (32k vocab)
- RoPE, SDPA attention, gated MLP, pre-LN
- AMP, warmup + cosine LR, grad accum, DDP
- checkpoint ve kaldigi yerden devam etme destegi

## Kurulum

```
pip install torch tokenizers datasets numpy
```

## Klasorler

```
qlm/     paket
scripts/     veri hazirlama ve yerel test araclari
```

## Kullanim

```python
from qlm import BPETokenizer, train_bpe, ModelConfig, train, generate

tok = train_bpe("metin.txt", "bpe.json", vocab_size=4000)
tokens = tok.encode(open("metin.txt", encoding="utf-8").read(), add_eos=True)
model = train(tokens, tok, ModelConfig(vocab_size=len(tok)))
print(generate(model, tok, "türkiye'nin başkenti"))
```

Yerel sohbet testi: `python scripts/sohbet.py`
