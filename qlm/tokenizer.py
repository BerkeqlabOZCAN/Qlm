from tokenizers import Tokenizer as HFTokenizer
from tokenizers import models, trainers, pre_tokenizers, decoders, normalizers

SPECIAL_TOKENS = ["<unk>", "<pad>", "<eos>"]

def train_bpe(files, save_path="bpe.json", vocab_size=4000, min_frequency=2,
              nfc=True):
  if isinstance(files, str):
    files = [files]

  tok = HFTokenizer(models.BPE(unk_token="<unk>"))
  if nfc:
    tok.normalizer = normalizers.NFC()
  tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
  tok.decoder = decoders.ByteLevel()

  trainer = trainers.BpeTrainer(
      vocab_size=vocab_size,
      special_tokens=SPECIAL_TOKENS,
      min_frequency=min_frequency,
      show_progress=False,
      # nadir karakterleri kaybetme
      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
  )
  tok.train(files, trainer)
  tok.save(save_path)
  return BPETokenizer(save_path)


class BPETokenizer:

  def __init__(self, path="bpe.json"):
    self.tok = HFTokenizer.from_file(path)
    self.eos_id = self.tok.token_to_id("<eos>")
    self.pad_id = self.tok.token_to_id("<pad>")
    self.vocab = self.tok.get_vocab()

  def __len__(self):
    return self.tok.get_vocab_size()

  @property
  def vocab_size(self):
    return self.tok.get_vocab_size()

  def encode(self, text, add_eos=False):
    ids = self.tok.encode(text).ids
    if add_eos and self.eos_id is not None:
      ids = ids + [self.eos_id]
    return ids
  def encode_batch(self, texts, add_eos=False):
    kodlar = self.tok.encode_batch(list(texts))
    if add_eos and self.eos_id is not None:
      return [k.ids + [self.eos_id] for k in kodlar]
    return [k.ids for k in kodlar]

  def decode(self, ids):
    return self.tok.decode(ids)
