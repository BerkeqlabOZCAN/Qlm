import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class CokParcaliVeri:

  def __init__(self, parcalar, bas=0, son=None):
    self.parcalar = list(parcalar)
    self.uzunluklar = [len(p) for p in self.parcalar]
    self.toplam = sum(self.uzunluklar)
    self.bas = max(0, bas)
    self.son = self.toplam if son is None else min(son, self.toplam)

  def __len__(self):
    return max(0, self.son - self.bas)

  def __getitem__(self, idx):
    if isinstance(idx, slice):
      b, s, adim = idx.indices(len(self))
      if adim != 1:
        raise ValueError("CokParcaliVeri sadece adim=1 dilimlemeyi destekler")
      return CokParcaliVeri(self.parcalar, self.bas + b, self.bas + s)
    if idx < 0:
      idx += len(self)
    return self._oku(self.bas + idx, self.bas + idx + 1)[0]

  def _oku(self, mutlak_bas, mutlak_son):
    dilimler, konum = [], 0
    for parca, uzunluk in zip(self.parcalar, self.uzunluklar):
      p_bas, p_son = konum, konum + uzunluk
      if mutlak_son > p_bas and mutlak_bas < p_son:
        dilimler.append(np.asarray(
            parca[max(0, mutlak_bas - p_bas):min(uzunluk, mutlak_son - p_bas)]))
      konum = p_son
      if konum >= mutlak_son:
        break
    if not dilimler:
      return np.empty(0, dtype=self.parcalar[0].dtype if self.parcalar else np.uint16)
    return dilimler[0] if len(dilimler) == 1 else np.concatenate(dilimler)

  def __array__(self, dtype=None, copy=None):
    a = self._oku(self.bas, self.son)
    return a.astype(dtype) if dtype is not None else a

class TextDataset(Dataset):

  def __init__(self, token_ids, context_length, stride, pad_id=0):
    super().__init__()
    if isinstance(token_ids, list):
      token_ids = np.asarray(token_ids, dtype=np.int32)
    self.tokens = token_ids
    self.ctx = context_length
    self.stride = stride
    # liste ram'e sigmiyor
    son = len(token_ids) - context_length - 1
    self.n = (max(0, son) // stride + 1) if son >= 0 else 1
  def __len__(self):
    return self.n


  def __getitem__(self, idx):
    s = idx * self.stride
    chunk = self.tokens[s:s + self.ctx + 1]
    if not torch.is_tensor(chunk):
      # salt okunur memmap uyarir
      chunk = torch.as_tensor(np.array(chunk), dtype=torch.long)
    else:
      chunk = chunk.long()
    x = chunk[:self.ctx]
    y = chunk[1:self.ctx + 1]
    if x.numel() < self.ctx:
      x = torch.cat([x, torch.zeros(self.ctx - x.numel(), dtype=torch.long)])
      y = torch.cat([y, torch.zeros(self.ctx - y.numel(), dtype=torch.long)])
    return x, y



def create_data_loader(token_ids, context_length, stride, batch_size,
                       shuffle=True, pad_id=0, device="cpu", num_workers=2,
                       sampler=None):
  dataset = TextDataset(token_ids, context_length, stride, pad_id)
  return DataLoader(dataset, batch_size=batch_size,
                    shuffle=False if sampler is not None else shuffle,
                    sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=str(device).startswith("cuda"))

def tokenize_corpus(tokenizer, corpus_path, bin_path, add_eos=True, sep="\n\n",
                    toplu=1000):
  toplam = 0
  yigin = []
  def bosalt(out):
    nonlocal toplam, yigin
    if not yigin:
      return
    for ids in tokenizer.encode_batch(yigin, add_eos=add_eos):
      if ids:
        np.asarray(ids, dtype=np.uint16).tofile(out)
        toplam += len(ids)
    yigin = []

  with open(bin_path, "wb") as out:
    buf = ""
    with open(corpus_path, "r", encoding="utf-8") as f:
      for line in f:
        buf += line
        while sep in buf:
          belge, buf = buf.split(sep, 1)
          belge = belge.strip()
          if belge:
            yigin.append(belge)
            if len(yigin) >= max(1, toplu):
              bosalt(out)
    belge = buf.strip()
    if belge:
      yigin.append(belge)
    bosalt(out)
  return np.memmap(bin_path, dtype=np.uint16, mode="r", shape=(toplam,))

def read_documents(path, sep="\n\n"):
  with open(path, "r", encoding="utf-8") as f:
    return [d for d in f.read().split(sep) if d.strip()]

def tokenize_documents(tokenizer, texts, add_eos=True):
  import numpy as np
  parcalar = []
  for t in texts:
    ids = tokenizer.encode(t, add_eos=add_eos)
    if ids:
      parcalar.append(np.asarray(ids, dtype=np.int32))
  if not parcalar:
    return torch.zeros(0, dtype=torch.int32)
  return torch.from_numpy(np.concatenate(parcalar))
