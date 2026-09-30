import os
from multiprocessing import Pool

from .corpus import VARSAYILAN_KAYNAKLAR, _paylar, belge_temiz_mi, tek_belge

SEP = "\n\n"


def _parca_yaz(arg):
  kaynak, cikti, hedef, parca_listesi, toplam_parca, alan, filtre, filtre_ayar = arg
  from datasets import load_dataset
  yazilan = elenen = 0
  # load_dataset'i shard basina cagirinca HF 429 donuyordu bir kez cagirıyoruz
  temel = load_dataset(kaynak["path"], kaynak.get("name"),
                       split=kaynak.get("split", "train"), streaming=True)
  with open(cikti, "w", encoding="utf-8", newline="\n") as f:
    for parca_no in parca_listesi:
      if yazilan >= hedef:
        break
      ds = temel
      if toplam_parca > 1:
        try:
          ds = temel.shard(num_shards=toplam_parca, index=parca_no)
        except Exception:                    # shard desteklenmiyor
          if parca_no != parca_listesi[0]:
            break
      for ornek in ds:
        if yazilan >= hedef:
          break
        metin = (ornek.get(alan) or "").strip()
        if (filtre and not belge_temiz_mi(metin, **filtre_ayar)) or len(metin) < 50:
          elenen += 1
          continue
        f.write(tek_belge(metin) + SEP)
        yazilan += 1
  return (cikti, yazilan, elenen)

def _belgeler(yollar):
  for yol in yollar:
    if not os.path.exists(yol):
      continue
    tampon = ""
    with open(yol, encoding="utf-8") as f:
      for satir in f:
        tampon += satir
        while SEP in tampon:
          belge, tampon = tampon.split(SEP, 1)
          if belge.strip():
            yield belge.strip()
    if tampon.strip():
      yield tampon.strip()


def _surec_dagit(oranlar, toplam_surec, ust_sinirlar):
  toplam = sum(oranlar) or 1.0
  dagilim = []
  for o, ust in zip(oranlar, ust_sinirlar):
    dagilim.append(max(1, min(int(round(toplam_surec * o / toplam)), ust)))
  return dagilim


def korpus_yaz_paralel(kaynaklar=None, cikti="korpus.txt", hedef_belge=1_000_000,
                       surec=None, alan="text", filtre=True, verbose=True,
                       toplam_parca=1024, parca_ofset=0, **filtre_ayar):
  kaynaklar = kaynaklar or VARSAYILAN_KAYNAKLAR
  surec = surec or max(1, (os.cpu_count() or 2) - 1)
  oranlar = [k.get("oran", 1.0) for k in kaynaklar]
  toplam_oran = sum(oranlar) or 1.0
  hedefler = [max(1, int(hedef_belge * o / toplam_oran)) for o in oranlar]
  ust = [k.get("parca_ust", 64) for k in kaynaklar]
  dagilim = _surec_dagit(oranlar, surec, ust)
  if verbose:
    print(f"paralel akis: {surec} surec | hedefler {hedefler} | "
          f"kaynak basina surec {dagilim} | shard {parca_ofset}..{toplam_parca}")

  isler, kaynak_dosyalari = [], []
  for i, (k, hedef, n) in enumerate(zip(kaynaklar, hedefler, dagilim)):
    dosyalar = []
    kaynak_parca = k.get("toplam_parca", toplam_parca)
    kaynak_ofset = k.get("parca_ofset", parca_ofset)
    for j in range(n):
      yol = f"{cikti}.k{i}p{j}"
      dosyalar.append(yol)
      # ofset+j ofset+j+n ofset+j+2n
      parca_listesi = list(range(kaynak_ofset + j, kaynak_parca, n))
      isler.append((k, yol, hedef // n + 1, parca_listesi, kaynak_parca,
                    alan, filtre, filtre_ayar))
    kaynak_dosyalari.append(dosyalar)

  with Pool(processes=len(isler)) as havuz:
    sonuc = havuz.map(_parca_yaz, isler)
  if verbose:
    bilgi = {yol: (y, e) for yol, y, e in sonuc}
    for k, dosyalar in zip(kaynaklar, kaynak_dosyalari):
      y = sum(bilgi[d][0] for d in dosyalar)
      e = sum(bilgi[d][1] for d in dosyalar)
      print(f"  {k['path']}: {y:,} yazildi, {e:,} elendi "
            f"(%{e / max(1, y + e) * 100:.1f})")

  pay = _paylar(oranlar)
  akislar = [_belgeler(d) for d in kaynak_dosyalari]
  bitti = [False] * len(akislar)
  yazilan = 0
  with open(cikti, "w", encoding="utf-8", newline="\n") as f:
    while not all(bitti) and yazilan < hedef_belge:
      for i, akis in enumerate(akislar):
        if bitti[i]:
          continue
        for _ in range(pay[i]):
          try:
            f.write(next(akis) + SEP)
            yazilan += 1
          except StopIteration:
            bitti[i] = True
            break
  # kapatmazsak dosyalar acik kaliyor Windows silmeye izin vermiyor
  for akis in akislar:
    akis.close()
  for dosyalar in kaynak_dosyalari:
    for d in dosyalar:
      try:
        if os.path.exists(d):
          os.remove(d)
      except OSError as e:
        print(f"  uyari: {d} silinemedi ({type(e).__name__})")
  if verbose:
    print(f"korpus: {cikti} | {yazilan:,} belge (serpistirildi)")
  return yazilan
