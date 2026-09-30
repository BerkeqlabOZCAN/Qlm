import re
import unicodedata

# oran'lar normalize ediliyor toplamin 1 olmasi gerekmıyor
VARSAYILAN_KAYNAKLAR = [
    {"path": "allenai/c4", "name": "tr", "oran": 0.70},
    {"path": "wikimedia/wikipedia", "name": "20231101.tr", "oran": 0.20},
    {"path": "HuggingFaceFW/fineweb-2", "name": "tur_Latn", "oran": 0.10},
]


# ilk turu tekrar okuma
TAZE_KAYNAKLAR = [
    {"path": "allenai/c4", "name": "tr", "oran": 0.80,
     "toplam_parca": 1024, "parca_ofset": 128},
    {"path": "HuggingFaceFW/fineweb-2", "name": "tur_Latn", "oran": 0.20,
     "toplam_parca": 30, "parca_ofset": 8, "parca_ust": 8},
]


# num shards degerleri
KALITE_KAYNAKLAR = [
    {"path": "turkish-nlp-suite/BellaTurca", "name": "OzenliDerlem",
     "oran": 0.325, "toplam_parca": 217, "parca_ust": 6},
    {"path": "turkish-nlp-suite/BellaTurca", "name": "AkademikDerlem",
     "oran": 0.157, "toplam_parca": 21, "parca_ust": 4},
    {"path": "turkish-nlp-suite/BellaTurca", "name": "temiz-OSCAR",
     "oran": 0.518, "toplam_parca": 4, "parca_ust": 4},
]
# toplam hedef
KALITE_BELGE_SAYISI = 1_388_533 + 668_109 + 2_211_181

# bunlar alt dize olarak araniyor
SPAM_KELIMELER = (
    "escort", "eskort", "porno", "sikis", "casino", "rulet", "viagra", "cialis",
    "bahis", "iddaa", "bonus veren", "deneme bonusu", "bedava bonus",
    "guvenilir site", "giris adresi", "canli mac izle", "takipci satin al",
    "slot oyun", "betting",
)
# bunlar kisa \b olmazsa "seksen" "gayet" "sohbet" de eleniyor
SPAM_KOKLER = re.compile(
    r"\b(sex|seks|gay|kumar|bet|bets|porn|kacak|jigolo|sohbet hatti)\b")
TURKCE_HARFLER = set("çğıöşüÇĞİÖŞÜ")

# belge icindeki bos satirlar ayirac sanilmasin
_IC_BOS_SATIR = re.compile(r"\n[ \t]*\n+")



def tek_belge(metin):
  return _IC_BOS_SATIR.sub("\n", metin)

def _sadelestir(metin):
  metin = metin.replace("İ", "i").replace("I", "i").lower()
  nfkd = unicodedata.normalize("NFKD", metin)
  return "".join(c for c in nfkd if not unicodedata.combining(c))

def belge_temiz_mi(metin, min_uzunluk=200, max_rakam_orani=0.15,
                   min_turkce_orani=0.005, min_satir_cesitliligi=0.5):
  metin = metin.strip()
  if len(metin) < min_uzunluk:
    return False
  sade = _sadelestir(metin)
  for kelime in SPAM_KELIMELER:
    if kelime in sade:
      return False
  if SPAM_KOKLER.search(sade):
    return False
  rakam = sum(c.isdigit() for c in metin)
  if rakam / len(metin) > max_rakam_orani:
    return False

  turkce = sum(c in TURKCE_HARFLER for c in metin)
  if turkce / len(metin) < min_turkce_orani:
    return False

  satirlar = [s.strip() for s in metin.splitlines() if s.strip()]
  if len(satirlar) > 4 and len(set(satirlar)) / len(satirlar) < min_satir_cesitliligi:
    return False

  return True

def ornek_al(korpus, cikti, bayt=2_000_000_000, sep="\n\n"):
  sep_b = sep.encode("utf-8")
  yazilan = 0
  with open(korpus, "rb") as g, open(cikti, "wb") as h:
    kalan = bayt
    while kalan > 0:
      parca = g.read(min(64 << 20, kalan))
      if not parca:
        break
      h.write(parca)
      kalan -= len(parca)
      yazilan += len(parca)

  with open(cikti, "r+b") as h:
    geri = min(yazilan, 1 << 20)
    h.seek(yazilan - geri)
    kuyruk = h.read(geri)
    yer = kuyruk.rfind(sep_b)
    if yer < 0:                               # eski CRLF korpuslar
      yer = kuyruk.rfind(b"\r\n\r\n")
    if yer >= 0:
      yazilan = yazilan - geri + yer
    else:
      # ayirac bulunamadi en azindan utf-8 karakterini bolme
      kes = len(kuyruk)
      for geri_adim in range(1, min(4, len(kuyruk)) + 1):
        b = kuyruk[-geri_adim]
        if b & 0xC0 != 0x80:
          uzunluk = 1 if b < 0x80 else 2 if b < 0xE0 else 3 if b < 0xF0 else 4
          if geri_adim < uzunluk:
            kes = len(kuyruk) - geri_adim
          break
      yazilan = yazilan - geri + kes
    h.truncate(yazilan)
  return yazilan



def _paylar(oranlar, tur_boyu=100):
  toplam = sum(oranlar) or 1.0
  pay = [max(1, round(o / toplam * tur_boyu)) for o in oranlar]
  return pay
def korpus_yaz(kaynaklar, cikti, hedef_belge, filtre=True, sep="\n\n",
               alan="text", verbose=True, log_her=100_000, **filtre_ayar):
  from datasets import load_dataset

  akislar, adlar = [], []
  for k in kaynaklar:
    ds = load_dataset(k["path"], k.get("name"), split=k.get("split", "train"),
                      streaming=True)
    akislar.append(iter(ds))
    adlar.append(k.get("path", "?") + (f"/{k['name']}" if k.get("name") else ""))
  pay = _paylar([k.get("oran", 1.0) for k in kaynaklar])

  yazilan = [0] * len(akislar)
  elenen = [0] * len(akislar)
  bitti = [False] * len(akislar)
  toplam = 0

  # windows crlf kullanmasin
  with open(cikti, "w", encoding="utf-8", newline="\n") as f:
    while toplam < hedef_belge and not all(bitti):
      for i, akis in enumerate(akislar):
        if bitti[i]:
          continue
        alinan = 0
        while alinan < pay[i] and toplam < hedef_belge:
          try:
            ornek = next(akis)
          except StopIteration:
            bitti[i] = True
            if verbose:
              print(f"  [{adlar[i]}] akis bitti ({yazilan[i]:,} belge)")
            break
          metin = (ornek.get(alan) or "").strip()
          if filtre and not belge_temiz_mi(metin, **filtre_ayar):
            elenen[i] += 1
            continue
          if not filtre and len(metin) < 50:
            elenen[i] += 1
            continue
          f.write(tek_belge(metin) + sep)
          yazilan[i] += 1
          alinan += 1
          toplam += 1
          if verbose and toplam % log_her == 0:
            print(f"  {toplam:,} belge yazildi "
                  + " | ".join(f"{a}: {y:,}" for a, y in zip(adlar, yazilan)))

  if verbose:
    print(f"korpus: {cikti} | toplam {toplam:,} belge")
    for a, y, e in zip(adlar, yazilan, elenen):
      oran = e / max(1, y + e) * 100
      print(f"  {a}: {y:,} yazildi, {e:,} elendi (%{oran:.1f})")
  return toplam
