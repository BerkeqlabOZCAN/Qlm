"""Egitim durumunu Telegram'dan haber verir"""
import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
AYAR_DOSYA = os.path.join(os.path.expanduser("~"), ".telegram", "qlm.json")


def _env_oku(yol):
  """Basit env ayristirici (KEY=value)"""
  d = {}
  with open(yol, encoding="utf-8", errors="ignore") as f:
    for satir in f:
      satir = satir.strip()
      if satir.startswith("#") or "=" not in satir:
        continue
      k, v = satir.split("=", 1)
      d[k.strip()] = v.strip().strip('"').strip("'")
  return d


def ayar(env_dosya=None):
  """Token/chat_id bulur Deger hicbir zaman ekrana basilmaz"""
  token = os.environ.get("TELEGRAM_BOT_TOKEN")
  chat = os.environ.get("TELEGRAM_CHAT_ID")
  d = {}
  if os.path.exists(AYAR_DOSYA):
    with open(AYAR_DOSYA, encoding="utf-8") as f:
      d = json.load(f)
  env_dosya = env_dosya or d.get("env_dosya")
  if env_dosya and os.path.exists(env_dosya):
    e = _env_oku(env_dosya)
    token = token or e.get("TELEGRAM_BOT_TOKEN")
    chat = chat or e.get("TELEGRAM_CHAT_ID")
  token = token or d.get("token")
  chat = chat or d.get("chat_id")
  return token, chat


def istek(token, metot, veri=None):
  url = f"https://api.telegram.org/bot{token}/{metot}"
  gonderi = urllib.parse.urlencode(veri).encode() if veri else None
  with urllib.request.urlopen(urllib.request.Request(url, data=gonderi), timeout=20) as r:
    return json.load(r)



def gonder(mesaj, env_dosya=None):
  token, chat = ayar(env_dosya)
  if not token or not chat:
    print(f"Telegram ayari yok. {AYAR_DOSYA} olustur ya da ortam degiskenlerini ver.\n"
          "Ayrinti icin: python scripts/haber_ver.py --help", file=sys.stderr)
    return 1
  try:
    c = istek(token, "sendMessage", {"chat_id": chat, "text": mesaj,
                                     "disable_web_page_preview": "true"})
  except Exception as e:
    print(f"Telegram gonderilemedi: {type(e).__name__}: {e}", file=sys.stderr)
    return 1
  print("Telegram: gonderildi" if c.get("ok") else f"Telegram hatasi: {c}")
  return 0 if c.get("ok") else 1


def chat_id_bul(env_dosya=None):
  token, _ = ayar(env_dosya)
  if not token:
    sys.exit("Once bot token'i gerek (ayar dosyasi ya da TELEGRAM_BOT_TOKEN).")
  c = istek(token, "getUpdates")
  bulunan = []
  for g in c.get("result", []):
    sohbet = (g.get("message") or g.get("channel_post") or {}).get("chat") or {}
    if sohbet.get("id") and sohbet["id"] not in [b[0] for b in bulunan]:
      bulunan.append((sohbet["id"], sohbet.get("username") or sohbet.get("title") or "?"))
  if not bulunan:
    sys.exit("Hic mesaj yok. Telegram'da botuna /start yaz, sonra tekrar calistir.")
  print("Bulunan sohbetler (chat_id -> kim):")
  for cid, kim in bulunan:
    print(f"  {cid}  ->  {kim}")
  print(f"\nDogru olani {AYAR_DOSYA} icindeki \"chat_id\" alanina yaz.")

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("mesaj", nargs="?", default=None)
  ap.add_argument("--test", action="store_true", help="deneme mesaji gonder")
  ap.add_argument("--chat-id-bul", action="store_true")
  ap.add_argument("--env", default=None, help="TELEGRAM_* iceren .env yolu")
  args = ap.parse_args()

  if args.chat_id_bul:
    return chat_id_bul(args.env)
  if args.test:
    return gonder("qlm: test mesaji — Telegram bildirimleri calisiyor.", args.env)
  if not args.mesaj:
    ap.error("mesaj gerekli (ya da --test / --chat-id-bul)")
  return gonder(args.mesaj, args.env)

if __name__ == "__main__":
  sys.exit(main() or 0)
