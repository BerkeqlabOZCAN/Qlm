import torch



@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens=30,
             temperature=0.8, top_k=None, top_p=0.9, repetition_penalty=1.3):
  model.eval()
  ids = tokenizer.encode(prompt)
  prompt_len = len(ids)
  ctx = model.context_length


  for _ in range(max_tokens):
    girdi = torch.tensor(ids[-ctx:], device=model.device).unsqueeze(0)
    logits = model(girdi)[0, -1]

    # temperature'dan once
    if repetition_penalty != 1.0 and ids:
      gecmis = torch.tensor(list(set(ids)), device=logits.device)
      degerler = logits[gecmis]
      logits[gecmis] = torch.where(degerler > 0,
                                   degerler / repetition_penalty,
                                   degerler * repetition_penalty)

    logits = logits / max(temperature, 1e-6)

    if top_k is not None:
      esik = torch.topk(logits, top_k).values[-1]
      logits = torch.where(logits < esik, torch.tensor(float("-inf"), device=logits.device), logits)

    probs = torch.softmax(logits, dim=-1)

    if top_p is not None:
      sirali, idx = torch.sort(probs, descending=True)
      kumulatif = torch.cumsum(sirali, dim=-1)
      sirali[kumulatif - sirali > top_p] = 0.0
      sirali = sirali / sirali.sum()
      secim = idx[torch.multinomial(sirali, 1)].item()
    else:
      secim = torch.multinomial(probs, 1).item()

    if secim == tokenizer.eos_id:
      break
    ids.append(secim)

  return tokenizer.decode(ids[prompt_len:])
