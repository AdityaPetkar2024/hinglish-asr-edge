"""Fine-tune Whisper on MUCS 2021 Hinglish jsonl splits (plain PyTorch loop, fp16 AMP).

Writes <out>/best (model + processor, chosen by dev WER), <out>/log.jsonl, <out>/summary.json.
Labels are the references as-is (mixed Devanagari/Latin), so the model learns the dataset's
own script convention.
"""
import argparse, json, math, os, random, time

import numpy as np, soundfile as sf, torch
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import jiwer
import unicodedata


def norm(t):  # identical to eval.py's normalizer
    t = unicodedata.normalize("NFC", t).lower().replace("।", " ")
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return " ".join(t.split())


class Clips(Dataset):
    def __init__(self, rows, root, fe, tok, max_label_len=448):
        self.root, self.fe, self.tok = root, fe, tok
        self.rows, self.labels = [], []
        dropped = 0
        for r in rows:
            ids = tok(r["text"]).input_ids
            if len(ids) > max_label_len:
                dropped += 1; continue
            self.rows.append(r); self.labels.append(ids)
        if dropped:
            print(f"dropped {dropped} rows with > {max_label_len} label tokens", flush=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        a, sr = sf.read(os.path.join(self.root, r["path"]), dtype="float32")
        assert sr == 16000, r["path"]
        feats = self.fe(a, sampling_rate=16000, return_tensors="np").input_features[0]
        return feats.astype(np.float32), self.labels[i], i


class Collate:
    def __init__(self, start_id):
        self.start_id = start_id

    def __call__(self, batch):
        feats = torch.from_numpy(np.stack([b[0] for b in batch]))
        labs = [b[1][1:] if b[1][0] == self.start_id else b[1] for b in batch]  # model re-adds SOT
        L = max(map(len, labs))
        labels = torch.full((len(labs), L), -100, dtype=torch.long)
        for j, l in enumerate(labs):
            labels[j, :len(l)] = torch.tensor(l)
        return feats, labels, [b[2] for b in batch]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


@torch.no_grad()
def evaluate_dev(model, loader, tok, dev, device, max_new_tokens):
    model.eval()
    preds, refs = [], []
    for feats, _, idx in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model.generate(input_features=feats.to(device), language="hindi", task="transcribe",
                                 max_new_tokens=max_new_tokens)
        preds += tok.batch_decode(out, skip_special_tokens=True)
        refs += [dev.rows[i]["text"] for i in idx]
    model.train()
    P, R = [norm(p) for p in preds], [norm(r) for r in refs]
    return jiwer.wer(R, P), list(zip(refs[:3], preds[:3]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--root", required=True, help="dir that jsonl 'path' fields are relative to")
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--batch_size", type=int, default=32, help="effective batch")
    ap.add_argument("--micro_batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--dev_n", type=int, default=0, help="0 = whole dev set")
    ap.add_argument("--eval_max_new_tokens", type=int, default=225)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--spec_augment", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert args.batch_size % args.micro_batch == 0
    accum = args.batch_size // args.micro_batch

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    proc = WhisperProcessor.from_pretrained(args.model, language="hindi", task="transcribe")
    tok, fe = proc.tokenizer, proc.feature_extractor
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "hindi"
    model.generation_config.task = "transcribe"
    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = 0.05
    model.to(device).train()

    train_rows, dev_rows = read_jsonl(args.train), read_jsonl(args.dev)
    if args.dev_n:
        dev_rows = dev_rows[:args.dev_n]
    train = Clips(train_rows, args.root, fe, tok)
    dev = Clips(dev_rows, args.root, fe, tok)
    collate = Collate(model.config.decoder_start_token_id)
    kw = dict(num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_dl = DataLoader(train, batch_size=args.micro_batch, shuffle=True, drop_last=True, **kw)
    dev_dl = DataLoader(dev, batch_size=args.micro_batch, shuffle=False, **kw)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    def lr_at(s):  # linear warmup, then linear decay to 0
        if s < args.warmup:
            return (s + 1) / args.warmup
        return max(0.0, (args.max_steps - s) / max(1, args.max_steps - args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    hours = sum(r["duration"] for r in train.rows) / 3600
    epochs = args.max_steps * args.batch_size / len(train)
    print(f"train {len(train)} utts ({hours:.2f} h), dev {len(dev)}; {args.max_steps} steps x {args.batch_size} "
          f"= {epochs:.1f} epochs; accum {accum}; device {device}", flush=True)

    log = open(f"{args.out}/log.jsonl", "a")
    best, step, t0, run_loss = math.inf, 0, time.time(), 0.0

    def do_eval():
        nonlocal best
        te = time.time()
        w, ex = evaluate_dev(model, dev_dl, tok, dev, device, args.eval_max_new_tokens)
        rec = {"step": step, "dev_wer": round(w, 4), "eval_sec": round(time.time() - te, 1),
               "elapsed_min": round((time.time() - t0) / 60, 1)}
        if w < best:
            best = w
            model.save_pretrained(f"{args.out}/best")
            proc.save_pretrained(f"{args.out}/best")
            rec["saved"] = True
        print(json.dumps(rec), flush=True)
        for r, p in ex:
            print("  REF :", r, "\n  PRED:", p, flush=True)
        log.write(json.dumps(rec) + "\n"); log.flush()

    do_eval()  # step 0 = the pretrained model on dev, for the training curve
    it = iter(train_dl)
    while step < args.max_steps:
        for _ in range(accum):
            try:
                feats, labels, _ = next(it)
            except StopIteration:
                it = iter(train_dl)
                feats, labels, _ = next(it)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(input_features=feats.to(device, non_blocking=True),
                             labels=labels.to(device, non_blocking=True)).loss / accum
            scaler.scale(loss).backward()
            run_loss += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        step += 1
        if step % 50 == 0 or step == 1:
            rec = {"step": step, "loss": round(run_loss / (1 if step == 1 else 50), 4),
                   "lr": sched.get_last_lr()[0], "grad_norm": round(float(gn), 3),
                   "sec_per_step": round((time.time() - t0) / step, 3)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            run_loss = 0.0
        if step % args.eval_every == 0 or step == args.max_steps:
            do_eval()

    summary = {"model": args.model, "train": args.train, "train_utts": len(train),
               "train_hours": round(hours, 2), "steps": args.max_steps, "batch_size": args.batch_size,
               "epochs": round(epochs, 2), "lr": args.lr, "best_dev_wer": round(best, 4),
               "train_minutes": round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=2)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
