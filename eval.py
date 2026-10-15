"""Evaluate a Whisper checkpoint on a fixed jsonl split (MUCS 2021 Hinglish).

Corpus-level metrics:
  wer / cer                    Devanagari-safe normalization (NFC, lowercase, strip P*/S*)
  wer_romanized / cer_rom.     Devanagari words transliterated to ITRANS first
  pred_script_share            majority script of each prediction (devanagari/latin/arabic/other/empty)
  loop_share_pred / _ref       share of texts containing a repetition loop (heuristic; _ref = false-positive check)
  rtf                          wall time / audio time on --device, batched, after warm-up
"""
import argparse, json, os, time

import evaluate, jiwer, soundfile as sf, torch
from transformers import pipeline


from metrics import norm, is_deva, romanize, script_of, has_loop, share, utt_wer  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-base")
    ap.add_argument("--data", default="data/eval_300.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4, help="CPU only")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=440,
                    help="decoder has 448 positions incl. 4 prompt tokens; Devanagari costs "
                         "several byte-BPE tokens per word, so a low cap truncates real output")
    ap.add_argument("--no_repeat_ngram", type=int, default=0, help="0 = off (plain greedy)")
    ap.add_argument("--tag", default=None, help="output prefix; defaults to the model name")
    args = ap.parse_args()

    tag = args.tag or args.model.strip("/").replace("/", "_")
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32

    rows_all = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]
    # jiwer rejects empty references; drop them identically for every model
    rows = [r for r in rows_all if norm(r["text"]) and romanize(r["text"])]
    n_dropped = len(rows_all) - len(rows)

    asr = pipeline("automatic-speech-recognition", model=args.model,
                   device=args.device, torch_dtype=dtype)
    gen = {"language": "hindi", "task": "transcribe", "max_new_tokens": args.max_new_tokens}
    if args.no_repeat_ngram:
        gen["no_repeat_ngram_size"] = args.no_repeat_ngram

    def load(path):
        a, sr = sf.read(path, dtype="float32")
        assert sr == 16000 and a.ndim == 1, path
        return {"raw": a, "sampling_rate": sr}

    # warm-up on fresh dicts (the pipeline may pop keys from its inputs)
    asr([load(rows[0]["path"]), load(rows[1]["path"])], batch_size=2, generate_kwargs=gen)

    audio = [load(r["path"]) for r in rows]
    t0 = time.perf_counter()
    out = asr(audio, batch_size=args.batch_size, generate_kwargs=gen)
    elapsed = time.perf_counter() - t0

    preds = [o["text"] for o in out]
    refs = [r["text"] for r in rows]
    P, R = [norm(p) for p in preds], [norm(r) for r in refs]
    Pr, Rr = [romanize(p) for p in preds], [romanize(r) for r in refs]
    audio_sec = sum(r["duration"] for r in rows)

    wer, cer = evaluate.load("wer"), evaluate.load("cer")
    metrics = {
        "tag": tag, "model": args.model, "device": args.device, "dtype": str(dtype).split(".")[-1],
        "n": len(rows), "n_dropped_empty_ref": n_dropped,
        "audio_min": round(audio_sec / 60, 2),
        "decoding": {k: v for k, v in gen.items()} | {"batch_size": args.batch_size},
        "wer": round(wer.compute(predictions=P, references=R), 4),
        "cer": round(cer.compute(predictions=P, references=R), 4),
        "wer_romanized": round(wer.compute(predictions=Pr, references=Rr), 4),
        "cer_romanized": round(cer.compute(predictions=Pr, references=Rr), 4),
        "pred_script_share": share([script_of(p) for p in preds]),
        "ref_script_share": share([script_of(r) for r in refs]),
        "loop_share_pred": round(sum(map(has_loop, preds)) / len(preds), 4),
        "loop_share_ref": round(sum(map(has_loop, refs)) / len(refs), 4),
        "rtf": round(elapsed / audio_sec, 4),
    }

    os.makedirs("results", exist_ok=True)
    with open(f"results/{tag}_preds.jsonl", "w", encoding="utf-8") as f:
        for r, p, pn, rn, pr, rr in zip(rows, preds, P, R, Pr, Rr):
            f.write(json.dumps({
                "id": r["id"], "speaker": r["speaker"], "duration": r["duration"],
                "ref": r["text"], "pred": p,
                "wer": round(utt_wer(rn, pn), 4), "wer_romanized": round(utt_wer(rr, pr), 4),
                "pred_script": script_of(p), "loop": has_loop(p),
            }, ensure_ascii=False) + "\n")
    with open(f"results/{tag}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    for r, p in list(zip(refs, preds))[:5]:
        print("\nREF :", r, "\nPRED:", p)


if __name__ == "__main__":
    main()
