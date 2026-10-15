"""Score a whisper.cpp (ggml) model on the same eval split as eval.py.

usage: python eval_cpp.py --cli ~/whisper.cpp/build/bin/whisper-cli --model models/ggml-base30h-q8_0.bin --tag cpp_base30h_q8
Greedy decoding without temperature fallback (-bs 1 -bo 1 -nf), to match eval.py's greedy runs.
Wall time includes one model load (amortized over all clips). Peak RAM comes from /usr/bin/time if present.
"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import norm, romanize, script_of, has_loop, share, utt_wer  # same metric code
import jiwer

ap = argparse.ArgumentParser()
ap.add_argument("--cli", required=True, help="path to whisper-cli (or main in old builds)")
ap.add_argument("--model", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--data", default="data/eval_300.jsonl")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--fallback", action="store_true",
                help="keep whisper.cpp's default temperature fallback (re-decodes low-entropy, i.e. looping, outputs)")
ap.add_argument("--extra", default="", help="extra whisper-cli flags, quoted")
args = ap.parse_args()

rows = [json.loads(l) for l in open(args.data, encoding="utf-8")][: args.n]
tmp = tempfile.mkdtemp()
# copy clips so the .txt outputs don't litter data/
for r in rows:
    shutil.copy(r["path"], os.path.join(tmp, r["id"] + ".wav"))
files = [os.path.join(tmp, r["id"] + ".wav") for r in rows]

cmd = [args.cli, "-m", args.model, "-l", "hi", "-t", str(args.threads), "-bs", "1", "-bo", "1",
       "-nt", "-otxt", "-np"] + ([] if args.fallback else ["-nf"]) + args.extra.split()
for f in files:
    cmd += ["-f", f]
timer = ["/usr/bin/time", "-v"] if os.path.exists("/usr/bin/time") else []
t0 = time.perf_counter()
# transcripts go to .txt files; stdout is discarded (it can contain split UTF-8 bytes)
p = subprocess.run(timer + cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
err = p.stderr.decode("utf-8", errors="replace")
elapsed = time.perf_counter() - t0
if p.returncode != 0:
    print(err[-3000:]); sys.exit("whisper-cli failed")
m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", err)
peak_mb = round(int(m.group(1)) / 1024, 1) if m else None

preds = []
for f in files:
    txt = f + ".txt"
    preds.append(open(txt, encoding="utf-8", errors="replace").read().strip() if os.path.exists(txt) else "")
shutil.rmtree(tmp)

refs = [r["text"] for r in rows]
P, R = [norm(x) for x in preds], [norm(x) for x in refs]
Pr, Rr = [romanize(x) for x in preds], [romanize(x) for x in refs]
audio_sec = sum(r["duration"] for r in rows)
metrics = {
    "tag": args.tag, "model": args.model, "engine": "whisper.cpp", "threads": args.threads,
    "decoding": "greedy + temperature fallback" if args.fallback else "greedy, no fallback",
    "model_mb": round(os.path.getsize(args.model) / 2**20, 1), "n": len(rows),
    "wer": round(jiwer.wer(R, P), 4), "cer": round(jiwer.cer(R, P), 4),
    "wer_romanized": round(jiwer.wer(Rr, Pr), 4), "cer_romanized": round(jiwer.cer(Rr, Pr), 4),
    "pred_script_share": share([script_of(x) for x in preds]),
    "loop_share_pred": round(sum(map(has_loop, preds)) / len(preds), 4),
    "empty_preds": sum(1 for x in preds if not x),
    "rtf": round(elapsed / audio_sec, 4), "peak_ram_mb": peak_mb,
}
os.makedirs("results", exist_ok=True)
with open(f"results/{args.tag}_preds.jsonl", "w", encoding="utf-8") as f:
    for r, pr, pn, rn in zip(rows, preds, P, R):
        f.write(json.dumps({"id": r["id"], "ref": r["text"], "pred": pr,
                            "wer": round(utt_wer(rn, pn), 4)}, ensure_ascii=False) + "\n")
json.dump(metrics, open(f"results/{args.tag}_metrics.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(metrics, indent=2, ensure_ascii=False))
