"""Run on the LAPTOP: score the phone's transcripts and compare them with the laptop q8_0 run.
usage: python compare_phone.py phone_out_q8_0_t4.tgz"""
import json, os, sys, tarfile, tempfile
from metrics import norm
import jiwer

tgz = sys.argv[1]
d = tempfile.mkdtemp(); tarfile.open(tgz).extractall(d, filter="data")
out = os.path.join(d, os.listdir(d)[0])
q = "q5_0" if "q5_0" in os.path.basename(out) else "q8_0"
# phone runs greedy without fallback, so compare with the laptop's no-fallback run of the same model
lap = {json.loads(l)["id"]: json.loads(l) for l in open(f"results/cpp_base30h_{q}_preds.jsonl", encoding="utf-8")}
ids = [f[:-8] for f in sorted(os.listdir(out)) if f.endswith(".wav.txt")]
R, P, L, same = [], [], [], 0
for i in ids:
    p = open(os.path.join(out, i + ".wav.txt"), encoding="utf-8", errors="replace").read().strip()
    R.append(norm(lap[i]["ref"])); P.append(norm(p)); L.append(norm(lap[i]["pred"]))
    same += norm(p) == norm(lap[i]["pred"])
print(open(os.path.join(out, "device.txt")).read()); print(open(os.path.join(out, "summary.txt")).read())
print(f"clips scored            : {len(ids)}")
print(f"phone WER on these clips: {jiwer.wer(R, P):.4f}")
print(f"laptop WER, same clips  : {jiwer.wer(R, L):.4f}")
print(f"phone == laptop output  : {same}/{len(ids)} clips identical")
