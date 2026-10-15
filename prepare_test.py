import json, os, random
import soundfile as sf

BASE = "data/test"
TR = f"{BASE}/transcripts"
OUT = "data/test_segments"
os.makedirs(OUT, exist_ok=True)

def read_map(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d

text = read_map(f"{TR}/text")
wavscp = read_map(f"{TR}/wav.scp")
utt2spk = read_map(f"{TR}/utt2spk")

segs = []
with open(f"{TR}/segments", encoding="utf-8") as f:
    for line in f:
        uid, rec, start, end = line.split()
        segs.append((uid, rec, float(start), float(end)))

cache, rows = {}, []
skipped = {"no_text": 0, "too_short": 0, "too_long": 0}
for uid, rec, start, end in segs:
    t = text.get(uid, "").strip()
    dur = end - start
    if not t:
        skipped["no_text"] += 1; continue
    if dur < 1.0:
        skipped["too_short"] += 1; continue
    if dur > 30.0:
        skipped["too_long"] += 1; continue
    if rec not in cache:
        cache[rec] = sf.read(os.path.join(BASE, wavscp[rec]))
    audio, sr = cache[rec]
    clip = audio[int(start * sr):int(end * sr)]
    path = f"{OUT}/{uid}.wav"
    sf.write(path, clip, sr)
    rows.append({"id": uid, "path": path, "text": t,
                 "speaker": utt2spk.get(uid, "?"), "duration": round(dur, 2)})

with open("data/test_all.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

random.seed(0)
subset = random.sample(rows, min(300, len(rows)))
with open("data/eval_300.jsonl", "w", encoding="utf-8") as f:
    for r in subset:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

hours = sum(r["duration"] for r in rows) / 3600
print(f"segments kept: {len(rows)} ({hours:.2f} h)  skipped: {skipped}")
print(f"speakers: {len(set(r['speaker'] for r in rows))}")
print(f"eval subset: {len(subset)} clips, "
      f"{len(set(r['speaker'] for r in subset))} speakers, "
      f"{sum(r['duration'] for r in subset)/60:.1f} min")
