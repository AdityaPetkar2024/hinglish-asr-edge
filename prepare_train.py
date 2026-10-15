"""Build training subsets from MUCS 2021 Hinglish train (OpenSLR 104).

- same filters as the test prep: non-empty text, 1 s <= duration <= 30 s
- checks overlap with the test set: recording ids, recording audio (md5),
  speaker ids, and exact long transcripts; overlapping material is excluded
- dev split is speaker-disjoint (whole speakers held out), like the test set
- train subsets are nested: the 8 h set is a prefix of the 30 h set
- only the clips that are actually used get cut to disk
Writes <out>/clips/*.wav, train_<H>h.jsonl, dev.jsonl, stats.json (paths relative to <out>).
"""
import argparse, collections, hashlib, json, os, random, unicodedata

import soundfile as sf


def read_map(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                d[parts[0]] = parts[1]
    return d


def norm(t):
    t = unicodedata.normalize("NFC", t).lower()
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return " ".join(t.split())


class Kaldi:
    def __init__(self, root):
        self.root = root
        self.dir = next((d for d, _, fs in os.walk(root) if "segments" in fs and "text" in fs), None)
        if self.dir is None:
            raise SystemExit(f"no Kaldi dir (segments + text) under {root}")
        self.text = read_map(f"{self.dir}/text")
        self.wavscp = read_map(f"{self.dir}/wav.scp")
        self.utt2spk = read_map(f"{self.dir}/utt2spk")
        self.segs = []
        with open(f"{self.dir}/segments", encoding="utf-8") as f:
            for line in f:
                uid, rec, s, e = line.split()
                self.segs.append((uid, rec, float(s), float(e)))
        self._index = None

    def wav(self, rec):
        p = self.wavscp[rec]
        for base in (os.path.dirname(self.dir), self.root):
            q = os.path.join(base, p)
            if os.path.exists(q):
                return q
        if self._index is None:  # fall back to a basename lookup
            self._index = {f: os.path.join(d, f) for d, _, fs in os.walk(self.root)
                           for f in fs if f.lower().endswith((".wav", ".flac"))}
        q = self._index.get(os.path.basename(p))
        if q is None:
            raise SystemExit(f"cannot resolve audio for {rec}: {p}")
        return q


def md5_file(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def hours(rows):
    return round(sum(r["duration"] for r in rows) / 3600, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", default="/tmp/mucs_train")
    ap.add_argument("--test_root", default="data/test")
    ap.add_argument("--out", default="mucs_prep")
    ap.add_argument("--hours", type=float, nargs="+", default=[30, 8])
    ap.add_argument("--dev_hours", type=float, default=1.0)
    ap.add_argument("--min_dup_words", type=int, default=5,
                    help="exclude train utts whose normalized text equals a test transcript of >= this many words")
    ap.add_argument("--keep_test_speakers", action="store_true",
                    help="keep train utts from speaker ids that also appear in test (default: drop them)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, test = Kaldi(args.train_root), Kaldi(args.test_root)
    stats = {"train_kaldi_dir": train.dir, "test_kaldi_dir": test.dir}

    # 1. filters
    utts, skipped = [], collections.Counter()
    for uid, rec, s, e in train.segs:
        t, dur = train.text.get(uid, "").strip(), e - s
        if not t or not norm(t):
            skipped["no_text"] += 1; continue
        if dur < 1.0:
            skipped["too_short"] += 1; continue
        if dur > 30.0:
            skipped["too_long"] += 1; continue
        utts.append({"id": uid, "recording": rec, "start": s, "end": e, "text": t,
                     "speaker": train.utt2spk.get(uid, rec), "duration": round(dur, 2)})
    stats["train_after_filters"] = {"utts": len(utts), "hours": hours(utts),
                                    "speakers": len({u["speaker"] for u in utts}),
                                    "recordings": len({u["recording"] for u in utts}),
                                    "skipped": dict(skipped)}
    print("after filters:", stats["train_after_filters"])

    # 2. overlap with test
    test_md5 = {md5_file(test.wav(r)): r for r in test.wavscp}
    train_recs = sorted({u["recording"] for u in utts})
    rec_audio_dup = {r for r in train_recs if md5_file(train.wav(r)) in test_md5}
    rec_id_dup = set(train_recs) & set(test.wavscp)
    test_spk = set(test.utt2spk.values())
    spk_dup = {u["speaker"] for u in utts} & test_spk
    test_text = {norm(t) for t in test.text.values() if len(norm(t).split()) >= args.min_dup_words}
    text_dup = {u["id"] for u in utts if norm(u["text"]) in test_text}

    def why_excluded(u):
        if u["recording"] in rec_audio_dup: return "same_audio_as_test"
        if u["recording"] in rec_id_dup: return "same_recording_id_as_test"
        if u["speaker"] in spk_dup and not args.keep_test_speakers: return "test_speaker"
        if u["id"] in text_dup: return "long_transcript_in_test"
        return None

    excl = collections.defaultdict(list)
    clean = []
    for u in utts:
        w = why_excluded(u)
        (excl[w].append(u) if w else clean.append(u))
    stats["overlap"] = {
        "recordings_same_audio_md5": sorted(rec_audio_dup),
        "recording_ids_shared": sorted(rec_id_dup),
        "speaker_ids_shared": sorted(spk_dup),
        "utts_long_transcript_in_test": len(text_dup),
        "excluded": {k: {"utts": len(v), "hours": hours(v)} for k, v in excl.items()},
        "note": "speaker-id overlap is only meaningful if both splits use the same id scheme; "
                "audio md5 is the hard check for duplicated recordings",
    }
    print("overlap:", json.dumps(stats["overlap"], ensure_ascii=False)[:1500])

    # 3. speaker-disjoint dev
    rng = random.Random(args.seed)
    by_spk = collections.defaultdict(list)
    for u in clean:
        by_spk[u["speaker"]].append(u)
    spks = sorted(by_spk)
    rng.shuffle(spks)
    dev, dev_spk = [], []
    for s in spks:
        h = hours(by_spk[s])
        if hours(dev) >= args.dev_hours:
            break
        if hours(dev) + h <= 1.5 * args.dev_hours and len(dev_spk) < len(spks) - 1:
            dev += by_spk[s]; dev_spk.append(s)
    pool = [u for u in clean if u["speaker"] not in set(dev_spk)]

    # 4. nested train subsets
    rng.shuffle(pool)
    targets = sorted(args.hours, reverse=True)
    subsets, acc, sec = {}, [], 0.0
    for u in pool:
        if sec >= targets[0] * 3600:
            break
        acc.append(u); sec += u["duration"]
    for h in targets:
        cut, s = [], 0.0
        for u in acc:
            if s >= h * 3600:
                break
            cut.append(u); s += u["duration"]
        subsets[h] = cut
    if hours(acc) < targets[0] * 0.98:
        print(f"WARNING: only {hours(acc)} h available for the {targets[0]} h subset")

    # 5. cut only the clips we use, one recording in memory at a time
    os.makedirs(f"{args.out}/clips", exist_ok=True)
    need = collections.defaultdict(list)
    for u in subsets[targets[0]] + dev:
        need[u["recording"]].append(u)
    for i, (rec, us) in enumerate(sorted(need.items())):
        audio, sr = sf.read(train.wav(rec), dtype="int16")
        if audio.ndim > 1:
            audio = audio.mean(axis=1).astype("int16")
        assert sr == 16000, (rec, sr)
        for u in us:
            sf.write(f"{args.out}/clips/{u['id']}.wav",
                     audio[int(u["start"] * sr):int(u["end"] * sr)], sr, subtype="PCM_16")
        if i % 20 == 0:
            print(f"cut {i + 1}/{len(need)} recordings", flush=True)

    def dump(name, rows):
        with open(f"{args.out}/{name}", "w", encoding="utf-8") as f:
            for u in rows:
                f.write(json.dumps({"id": u["id"], "path": f"clips/{u['id']}.wav", "text": u["text"],
                                    "speaker": u["speaker"], "recording": u["recording"],
                                    "duration": u["duration"]}, ensure_ascii=False) + "\n")

    dump("dev.jsonl", dev)
    stats["dev"] = {"utts": len(dev), "hours": hours(dev), "speakers": len(dev_spk)}
    for h, rows in subsets.items():
        name = f"train_{h:g}h.jsonl"
        dump(name, rows)
        stats[name] = {"utts": len(rows), "hours": hours(rows),
                       "speakers": len({u["speaker"] for u in rows})}
    assert not ({u["speaker"] for u in dev} & {u["speaker"] for u in acc}), "dev/train speaker leak"
    with open(f"{args.out}/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in stats.items() if k != "overlap"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
