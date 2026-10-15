#!/usr/bin/env bash
# Run on the LAPTOP. Builds phone_kit/ (model + 50 test clips + phone script) to copy to the phone.
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-50}
rm -rf phone_kit && mkdir -p phone_kit/clips
cp models/ggml-base30h-q8_0.bin models/ggml-base30h-q5_0.bin phone_kit/
python3 - "$N" <<'PY'
import json, shutil, sys
n = int(sys.argv[1])
rows = [json.loads(l) for l in open("data/eval_300.jsonl", encoding="utf-8")][:n]
with open("phone_kit/clips.tsv", "w", encoding="utf-8") as f:
    for r in rows:
        shutil.copy(r["path"], f"phone_kit/clips/{r['id']}.wav")
        f.write(f"{r['id']}\t{r['duration']}\t{r['text']}\n")
print(n, "clips,", round(sum(r["duration"] for r in rows) / 60, 1), "min of audio")
PY
cp run_phone.sh phone_kit/
du -sh phone_kit
echo "Now copy the phone_kit folder to the phone's Download folder (USB cable or Google Drive)."
