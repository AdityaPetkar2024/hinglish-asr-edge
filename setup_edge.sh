#!/usr/bin/env bash
# Build whisper.cpp, convert the fine-tuned whisper-base (30 h) to ggml, quantize, and score each version.
# Run from ~/hinglish-asr-edge with the venv active. Expects the Kaggle files in models/base_30h/.
set -euo pipefail
M=models/base_30h
for f in config.json generation_config.json model.safetensors; do
  [ -f "$M/$f" ] || { echo "missing $M/$f: download it from the mucs-train Output tab"; exit 1; }
done

# 1. build whisper.cpp (a few minutes, once)
[ -d whisper.cpp ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp
[ -x whisper.cpp/build/bin/whisper-cli ] || {
  cmake -S whisper.cpp -B whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
  cmake --build whisper.cpp/build -j4 --config Release
}

# 2. converter inputs: mel filters from openai/whisper, tokenizer vocab from stock whisper-base
#    (fine-tuning didn't change the tokenizer; newer transformers just doesn't save these two files)
[ -d openai-whisper ] || git clone --depth 1 https://github.com/openai/whisper openai-whisper
for f in vocab.json added_tokens.json; do
  [ -f "$M/$f" ] || wget -q -O "$M/$f" "https://huggingface.co/openai/whisper-base/resolve/main/$f"
done

# 3. convert (f16) and quantize
if [ ! -f models/ggml-base30h-f16.bin ]; then
  python whisper.cpp/models/convert-h5-to-ggml.py "$M" openai-whisper models/
  mv models/ggml-model.bin models/ggml-base30h-f16.bin
fi
for q in q8_0 q5_0; do
  [ -f models/ggml-base30h-$q.bin ] || whisper.cpp/build/bin/whisper-quantize models/ggml-base30h-f16.bin models/ggml-base30h-$q.bin $q
done
ls -lh models/*.bin

# 4. one-clip sanity check before the full runs
CLIP=$(python -c "import json;print(json.loads(open('data/eval_300.jsonl').readline())['path'])")
echo "REF : $(python -c "import json;print(json.loads(open('data/eval_300.jsonl').readline())['text'])")"
echo -n "PRED: "; whisper.cpp/build/bin/whisper-cli -m models/ggml-base30h-q8_0.bin -l hi -nt -np -f "$CLIP"

# 5. score all three on the same 300 clips (laptop CPU, 4 threads)
for q in f16 q8_0 q5_0; do
  [ -f results/cpp_base30h_${q}_metrics.json ] && { echo "already scored: $q"; continue; }
  echo "scoring $q on 300 clips (about 20 min, quiet while it runs) ... started $(date +%H:%M)"
  python eval_cpp.py --cli whisper.cpp/build/bin/whisper-cli --model models/ggml-base30h-$q.bin --tag cpp_base30h_$q
done
python - <<'PY'
import json, glob
for f in sorted(glob.glob("results/cpp_*_metrics.json")):
    m = json.load(open(f))
    print(f"{m['tag']:22s} size {m['model_mb']:6.1f} MB  WER {m['wer']:.4f}  CER {m['cer']:.4f}  RTF {m['rtf']:.3f}  RAM {m['peak_ram_mb']} MB")
PY
