#!/data/data/com.termux/files/usr/bin/bash
# Run on the PHONE inside Termux:   bash ~/phone_kit/run_phone.sh [threads] [q8_0|q5_0]   (benchmark, ~5-15 min)
#                                    bash ~/phone_kit/run_phone.sh demo                     (3 clips, for the video)
set -uo pipefail
KIT="$(cd "$(dirname "$0")" && pwd)"
CLI="$HOME/whisper.cpp/build/bin/whisper-cli"
Q=${2:-q8_0}
MODEL="$KIT/ggml-base30h-$Q.bin"
[ -x "$CLI" ] || { echo "whisper-cli not found at $CLI (build whisper.cpp first)"; exit 1; }

if [ "${1:-}" = "demo" ]; then
  head -n 3 "$KIT/clips.tsv" | while IFS=$'\t' read -r id dur ref; do
    echo; echo "=== clip $id (${dur}s)"; echo "REFERENCE : $ref"
    t0=$(date +%s.%N)
    out=$("$CLI" -m "$MODEL" -l hi -t 4 -bs 1 -bo 1 -nf -nt -np -f "$KIT/clips/$id.wav" 2>/dev/null)
    t1=$(date +%s.%N)
    echo "PHONE ASR : $out"
    echo "time      : $(echo "$t1 - $t0" | bc) s for ${dur} s of audio"
  done
  exit 0
fi

T=${1:-4}
[ -f "$MODEL" ] || { echo "model not found: $MODEL"; exit 1; }
OUT="$KIT/out_${Q}_t$T"; rm -rf "$OUT"; mkdir -p "$OUT"
{
  echo "device : $(getprop ro.product.manufacturer) $(getprop ro.product.model)"
  echo "soc    : $(getprop ro.soc.manufacturer) $(getprop ro.soc.model) / $(getprop ro.board.platform)"
  echo "android: $(getprop ro.build.version.release)"
  echo "cores  : $(nproc)"
  echo "ram    : $(awk '/MemTotal/ {printf "%.1f GB", $2/1048576}' /proc/meminfo)"
  echo "threads: $T"
  echo "model  : $Q"
} | tee "$OUT/device.txt"

ARGS=(-m "$MODEL" -l hi -t "$T" -bs 1 -bo 1 -nf -nt -np -otxt)
while IFS=$'\t' read -r id dur ref; do
  cp "$KIT/clips/$id.wav" "$OUT/$id.wav"; ARGS+=(-f "$OUT/$id.wav")
done < "$KIT/clips.tsv"

AUDIO=$(awk -F'\t' '{s+=$2} END {print s}' "$KIT/clips.tsv")
TIMER=""; command -v time >/dev/null && [ -x "$PREFIX/bin/time" ] && TIMER="$PREFIX/bin/time -v"
echo "transcribing $(wc -l < "$KIT/clips.tsv") clips (${AUDIO}s of audio)... keep the screen on"
t0=$(date +%s.%N)
$TIMER "$CLI" "${ARGS[@]}" >/dev/null 2>"$OUT/stderr.txt"
rc=$?; t1=$(date +%s.%N)
rm -f "$OUT"/*.wav
ELAPSED=$(echo "$t1 - $t0" | bc)
RTF=$(echo "scale=3; $ELAPSED / $AUDIO" | bc)
RAM=$(grep -o "Maximum resident set size (kbytes): [0-9]*" "$OUT/stderr.txt" | awk '{printf "%.0f MB", $NF/1024}')
{
  echo "exit code   : $rc"
  echo "audio       : $AUDIO s"
  echo "wall time   : $ELAPSED s"
  echo "RTF         : $RTF   (below 1 = faster than real time)"
  echo "peak RAM    : ${RAM:-n/a (pkg install time)}"
  echo "transcripts : $(ls "$OUT"/*.txt 2>/dev/null | wc -l)"
} | tee "$OUT/summary.txt"
cd "$KIT" && tar czf "$HOME/storage/downloads/phone_out_${Q}_t$T.tgz" "out_${Q}_t$T" && echo "copied results to Download/phone_out_${Q}_t$T.tgz"
