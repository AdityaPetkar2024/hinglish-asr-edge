import glob
import soundfile as sf

base = "data/test"
for name in ["text", "wav.scp", "segments", "utt2spk"]:
    hits = glob.glob(f"{base}/**/{name}", recursive=True)
    print(name, "->", hits)
    if hits:
        with open(hits[0], encoding="utf-8") as f:
            for _ in range(3):
                print("   ", f.readline().strip())

wavs = glob.glob(f"{base}/**/*.wav", recursive=True)
print("\nwav files:", len(wavs))
if wavs:
    print(wavs[0], sf.info(wavs[0]))
