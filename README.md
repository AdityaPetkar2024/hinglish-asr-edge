# Offline Hinglish speech recognition on a phone

I fine tuned Whisper for Hindi and English code switched speech, quantized it with whisper.cpp, and ran it **fully offline on a Samsung Galaxy M17**. Along the way I studied why standard WER is misleading for code mixed speech.

**Demo video:** _(https://drive.google.com/file/d/1F19X8KvfLcE47qD5vv-T4ZYNJgqQMyy-/view?usp=sharing)_
**Data:** MUCS 2021 subtask 2 (OpenSLR 104, CC BY-SA 4.0). No audio is redistributed in this repo.

## Headline results

| | WER (300 test clips) | Size | Runs on |
|---|---|---|---|
| Stock whisper-base | 3.97 | 290 MB | GPU |
| **Fine tuned whisper-base (30 h)** | **0.389** [0.349, 0.434] | 290 MB | GPU (PyTorch) |
| Fine tuned, whisper.cpp **q8_0** with fallback | 0.427 [0.386, 0.468] | **78 MB** | Laptop CPU, RTF 0.38 |
| Same q8_0 model on the phone | 0.432 on 50 clips without fallback (laptop: 0.436 on the same clips and settings) | 78 MB | **Galaxy M17 in airplane mode, RTF 0.57, 173 MB RAM** |

Numbers in brackets are 95% bootstrap confidence intervals (2,000 resamples of utterances). RTF is processing time divided by audio duration, so a value below 1 means faster than real time.

## 1. Data

- **Source:** MUCS 2021 Hindi and English (Spoken Tutorial lectures), Kaldi format, 16 kHz mono. I kept segments that have text and last between 1 and 30 seconds.
- **Train pool:** 52,760 utterances, 89.0 hours, 520 speakers.
- **Leakage check against the test set:**
  - No recording IDs are shared, and no recording is identical to a test recording (checked by audio md5).
  - No speaker IDs are shared either. The corpus may number speakers separately in each split, though, so this does not prove the speakers are different people.
  - However, **3,036 train utterances (4.4 hours) had transcripts identical to test sentences** of 5 words or more, because the corpus reuses tutorial scripts across speakers. I removed all of them before training.
- **Dev set:** 1.0 hour, 531 utterances from 6 speakers kept entirely out of training.
- **Train subsets:** 30.0 hours (17,793 utterances), and an 8.0 hour subset of it (4,692 utterances).
- **Test set:** the official test set, 3,132 segments (5.14 hours, 30 speakers). Every number reported here uses the same fixed random split of 300 clips (29 minutes, all 30 speakers, `random.seed(0)`, md5 of the ID list `2699562b…`).

## 2. Evaluation, and why plain WER misleads here

All metrics share one normalizer: NFC, lowercase, punctuation and symbols removed. I did **not** use Whisper's `BasicTextNormalizer`. It treats Devanagari vowel signs (Unicode category M) as symbols and replaces them with spaces. For example, it turns *हमसे जुड़ने के लिए धन्यवाद* into *हमस ज ड न क ल ए धन यव द*.

For every model I report:
- WER and CER;
- **romanized WER:** Devanagari transliterated to ITRANS, with candra vowels and the final schwa handled;
- **transliteration aware WER (TA WER);**
- the share of each output script, the repetition loop rate, and RTF.

**TA WER** counts a substitution as correct when the reference and the hypothesis are the same word written in different scripts. Words are matched by a consonant skeleton key: *click* matches *क्लिक*, and *loop* matches *लूप*. When I inspected the 60 most frequent matched pairs by hand, 57 were genuinely the same word. Romanized WER follows the idea of transliteration optimized WER (toWER), which maps all text to one writing system (Emond et al., IEEE SLT 2018). TA WER is my looser variant of it, not a standard metric.

**Findings**

1. **The references themselves are inconsistent.** Within the same 300 test sentences, *click* is written in Latin script 41 times and as *क्लिक* 34 times. The same happens with box/बॉक्स (8 vs 12), panel/पैनल (7 vs 12) and dialog/डायलॉग (3 vs 8). No system can match both conventions, so plain WER has a floor that has nothing to do with recognition.
2. **About 14% of the fine tuned model's remaining errors are only about script.** For the fine tuned base model on 30 hours, TA WER is 0.335 while WER is 0.389.
3. **Stock Whisper's WER above 1 is mostly hallucination.** 77% of stock whisper-base's word errors are insertions, and 30% of its outputs contain repetition loops. Setting `no_repeat_ngram_size=5` alone lowers WER from 3.97 to 1.03, without any training.
4. **Some test labels are wrong, verified by listening.** I flagged clips where all three fine tuned models agreed with each other but disagreed with the reference, and listened to 10 of them:
   - 3 had the wrong transcript attached (misaligned segments);
   - 1 matched only partly;
   - 6 were genuine model errors (see section 5).

   Removing the 3 confirmed mislabeled clips changes the WER of the fine tuned base model (30 hours) only from 0.389 to 0.384, so the test set is mostly sound.

## 3. Fine tuning

`train.py` implements the Hugging Face Whisper recipe as a plain PyTorch loop:
- **Optimization:** AdamW, batch size 32, 2,000 steps, linear warmup over 200 steps, then linear decay. Learning rate 2.5e-5 for base and 1e-5 for small.
- **Precision and augmentation:** fp16 mixed precision, and SpecAugment time masking with probability 0.05.
- **Labels:** the reference transcripts as they are (mixed script), with the prefix tokens `<|hi|><|transcribe|><|notimestamps|>`.
- **Model selection:** the checkpoint with the best dev WER, evaluated every 500 steps.
- **Hardware:** one Kaggle T4 GPU per run. Each base model took 41 minutes to train, and the small model took 149 minutes.

All rows use greedy decoding in fp16 on a T4:

| Model | WER [95% CI] | TA WER | CER | Loops | Latin script share* | RTF |
|---|---|---|---|---|---|---|
| Stock whisper-tiny | 4.115 | 4.054 | 3.552 | 23% | 87% | 0.105 |
| Stock whisper-base | 3.966 [3.25, 4.78] | 3.912 | 3.786 | 30% | 60% (plus 25% Urdu script) | 0.187 |
| Stock whisper-small | 2.209 | 2.120 | 2.183 | 34% | 1% | 0.537 |
| Stock base, no repeat ngram 5 | 1.026 [0.97, 1.09] | 0.968 | 0.784 | 0.7% | 61% | 0.038 |
| **Fine tuned base, 8 h** | 0.426 [0.388, 0.468] | 0.371 | 0.329 | 0.3% | 29% | 0.040 |
| **Fine tuned base, 30 h** | **0.389 [0.349, 0.434]** | **0.335** | **0.316** | 0% | 28% | 0.046 |
| **Fine tuned small, 30 h** | 0.397 [0.343, 0.460] | 0.335 | 0.340 | 0.7% | 29% | 0.132 |

\*In the references, 27% of utterances are mostly in Latin script and 73% mostly in Devanagari. Fine tuning brings the model's output script mix in line with this convention.

**Paired comparisons** (same 300 clips, 95% bootstrap CI of the WER difference):
- **8 hours vs 30 hours of training data:** 30 hours lowers WER by 0.036 [0.009, 0.063]. More data helps, but modestly.
- **Base vs small, both trained on 30 hours:** small is worse by 0.008 on test, with a CI from 0.037 better to 0.065 worse, so there is no detectable difference. This holds even though small was better on dev (0.237 vs 0.271). Small is also 2.9× slower, so **the base model trained on 30 hours is the one to deploy**.
- **Best stock decoding vs the fine tuned base model (30 hours):** fine tuning lowers WER by 0.636 [0.566, 0.713].

**Dev WER during training:**

| Model | Step 0 | 500 | 1000 | 1500 | 2000 |
|---|---|---|---|---|---|
| Base, 8 h | 1.42 | 0.39 | 0.34 | 0.32 | 0.32 |
| Base, 30 h | 1.42 | 0.34 | 0.29 | 0.28 | 0.27 |
| Small, 30 h | 1.23 | 0.32 | 0.26 | 0.24 | 0.24 |

## 4. Edge deployment

I converted the fine tuned base model (30 hours) with whisper.cpp's `convert-h5-to-ggml.py` and then quantized it. Decoding settings:
- greedy decoding, no timestamps, language set to Hindi (`-l hi`);
- "fallback" means whisper.cpp's default temperature fallback, which decodes a segment again when the output looks repetitive.

**Laptop:** Dell Latitude 7480, Intel i7-7600U (2 cores, 4 threads), all 4 threads used, 300 clips.

| Variant | Size | WER without fallback | **WER with fallback** | Clips with loops (without → with fallback) | RTF (without / with fallback) | Peak RAM |
|---|---|---|---|---|---|---|
| f16 | 141 MB | 0.472 | not run | 4 | 0.54 | 247 MB |
| q8_0 | 78 MB | 0.457 | **0.427** [0.386, 0.468] | 3 → 0 | 0.34 / 0.38 | 181 MB |
| q5_0 | 53 MB | 0.542 | **0.422** [0.383, 0.465] | 9 → 0 | 0.42 / 0.42 | 156 MB |

**Phone:**
- **Device:** Samsung Galaxy M17 (SM-M176B), SoC reported by Android as s5e8535, 8 cores, 3.4 GB RAM, Android 16.
- **Setup:** whisper.cpp built on the phone itself in Termux, run in **airplane mode** with 4 threads and greedy decoding without fallback.
- **Test data:** 50 test clips (263 seconds of audio).

| Variant | RTF | Peak RAM | WER on these 50 clips | Identical to laptop output |
|---|---|---|---|---|
| **q8_0** | **0.575** (repeat run: 0.572) | 173 MB | 0.432 (laptop: 0.436) | 40 of 50 clips |
| q5_0 | 0.931 | 148 MB | 0.429 (laptop: 0.448) | 41 of 50 clips |

The phone reproduces the laptop's accuracy, and most of its transcripts are identical character for character. The rest differ slightly, most likely because ARM and x86 do floating point arithmetic in a different order (not verified), so greedy decoding sometimes picks a different token. These differences don't move WER in either direction. Both devices ran without fallback for this comparison.

**Findings**

5. **Quantization changed how the model fails, not how accurate it is.**
   - Without a loop safeguard, q5_0 looks much worse (WER 0.542).
   - But if the 13 clips where any variant loops are left out, f16, q8_0 and q5_0 score 0.413, 0.416 and 0.410.
   - With fallback, q5_0 and q8_0 differ by only 0.005 (CI from 0.027 better to 0.015 worse).
   - Measuring quantized models without a loop safeguard would have wrongly blamed the 5 bit model.
6. **Moving from PyTorch to whisper.cpp has a small but real cost:** WER rises by 0.037 [0.011, 0.062] (q8_0 with fallback vs PyTorch fp16). I have not isolated the cause.
7. **The smallest model is not the fastest on the phone.**
   - q5_0 is 32% smaller than q8_0 but **62% slower** on the phone's CPU (RTF 0.93 vs 0.57).
   - A plausible explanation, which I have not profiled, is that unpacking 5 bit weights costs more compute than the smaller size saves.
   - The repeat q8_0 run came after the q5_0 run and was just as fast as the first one, so heat does not explain the gap.
   - **Recommended deployment: q8_0 with fallback.**

## 5. Error analysis

Before and after fine tuning (stock whisper-base vs the fine tuned base model trained on 30 hours):

| Reference | Stock whisper-base | Fine tuned |
|---|---|---|
| file और close पर क्लिक करके file को बंद करें | file & close & click on file & close & close & close & close … | ✓ exact |
| menu bar पर और status bar पर भाषा में बदलाव को देखें | Manu baar paar paar paar paar … | ✓ exact |
| इस tutorial में हम कुछ उदाहरणों की मदद से निम्न के बारे में सीखेंगे | اس utorial میں ہم کچھ اُدھارا نوکی … (Urdu script) | ✓ exact |
| निम्न link पर उपलब्ध video देखें | نمدلنگ پر اپلگ دیو دیو دیکھیں … (Urdu script) | ✓ exact |
| ध्यान दें कि slides स्वतः से प्रस्तुत होती हैं | dhan de ki slides sata se prasthuth hoti hai … (romanized) | ✓ exact |

**What still goes wrong**
- **Script mismatches:** *click* vs *क्लिक*, and *command* vs *कमांड* (finding 2).
- **Stock tutorial phrases on unclear audio.** On short or unclear clips, all the fine tuned models sometimes produce fluent Spoken Tutorial stock phrases that were not said. Examples: "और फिर अपने खुद के layout बना सकते हैं" and "spoken hyphen tutorial dot org …".
  - All three models were trained on the same tutorial series, so they make the same mistake.
  - So agreement between the models is not evidence that they are right: 6 of the 10 "all models agree" clips I listened to were model errors.
- **English speech gets translated.** One clip spoken entirely in English (with an English reference) came out as a Hindi translation, because decoding forces the language to Hindi.

## 6. Limitations

- **One training run per configuration** (a single random seed), so small differences such as base vs small may not be stable.
- **Small test sample:** 300 test clips, with confidence intervals reported throughout. The phone measurements use 50 clips.
- **One domain:** read technical tutorials from a single series, so the model has partly memorized their phrasing. There is no spontaneous speech, background noise, or accent variety beyond this corpus.
- **No hyperparameter search.** The dev set comes from the training distribution, which is why dev WER (0.27) is lower than test WER (0.39) for the deployed model.
- **The 0.04 WER gap between PyTorch and whisper.cpp is unexplained.**
- **Recorded clips only:** the demo transcribes recorded clips, not live microphone input or a stream.

## Reproduce

| File | Where it runs | What it does |
|---|---|---|
| `prepare_test.py` | Anywhere | Cuts the test segments and builds the fixed split of 300 clips |
| `prep_data_kaggle.ipynb` with `prepare_train.py` | Kaggle CPU | Downloads and filters the data, checks for leakage, builds the dev, 8 h and 30 h splits |
| `day1_kaggle.ipynb` | Kaggle T4 | Stock model baselines |
| `train_kaggle.ipynb` with `train.py` | Kaggle, 2 T4 GPUs | Fine tunes base (8 h and 30 h) and small (30 h), then evaluates them |
| `eval.py`, `metrics.py` | GPU or CPU | PyTorch evaluation, plus the metric code shared by all scripts |
| `analyze.py` | CPU | Bootstrap CIs, paired differences, TA WER, error breakdown |
| `setup_edge.sh`, `eval_cpp.py` | Laptop | Builds whisper.cpp, converts and quantizes the model, scores each version |
| `make_phone_kit.sh`, `run_phone.sh`, `compare_phone.py` | Laptop and phone | Phone benchmark and demo |
| `listen_check.sh` | Laptop | Guided listening check of flagged clips |

**Credits:** dataset MUCS 2021 (OpenSLR SLR104, CC BY-SA 4.0); models OpenAI Whisper (MIT); inference engine whisper.cpp (MIT).
