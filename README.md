# colab-whisper-stt

**English** | [한국어](README.ko.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ehdtn0843ds-dss/colab-whisper-stt/blob/main/notebook/colab_whisper_stt.ipynb)
[![Guide](https://img.shields.io/badge/guide-EN%20%7C%20%ED%95%9C%EA%B5%AD%EC%96%B4-76b900)](https://ehdtn0843ds-dss.github.io/colab-whisper-stt/)

Transcribe long Korean/English talk recordings on a free Colab T4 GPU with **faster-whisper large-v3**.
I built this to turn my recordings from NVIDIA AI Day Seoul 2026 into text: the host spoke Korean, keynotes were in English, and booth demos mixed both with a lot of product names.

📖 **Full walkthrough (blog-style, English / 한국어): https://ehdtn0843ds-dss.github.io/colab-whisper-stt/**

## What it does

| Problem with plain Whisper | What this pipeline does |
|---|---|
| Korean and English switch within a session | Picks `ko` or `en` per speech chunk; short ambiguous chunks keep the previous language |
| Product names come out wrong | Per-session hotword list, in priority order, within a 120-token budget |
| Music/applause turns into invented lines | Transcribes VAD chunks, drops repeated segments and subtitle-credit phrases |
| Colab disconnects mid-file | Saves ~10-minute blocks to Drive; finished blocks are skipped on restart |

## One real run (T4, control_006)

| run | model | audio | RTF | fallbacks | mean avg_logprob |
|---|---|---|---|---|---|
| `full_v3` | large-v3 | 79-min keynote | 0.114 (542 s) | 16 | −0.49 |
| `full_v2` | large-v2 | same file | 0.182 (865 s) | 53 | −0.59 |
| `patch_qna_en` | large-v3, forced `en` | 53 s Q&A passage | 0.086 | 0 | −0.42 |

Peak GPU memory was 4.9 GB. These are speed and confidence figures from one recording; word error rate was not measured.

## Quick start

1. Click **Open in Colab** and set `Runtime → Change runtime type → T4 GPU`.
2. Run **SETUP**, then **BOOTSTRAP**. It creates `MyDrive/aiday_stt/` and copies the code and templates there.
3. Put recordings in `MyDrive/aiday_stt/audio/<id>.m4a` and describe them in `code/sessions.json`.
4. Edit the control file (start with the pilot one, then [`examples/control_002_full.json`](examples/control_002_full.json)).
5. Run **MOUNT_AND_LOAD** and **RUN**. A zip with the transcripts downloads when it finishes.

```
MyDrive/aiday_stt/
├── control_001_pilot.json     # the last control_*.json in sorted order is used
├── audio/01_keynote.m4a       # file name = id in sessions.json
├── code/stt_core.py, runner.py, sessions.json
└── out/                       # results, logs, checkpoints, STATUS.json
```

Turn a result into plain text and subtitles:

```bash
python code/export_text.py out/full_v3/01_keynote.json   # writes .txt and .srt
```

## Repository layout

| Path | What it is |
|---|---|
| [`notebook/colab_whisper_stt.ipynb`](notebook/colab_whisper_stt.ipynb) | The Colab notebook: SETUP → BOOTSTRAP → MOUNT_AND_LOAD → RUN |
| [`code/stt_core.py`](code/stt_core.py) | Transcription: VAD chunks, gap filling, per-chunk language, hotwords + context prompt, resume |
| [`code/runner.py`](code/runner.py) | Orchestration (stdlib only): audio prep, subprocess, logs, report, zip |
| [`code/export_text.py`](code/export_text.py) | Result JSON → `.txt` / `.srt` |
| [`code/sessions.json`](code/sessions.json) | Template for describing recordings |
| [`examples/`](examples/) | Control file templates, plus the actual AI Day sessions and controls |
| [`docs/index.html`](docs/index.html) | The bilingual guide (GitHub Pages) |
| [`tests/`](tests/) | CPU-only tests (`python -m pytest -q tests`) |

## Pinned versions

faster-whisper 1.2.1 · CTranslate2 4.8.5 · PyAV 16.0.1 · CUDA 12 cuBLAS/cuDNN wheels (installed by SETUP)

## Notes

- Record and transcribe only where the event allows it. This repo contains no audio and no transcripts.
- Code is under the [MIT License](LICENSE). Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2) and [Silero VAD](https://github.com/snakers4/silero-vad).
