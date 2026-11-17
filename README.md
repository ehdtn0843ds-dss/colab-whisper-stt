# colab-whisper-stt

**English** | [한국어](README.ko.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ehdtn0843ds-dss/colab-whisper-stt/blob/main/notebook/colab_whisper_stt.ipynb)
[![Guide](https://img.shields.io/badge/guide-EN%20%7C%20%ED%95%9C%EA%B5%AD%EC%96%B4-f26b1d)](https://ehdtn0843ds-dss.github.io/colab-whisper-stt/)

A Colab notebook for transcribing recordings where Korean and English are mixed. It runs faster-whisper large-v3 on a free T4 GPU and picks the language for each speech chunk, so it can follow a Korean lecture full of English terms or a meeting that switches to English partway through.

Plain Whisper tends to stick to one language for the whole file and to get jargon wrong. It also makes up sentences over silence and music. The notebook works around each of these (see the table below).

Full guide in English and Korean: https://ehdtn0843ds-dss.github.io/colab-whisper-stt/

## Good fit for

- Lectures and seminars in Korean with English technical terms
- Meetings where people switch between Korean and English
- Conference talks, panels and Q&A
- Interviews, podcasts and YouTube videos with mixed speech

## What it does

| Problem with plain Whisper | What this notebook does |
|---|---|
| Korean and English switch within one file | Picks `ko` or `en` per speech chunk; short ambiguous chunks keep the previous language |
| Names and jargon come out wrong | Per-recording hotword list, in priority order, within a 120-token budget |
| Silence, music and applause turn into invented lines | Transcribes VAD speech chunks, drops repeated segments and subtitle-credit phrases |
| Colab disconnects in the middle of a long file | Saves ~10-minute blocks to Drive; finished blocks are skipped on restart |

## How fast

Measured on a free-tier Tesla T4 with a 79-minute conference recording (about 56 min English, 22 min Korean):

| model | decoding time | RTF | temperature fallbacks | mean avg_logprob |
|---|---|---|---|---|
| large-v3 | 542 s (~9 min) | 0.114 | 16 | −0.49 |
| large-v2 | 865 s | 0.182 | 53 | −0.59 |

Peak GPU memory was 4.9 GB of 15 GB. These are speed and confidence figures from one recording; word error rate was not measured. The real session and control files are in [`examples/real-run-conference/`](examples/real-run-conference/).

## Quick start

1. Click Open in Colab and set `Runtime → Change runtime type → T4 GPU`.
2. Run SETUP, then BOOTSTRAP. It creates `MyDrive/whisper_stt/` and copies the code and templates there.
3. Put recordings in `MyDrive/whisper_stt/audio/<id>.m4a` and describe each one in `code/sessions.json` (main language + hotwords).
4. Start with the pilot control file (two minutes, two settings), then switch to [`examples/control_002_full.json`](examples/control_002_full.json).
5. Run MOUNT_AND_LOAD and RUN. A zip with the transcripts downloads when it finishes.

```
MyDrive/whisper_stt/
├── control_001_pilot.json     # the last control_*.json in sorted order is used
├── audio/01_lecture.m4a       # file name = id in sessions.json
├── code/stt_core.py, runner.py, sessions.json
└── out/                       # results, logs, checkpoints, STATUS.json
```

A `sessions.json` entry:

```json
{
  "id": "01_lecture",
  "source_file": "deep_learning_week3.m4a",
  "recorded_kst": "2026-03-17 10:30-12:00",
  "session": "Lecture in Korean with English technical terms",
  "default_language": "ko",
  "hotwords": ["PyTorch", "Transformer", "attention", "fine-tuning", "역전파", "경사하강법"]
}
```

Turn a result into plain text and subtitles:

```bash
python code/export_text.py out/full_v3/01_lecture.json   # writes .txt and .srt
```

## Repository layout

| Path | What it is |
|---|---|
| [`notebook/colab_whisper_stt.ipynb`](notebook/colab_whisper_stt.ipynb) | The Colab notebook: SETUP → BOOTSTRAP → MOUNT_AND_LOAD → RUN |
| [`code/stt_core.py`](code/stt_core.py) | Transcription: VAD chunks, gap filling, per-chunk language, hotwords + context prompt, resume |
| [`code/runner.py`](code/runner.py) | Orchestration (stdlib only): audio prep, subprocess, logs, report, zip |
| [`code/export_text.py`](code/export_text.py) | Result JSON → `.txt` / `.srt` |
| [`code/sessions.json`](code/sessions.json) | Template for describing recordings |
| [`examples/`](examples/) | Control file templates and a real conference run |
| [`docs/index.html`](docs/index.html) | The bilingual guide (GitHub Pages) |
| [`tests/`](tests/) | CPU-only tests (`python -m pytest -q tests`) |

## Pinned versions

faster-whisper 1.2.1 · CTranslate2 4.8.5 · PyAV 16.0.1 · CUDA 12 cuBLAS/cuDNN wheels (installed by SETUP)

`--languages` takes any languages Whisper supports (for example `ja,en`), but the defaults and the hallucination filter were tuned on Korean/English audio.

## Notes

- Transcribe recordings only where you have the right to record them. This repo contains no audio and no transcripts.
- Code is under the [MIT License](LICENSE). Built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2) and [Silero VAD](https://github.com/snakers4/silero-vad).
