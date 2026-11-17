"""CPU-only tests. No GPU, no model download.

stt_core.py imports ctranslate2 and faster_whisper at module level, so these
tests register small stand-ins before importing it and only exercise the pure
logic (chunking, gap filling, context, filters). runner.py is exercised end to
end with real ffmpeg and a fake transcriber script.

Run:  python -m pytest -q tests
"""
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code'


def _stub_gpu_modules():
    for name in ('ctranslate2', 'faster_whisper', 'faster_whisper.tokenizer', 'faster_whisper.vad'):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules['ctranslate2'].get_cuda_device_count = lambda: 0
    fw = sys.modules['faster_whisper']
    fw.WhisperModel = object
    fw.decode_audio = lambda *a, **k: None
    sys.modules['faster_whisper.tokenizer'].Tokenizer = object
    vad = sys.modules['faster_whisper.vad']
    vad.VadOptions = dict
    vad.get_speech_timestamps = lambda *a, **k: []


_stub_gpu_modules()
sys.path.insert(0, str(CODE))
import stt_core  # noqa: E402

SR = stt_core.SAMPLE_RATE


def span(start_s, end_s):
    return {'start': int(start_s * SR), 'end': int(end_s * SR)}


def test_make_chunks_merges_close_spans_and_splits_long_ones():
    max_samples = 28 * SR
    chunks = stt_core.make_chunks([span(0, 5), span(6, 10), span(20, 25)], max_samples, 2.0)
    # 5 -> 6 gap is 1 s (merged); 10 -> 20 gap is 10 s (new chunk)
    assert chunks == [(0, 10 * SR), (20 * SR, 25 * SR)]
    long = stt_core.make_chunks([span(0, 70)], max_samples, 2.0)
    assert [(s / SR, e / SR) for s, e in long] == [(0, 28), (28, 56), (56, 70)]
    assert all(e - s <= max_samples for s, e in long)


def test_fill_gaps_covers_uncovered_audio_and_tags_sources():
    audio = np.zeros(60 * SR, dtype=np.float32)
    chunks = [(10 * SR, 20 * SR)]
    out, sources, dropped = stt_core.fill_gaps(chunks, audio, 3, 28 * SR)
    assert sources == ['gap', 'vad', 'gap', 'gap']
    covered = sum(e - s for s, e in out)
    assert covered == len(audio)
    assert dropped == 0
    assert all(e - s <= 30 * SR for s, e in out)
    # Disabled gap filling returns the input unchanged.
    same, src, _ = stt_core.fill_gaps(chunks, audio, 0, 28 * SR)
    assert same == chunks and src == ['vad']


def test_fill_gaps_cuts_at_quiet_frame():
    audio = np.ones(40 * SR, dtype=np.float32)
    quiet_end = 25 * SR
    audio[quiet_end - SR // 5:quiet_end] = 0.0  # one silent 0.2 s frame ending at 25 s
    out, _, _ = stt_core.fill_gaps([], audio, 3, 28 * SR)
    assert out[0] == (0, quiet_end)


def test_make_blocks_groups_by_duration():
    chunks = [(i * 10 * SR, (i * 10 + 8) * SR) for i in range(10)]
    blocks = stt_core.make_blocks(chunks, 30 * SR)
    assert blocks[0] == [0, 1, 2]
    assert sum(len(b) for b in blocks) == 10


def test_context_prompt_rules():
    prev = dict(last_text_tail=' alpha beta gamma', language='en', end=10.0,
                max_compression_ratio=1.5, max_temperature=0.0)
    assert stt_core.context_prompt(prev, 'en', 12.0, 10) == 'beta gamma'  # cut lands on a space
    assert stt_core.context_prompt(prev, 'en', 12.0, 8) == 'gamma'        # cut inside "beta" skips it
    assert stt_core.context_prompt(prev, 'ko', 12.0, 10) == ''        # language changed
    assert stt_core.context_prompt(prev, 'en', 16.0, 10) == ''        # gap > 5 s
    bad = dict(prev, max_compression_ratio=2.5)
    assert stt_core.context_prompt(bad, 'en', 12.0, 10) == ''         # looked repetitive
    assert stt_core.context_prompt(None, 'en', 12.0, 10) == ''


def test_stock_phrase_and_duplicate_filters():
    segs = [
        {'chunk': 0, 'text': '시청해 주셔서 감사합니다.'},
        {'chunk': 0, 'text': 'Isaac Sim을 씁니다'},
        {'chunk': 0, 'text': 'Isaac Sim을 씁니다'},
        {'chunk': 1, 'text': 'Isaac Sim을 씁니다'},
    ]
    stt_core.drop_duplicates(segs)
    stt_core.drop_stock_phrases(segs)
    assert segs[0].get('dropped') == 'stock'
    assert 'dropped' not in segs[1]
    assert segs[2].get('dropped') == 'dup'
    assert 'dropped' not in segs[3]  # same text in a new chunk is kept


class CharTokenizer:
    """One token per character: enough to test budgeting logic."""
    def encode(self, text):
        return list(text)


def test_hotword_budget_keeps_priority_prefix():
    used, dropped, count = stt_core.budget_hotwords(['PyTorch', 'Transformer', 'fine-tuning'], CharTokenizer(), 22)
    assert used == ['PyTorch', 'Transformer'] and dropped == ['fine-tuning']
    assert count == len(' PyTorch, Transformer')


def test_trim_context_tokens_keeps_word_boundary_suffix():
    text, count = stt_core.trim_context_tokens('one two three four', CharTokenizer(), 12)
    assert text == 'three four' and count == len(' three four')


# --- runner.py end to end ------------------------------------------------------

FAKE_CORE = r'''
import argparse, json, pathlib
p = argparse.ArgumentParser()
for a in ("--audio", "--vad-audio", "--file-id", "--sessions", "--out-dir", "--tag"):
    p.add_argument(a)
p.add_argument("--start", type=float); p.add_argument("--duration", type=float)
args, extra = p.parse_known_args()
print("block 0000 RUN audio_s=1.000 wall_s=0.100 rtf=0.100000", flush=True)
seg = dict(start=0.0, end=1.0, text=" hello", language="en", avg_logprob=-0.2, temperature=0.0)
data = dict(range=dict(start_s=args.start, duration_s=2.0),
            config=dict(hotword_tokens=3, hotwords_dropped=[], extra=extra, audio=args.audio),
            stats=dict(rtf=0.1, speech_s=1.0, n_chunks=1, n_segments=1, gpu_mem_peak_mb=None),
            chunks=[dict(start=0.0, end=1.0, tail_gap_s=0.0, context_used=False)],
            segments=[seg])
out = pathlib.Path(args.out_dir) / args.tag
out.mkdir(parents=True, exist_ok=True)
(out / (args.file_id + ".json")).write_text(json.dumps(data))
'''


@pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not installed')
def test_runner_end_to_end(tmp_path):
    base = tmp_path / 'whisper_stt'
    (base / 'audio').mkdir(parents=True)
    subprocess.run(['ffmpeg', '-loglevel', 'error', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
                    '-c:a', 'aac', str(base / 'audio' / '01_lecture.m4a')], check=True)
    code = base / 'code'
    code.mkdir()
    shutil.copy(CODE / 'sessions.json', code / 'sessions.json')
    fake = tmp_path / 'fake_core.py'
    fake.write_text(FAKE_CORE)

    control = json.loads((ROOT / 'examples' / 'control_001_pilot.json').read_text())
    for run in control['pilot_runs']:
        run.update(start=0, duration=0)
    control['_name'] = 'control_001_pilot'

    namespace = {}
    exec(compile((CODE / 'runner.py').read_text(), 'runner.py', 'exec'), namespace)
    result = namespace['main'](str(base), control, str(code), {'versions': {}},
                               core_path=fake, zip_dir=tmp_path / 'zip')

    assert [r['exit_code'] for r in result['results']] == [0, 0]
    report = (base / 'out' / 'report_control_001_pilot.md').read_text()
    assert '[00:00][en] hello' in report
    status = json.loads((base / 'out' / 'STATUS.json').read_text())
    assert status['stage'] == 'done'
    # The loudnorm variant was rendered to 16 kHz mono WAV and passed to the core.
    data = json.loads((base / 'out' / 'pilot_loudnorm' / '01_lecture.json').read_text())
    assert data['config']['audio'].endswith('loudnorm/01_lecture.wav')
    import zipfile
    names = zipfile.ZipFile(result['zip']).namelist()
    assert 'report_control_001_pilot.md' in names and 'pilot_raw/01_lecture.json' in names


def test_export_text_formats():
    sys.path.insert(0, str(CODE))
    import export_text
    data = {'session': 'Demo', 'source_file': 'demo.m4a', 'recorded_kst': '2026-11-09',
            'segments': [{'start': 3725.5, 'end': 3727.25, 'language': 'ko', 'text': ' 안녕하세요'},
                         {'start': 3728.0, 'end': 3729.0, 'language': 'en', 'text': '  '}]}
    assert '[01:02:05] [ko] 안녕하세요' in export_text.to_txt(data)
    assert export_text.to_srt(data) == '1\n01:02:05,500 --> 01:02:07,250\n안녕하세요\n'
