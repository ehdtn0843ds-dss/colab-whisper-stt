"""Standalone faster-whisper transcription with VAD chunks and atomic resume."""
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unicodedata

import ctranslate2
import numpy as np
from faster_whisper import WhisperModel, decode_audio
from faster_whisper.tokenizer import Tokenizer
from faster_whisper.vad import VadOptions, get_speech_timestamps

SAMPLE_RATE = 16000
STOCK_PHRASES = (
    '자막제공', '자막제공및광고를포함하고있습니다', '광고를포함하고있습니다',
    '시청해주셔서감사합니다', '구독과좋아요', '한글자막by', '제작지원',
    '다음영상에서만나요', 'mbc뉴스', 'thankyouforwatching', 'thanksforwatching',
    'subtitlesby', 'amaraorg',
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audio', required=True)
    p.add_argument('--vad-audio', help='VAD source; defaults to --audio')
    p.add_argument('--file-id', required=True)
    p.add_argument('--sessions', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--model', default='large-v3')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--compute-type', default='auto')
    p.add_argument('--start', type=float, default=0)
    p.add_argument('--duration', type=float, default=0)
    p.add_argument('--tag', default='main')
    p.add_argument('--beam-size', type=int, default=5)
    p.add_argument('--patience', type=float, default=1.0)
    p.add_argument('--temperatures', default='0,0.2,0.4,0.6,0.8,1.0')
    p.add_argument('--languages', default='ko,en')
    p.add_argument('--lang-margin', type=float, default=0.15)
    p.add_argument('--lang-min-s', type=float, default=3.0)
    p.add_argument('--lang-short-conf', type=float, default=0.9)
    p.add_argument('--use-hotwords', choices=[0, 1], type=int, default=1)
    p.add_argument('--hotword-token-budget', type=int, default=120)
    p.add_argument('--context-token-budget', type=int, default=80)
    p.add_argument('--vad-threshold', type=float, default=0.5)
    p.add_argument('--vad-min-silence-ms', type=int, default=500)
    p.add_argument('--vad-speech-pad-ms', type=int, default=300)
    p.add_argument('--max-chunk-s', type=float, default=28)
    p.add_argument('--min-chunk-s', type=float, default=0)
    p.add_argument('--fill-gaps-s', type=float, default=0,
                   help='Fill uncovered stretches at least this long; 0 disables gap filling')
    p.add_argument('--drop-stock-phrases', choices=[0, 1], type=int, default=1)
    p.add_argument('--max-merge-gap-s', type=float, default=2.0)
    p.add_argument('--compression-ratio-threshold', type=float, default=2.4)
    p.add_argument('--log-prob-threshold', type=float, default=-1.0)
    p.add_argument('--no-speech-threshold', type=float, default=0.6)
    p.add_argument('--repetition-penalty', type=float, default=1.0)
    p.add_argument('--no-repeat-ngram-size', type=int, default=0)
    p.add_argument('--block-s', type=float, default=600)
    p.add_argument('--context-chars', type=int, default=0)
    p.add_argument('--hallucination-silence-s', type=float, default=0)
    args = p.parse_args()
    args.vad_audio = args.vad_audio or args.audio
    args.temperatures = [float(x) for x in args.temperatures.split(',')]
    args.languages = list(dict.fromkeys(x.strip() for x in args.languages.split(',') if x.strip()))
    for name in ('start', 'duration', 'lang_margin', 'lang_min_s', 'max_merge_gap_s',
                 'vad_min_silence_ms', 'vad_speech_pad_ms', 'no_repeat_ngram_size',
                 'context_chars', 'hallucination_silence_s', 'min_chunk_s', 'fill_gaps_s',
                 'hotword_token_budget', 'context_token_budget'):
        if getattr(args, name) < 0:
            p.error(f'--{name.replace("_", "-")} must be nonnegative')
    for name in ('max_chunk_s', 'block_s', 'beam_size', 'patience'):
        if getattr(args, name) <= 0:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if not args.languages or not args.temperatures:
        p.error('languages and temperatures must be nonempty')
    if not 0 <= args.lang_short_conf <= 1:
        p.error('--lang-short-conf must be between 0 and 1')
    for value in (args.tag, args.file_id):
        if value in ('.', '..') or any(c in value for c in '/\\:'):
            p.error('tag and file-id must be single path components')
    return args


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.name + '.', suffix='.tmp', delete=False) as f:
            temp = f.name
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if temp and os.path.exists(temp):
            os.unlink(temp)


def make_chunks(spans, max_samples, max_merge_gap_s=2.0):
    """Merge nearby speech spans; split at long silences or the size limit."""
    max_gap_samples = round(max_merge_gap_s * SAMPLE_RATE)
    chunks = []
    current = None
    for span in spans:
        start, end = int(span['start']), int(span['end'])
        while end - start > max_samples:
            if current:
                chunks.append(current)
                current = None
            chunks.append((start, start + max_samples))
            start += max_samples
        if end <= start:
            continue
        if (current and end - current[0] <= max_samples
                and start - current[1] <= max_gap_samples):
            current = (current[0], end)
        else:
            if current:
                chunks.append(current)
            current = (start, end)
    if current:
        chunks.append(current)
    return chunks


def fill_gaps(chunks, audio, fill_gaps_s, max_samples):
    """Fill the complement of VAD chunks in range-relative sample coordinates.

    Cut at the end of the quietest complete 0.2s frame in the last 6s
    of each maximum window. Ties prefer the latest cut. Only the explicit
    sub-second tail merge may exceed max_samples (up to Whisper's 30s).
    Return merged intervals, parallel source labels, and dropped tail seconds.
    """
    tagged = [(start, end, 'vad') for start, end in chunks]
    if fill_gaps_s <= 0:
        return list(chunks), ['vad'] * len(chunks), 0.0
    frame = round(.2 * SAMPLE_RATE)
    search = 6 * SAMPLE_RATE
    dropped = 0
    cursor = 0
    gaps = []
    for start, end in sorted(chunks):
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(audio):
        gaps.append((cursor, len(audio)))
    for start, end in gaps:
        if (end - start) / SAMPLE_RATE < fill_gaps_s:
            continue
        windows = []
        while start < end:
            remaining = end - start
            if remaining < SAMPLE_RATE:
                if windows and end - windows[-1][0] <= 30 * SAMPLE_RATE:
                    windows[-1] = (windows[-1][0], end)
                else:
                    dropped += remaining
                break
            boundary = min(end, start + max_samples)
            if remaining > max_samples and max_samples >= frame:
                # Anchor frames at the maximum boundary; no frame crosses it.
                candidates = range(boundary, max(start + frame - 1, boundary - search), -frame)
                best_rms = float('inf')
                best_boundary = boundary
                for cut in candidates:
                    samples = audio[cut - frame:cut].astype(np.float64)
                    rms = float(np.sqrt(np.mean(samples * samples)))
                    if rms < best_rms:
                        best_rms, best_boundary = rms, cut
                boundary = best_boundary
            windows.append((start, boundary))
            start = boundary
        tagged.extend((s, e, 'gap') for s, e in windows)
    tagged.sort(key=lambda chunk: chunk[0])
    return ([(s, e) for s, e, source in tagged],
            [source for s, e, source in tagged], dropped / SAMPLE_RATE)


def make_blocks(chunks, block_samples):
    blocks, current = [], []
    for index, (start, end) in enumerate(chunks):
        if current and end - chunks[current[0]][0] > block_samples:
            blocks.append(current)
            current = []
        current.append(index)
    if current:
        blocks.append(current)
    return blocks


class GPUMonitor:
    def __init__(self, enabled):
        self.enabled = enabled
        self.name = None
        self.peak = None
        self.error = None
        self.stop_event = threading.Event()
        self.thread = None

    def poll(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.used', '--format=csv,noheader,nounits'],
                check=True, capture_output=True, text=True, timeout=5)
            # WhisperModel uses GPU 0 by default. Report that same GPU.
            name, memory = result.stdout.strip().splitlines()[0].rsplit(',', 1)
            self.name = name.strip()
            self.peak = max(self.peak or 0, float(memory.strip()))
        except Exception as exc:
            self.error = str(exc)

    def watch(self):
        while not self.stop_event.wait(2):
            self.poll()

    def __enter__(self):
        if self.enabled:
            self.poll()
            self.thread = threading.Thread(target=self.watch, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            self.stop_event.set()
            self.thread.join(timeout=6)
            self.poll()


def choose_language(model, audio, allowed, previous, margin, lang_min_s=3.0, lang_short_conf=0.9):
    _language, _probability, all_probs = model.detect_language(audio=audio)
    probs = {lang: float(dict(all_probs).get(lang, 0.0)) for lang in allowed}
    ordered = sorted(allowed, key=lambda lang: probs[lang], reverse=True)
    chosen = ordered[0]
    allowed_total = sum(probs.values())
    confidence = probs[chosen] / allowed_total if allowed_total else 0.0
    if len(audio) / SAMPLE_RATE < lang_min_s and confidence < lang_short_conf:
        return previous, probs, 'inherited_short'
    if len(ordered) > 1 and probs[ordered[0]] - probs[ordered[1]] < margin:
        return previous, probs, 'inherited_margin'
    return chosen, probs, 'detected'


def context_prompt(previous, language, start, limit):
    """Take a safe suffix of the immediately preceding chunk, at a word boundary."""
    if (not previous or limit <= 0 or previous['language'] != language
            or not 0 <= start - previous['end'] <= 5
            or previous['max_compression_ratio'] > 2.0
            or previous['max_temperature'] > 0.4):
        return ''
    text = previous['last_text_tail']
    cut = max(0, len(text) - limit)
    # If the suffix begins inside a word, advance over that word.
    if cut and not text[cut - 1].isspace():
        while cut < len(text) and not text[cut].isspace():
            cut += 1
    return text[cut:].strip()


def context_state(segments, language, end, limit):
    text = ' '.join(s['text'].strip() for s in segments if not s.get('dropped')).strip()
    # Keep one extra character so the next call can distinguish an intact word
    # from a suffix cut in its middle. This state is serialized across blocks.
    return dict(last_text_tail=text[-(limit + 1):] if limit else '', language=language,
                end=end, max_compression_ratio=max((s['compression_ratio'] for s in segments), default=0),
                max_temperature=max((s['temperature'] for s in segments), default=0))


def prompt_text_tokens(tokenizer, text):
    """Match WhisperModel.generate_segments/get_prompt, including their space."""
    return tokenizer.encode(' ' + text.strip()) if text else []


def budget_hotwords(terms, tokenizer, budget):
    """Keep a priority prefix; never skip a large term to admit later terms."""
    used, count = [], 0
    for term in terms:
        candidate_count = len(prompt_text_tokens(tokenizer, ', '.join(used + [term])))
        if candidate_count > budget:
            break
        used.append(term)
        count = candidate_count
    return used, list(terms[len(used):]), count


def trim_context_tokens(text, tokenizer, budget):
    """Keep a suffix at an intact word boundary, after the character cut."""
    cut = 0
    while cut < len(text):
        count = len(prompt_text_tokens(tokenizer, text[cut:]))
        if count <= budget:
            return text[cut:], count
        while cut < len(text) and not text[cut].isspace():
            cut += 1
        while cut < len(text) and text[cut].isspace():
            cut += 1
    return '', 0


def normalize_text(text):
    return ''.join(c for c in text.lower()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def stock_phrase_remainder(text):
    normal = normalize_text(text)
    for phrase in sorted(STOCK_PHRASES, key=len, reverse=True):
        normal = normal.replace(phrase, '')
    return normal


def drop_stock_phrases(segments):
    for segment in segments:
        if len(stock_phrase_remainder(segment['text'])) < 3:
            segment['dropped'] = 'stock'
    return segments


def drop_duplicates(segments):
    previous, previous_chunk = None, None
    for segment in segments:
        normal = normalize_text(segment['text'])
        if segment['chunk'] == previous_chunk and len(normal) >= 4 and normal == previous:
            segment['dropped'] = 'dup'
        previous, previous_chunk = normal, segment['chunk']
    return segments


def load_saved_block(path, config_hash, indices, previous, incoming_context=None):
    try:
        candidate = json.loads(path.read_text(encoding='utf-8'))
        if (candidate.get('config_hash') == config_hash
                and candidate.get('chunk_indices') == indices
                and candidate.get('previous_language') == previous
                and 'incoming_context' in candidate
                and candidate['incoming_context'] == incoming_context):
            return candidate
    except (ValueError, OSError):
        pass
    return None


def transcribe_chunk(model, audio, chunk_index, absolute_start, language, hotwords, args, prompt=''):
    extra = {'initial_prompt': prompt or None}
    if args.hallucination_silence_s > 0:
        extra['hallucination_silence_threshold'] = args.hallucination_silence_s
    generated, _info = model.transcribe(
        audio, language=language, task='transcribe', condition_on_previous_text=False,
        word_timestamps=True, vad_filter=False, beam_size=args.beam_size,
        patience=args.patience, temperature=args.temperatures,
        compression_ratio_threshold=args.compression_ratio_threshold,
        log_prob_threshold=args.log_prob_threshold, no_speech_threshold=args.no_speech_threshold,
        repetition_penalty=args.repetition_penalty, no_repeat_ngram_size=args.no_repeat_ngram_size,
        hotwords=hotwords, **extra)
    result = []
    limit = absolute_start + len(audio) / SAMPLE_RATE
    previous_end = absolute_start
    for segment in generated:
        start = min(limit, max(previous_end, absolute_start + float(segment.start)))
        end = min(limit, max(start, absolute_start + float(segment.end)))
        words = []
        word_end = start
        for word in segment.words or []:
            ws = min(end, max(word_end, start, absolute_start + float(word.start)))
            we = min(end, max(ws, absolute_start + float(word.end)))
            words.append({'start': ws, 'end': we, 'word': word.word, 'prob': float(word.probability)})
            word_end = we
        result.append({'chunk': chunk_index, 'start': start, 'end': end, 'text': segment.text,
                       'language': language, 'avg_logprob': float(segment.avg_logprob),
                       'compression_ratio': float(segment.compression_ratio),
                       'no_speech_prob': float(segment.no_speech_prob),
                       'temperature': float(segment.temperature), 'words': words,
                       'tokens': list(getattr(segment, 'tokens', []))})
        previous_end = end
    result = drop_duplicates(result)
    return drop_stock_phrases(result) if getattr(args, 'drop_stock_phrases', 1) else result


def run(args):
    entries = json.loads(Path(args.sessions).read_text(encoding='utf-8-sig'))
    session = next(s for s in entries if s['id'] == args.file_id)
    device = args.device
    if device == 'auto':
        device = 'cuda' if ctranslate2.get_cuda_device_count() > 0 else 'cpu'
    compute_type = args.compute_type
    if compute_type == 'auto':
        compute_type = 'float16' if device == 'cuda' else 'int8'
    # Budgeting must use this model's tokenizer before hashing the used terms.
    load_started = time.perf_counter()
    model = WhisperModel(args.model, device=device, compute_type=compute_type)
    model_load_s = time.perf_counter() - load_started
    print(f'model_load_s={model_load_s:.6f}', flush=True)
    tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual,
                          task='transcribe', language=session['default_language'])
    used, dropped, hotword_tokens = budget_hotwords(
        session['hotwords'], tokenizer, args.hotword_token_budget) if args.use_hotwords else ([], session['hotwords'], 0)
    config = vars(args).copy()
    config.update(device=device, compute_type=compute_type, hotwords=used,
                  hotwords_used=used, hotwords_dropped=dropped, hotword_tokens=hotword_tokens,
                  faster_whisper_version=importlib.metadata.version('faster-whisper'),
                  ctranslate2_version=importlib.metadata.version('ctranslate2'),
                  av_version=importlib.metadata.version('av'),
                  audio_sha256=file_sha256(args.audio), vad_audio_sha256=file_sha256(args.vad_audio),
                  core_sha256=file_sha256(__file__),
                  session_metadata=session)
    hash_config = {k: v for k, v in config.items()
                   if k not in ('out_dir', 'device', 'audio', 'vad_audio', 'sessions', 'hotwords_dropped')}
    # Dropped terms are diagnostics only; only the admitted prefix affects decoding.
    hash_config['session_metadata'] = dict(session, hotwords=used)
    config_hash = hashlib.sha256(json.dumps(hash_config, sort_keys=True, ensure_ascii=False,
                                           allow_nan=False).encode('utf-8')).hexdigest()
    config['config_hash'] = config_hash
    full_audio = decode_audio(args.audio, sampling_rate=SAMPLE_RATE)
    full_vad_audio = (full_audio if Path(args.vad_audio).resolve() == Path(args.audio).resolve()
                      else decode_audio(args.vad_audio, sampling_rate=SAMPLE_RATE))
    audio_length_diff_s = abs(len(full_audio) - len(full_vad_audio)) / SAMPLE_RATE
    if audio_length_diff_s > 0.5:
        raise ValueError(f'Audio/VAD length difference {audio_length_diff_s:.6f}s exceeds 0.5s')
    shared_length = min(len(full_audio), len(full_vad_audio))
    audio_duration = shared_length / SAMPLE_RATE
    begin = round(args.start * SAMPLE_RATE)
    if begin >= shared_length:
        raise ValueError(f'--start {args.start} is outside audio duration {audio_duration}')
    finish = min(shared_length, begin + round(args.duration * SAMPLE_RATE)) if args.duration else shared_length
    audio = full_audio[begin:finish]
    vad_audio = full_vad_audio[begin:finish]
    offset = begin / SAMPLE_RATE
    duration = len(audio) / SAMPLE_RATE
    spans = get_speech_timestamps(vad_audio, VadOptions(
        threshold=args.vad_threshold, min_silence_duration_ms=args.vad_min_silence_ms,
        speech_pad_ms=args.vad_speech_pad_ms))
    chunks = make_chunks(spans, max(1, int(args.max_chunk_s * SAMPLE_RATE)), args.max_merge_gap_s)
    chunks, chunk_sources, gap_remainder_dropped_s = fill_gaps(
        chunks, audio, args.fill_gaps_s, max(1, int(args.max_chunk_s * SAMPLE_RATE)))
    blocks = make_blocks(chunks, max(1, int(args.block_s * SAMPLE_RATE)))
    folder = Path(args.out_dir) / args.tag / args.file_id
    folder.mkdir(parents=True, exist_ok=True)
    previous = session['default_language']
    previous_context = None
    if previous not in args.languages:
        previous = args.languages[0]
    all_chunks, all_segments, stored_blocks = [], [], []
    total_wall = 0.0
    hotwords = ', '.join(used) or None
    with GPUMonitor(device == 'cuda') as gpu:
        for block_index, indices in enumerate(blocks):
            path = folder / f'block_{block_index:04d}.json'
            saved = load_saved_block(path, config_hash, indices, previous, previous_context)
            action = 'SKIP' if saved is not None else 'RUN'
            if saved is None:
                started = time.perf_counter()
                block_chunks, block_segments = [], []
                incoming_language = previous
                incoming_context = previous_context
                for index in indices:
                    start, end = chunks[index]
                    source = chunk_sources[index]
                    absolute_start = offset + start / SAMPLE_RATE
                    if (end - start) / SAMPLE_RATE < args.min_chunk_s:
                        block_chunks.append({'index': index, 'start': absolute_start,
                                             'end': offset + end / SAMPLE_RATE, 'skipped': 'short',
                                             'source': source})
                        continue
                    piece = audio[start:end]
                    language, probs, decision = choose_language(
                        model, piece, args.languages, previous, args.lang_margin,
                        args.lang_min_s, args.lang_short_conf)
                    prompt = context_prompt(previous_context, language, absolute_start, args.context_chars)
                    tokenizer = Tokenizer(model.hf_tokenizer, model.model.is_multilingual,
                                          task='transcribe', language=language)
                    prompt, context_tokens = trim_context_tokens(prompt, tokenizer, args.context_token_budget)
                    decoder_prompt = model.get_prompt(tokenizer, prompt_text_tokens(tokenizer, prompt),
                                                      without_timestamps=False, prefix=None, hotwords=hotwords)
                    block_chunks.append({'index': index, 'start': absolute_start,
                                         'source': source,
                                         'end': offset + end / SAMPLE_RATE, 'language': language,
                                         'lang_probs': probs, 'lang_decision': decision,
                                         'context_used': bool(prompt), 'context_chars': len(prompt),
                                         'initial_prompt': prompt, 'context_tokens': context_tokens,
                                         'prompt_tokens': len(decoder_prompt)})
                    chunk_segments = transcribe_chunk(model, piece, index, absolute_start,
                                                      language, hotwords, args, prompt)
                    for segment in chunk_segments:
                        segment['source'] = source
                    output_tokens = sum(len(s['tokens']) for s in chunk_segments)
                    word_ends = [w['end'] for s in chunk_segments for w in s['words']]
                    block_chunks[-1].update(output_tokens=output_tokens,
                                            near_limit=len(decoder_prompt) + output_tokens >= 430,
                                            tail_gap_s=offset + end / SAMPLE_RATE - max(word_ends) if word_ends else None)
                    block_segments.extend(chunk_segments)
                    previous_context = context_state(chunk_segments, language, offset + end / SAMPLE_RATE,
                                                     args.context_chars)
                    previous = language
                block_wall_s = time.perf_counter() - started
                if device == 'cuda':
                    gpu.poll()
                saved = {'block_index': block_index, 'config_hash': config_hash,
                         'chunk_indices': indices, 'previous_language': incoming_language,
                         'incoming_context': incoming_context, 'last_context': previous_context,
                         'last_text_tail': previous_context['last_text_tail'] if previous_context else '',
                         'last_language': previous, 'wall_s': block_wall_s,
                         'gpu_name': gpu.name, 'gpu_mem_peak_mb': gpu.peak,
                         'chunks': block_chunks, 'segments': block_segments}
                atomic_json(path, saved)
            previous = saved['last_language']
            previous_context = saved['last_context']
            stored_blocks.append(saved)
            total_wall += saved['wall_s']
            all_chunks.extend(saved['chunks'])
            all_segments.extend(saved['segments'])
            audio_done = duration if block_index == len(blocks) - 1 else chunks[indices[-1]][1] / SAMPLE_RATE
            rtf = total_wall / audio_done if audio_done else 0.0
            print(f'block {block_index:04d} {action} audio_s={audio_done:.3f} wall_s={total_wall:.3f} rtf={rtf:.6f}', flush=True)
    dropped_dup = sum(s.get('dropped') == 'dup' for s in all_segments)
    stock_segments = [s for s in all_segments if s.get('dropped') == 'stock']
    skipped_short = [c for c in all_chunks if c.get('skipped') == 'short']
    all_segments = [s for s in all_segments if not s.get('dropped')]
    for index, segment in enumerate(all_segments):
        segment['id'] = index
    lang_seconds = {lang: 0.0 for lang in args.languages}
    for chunk in all_chunks:
        if not chunk.get('skipped'):
            lang_seconds[chunk['language']] += chunk['end'] - chunk['start']
    peaks = [b['gpu_mem_peak_mb'] for b in stored_blocks if b['gpu_mem_peak_mb'] is not None]
    if gpu.peak is not None:
        peaks.append(gpu.peak)
    names = [b['gpu_name'] for b in stored_blocks if b['gpu_name']]
    lang_decisions = dict.fromkeys(('detected', 'inherited_margin', 'inherited_short'), 0)
    lang_decisions.update(collections.Counter(chunk['lang_decision'] for chunk in all_chunks if not chunk.get('skipped')))
    stats = {'wall_s': total_wall, 'rtf': total_wall / duration, 'model_load_s': model_load_s,
             'speech_s': sum(int(s['end']) - int(s['start']) for s in spans) / SAMPLE_RATE,
             'n_chunks': len(all_chunks), 'n_segments': len(all_segments), 'lang_seconds': lang_seconds,
             'gap_chunks': chunk_sources.count('gap'),
             'gap_chunk_s': sum(end - start for (start, end), source in zip(chunks, chunk_sources)
                                if source == 'gap') / SAMPLE_RATE,
             'gap_remainder_dropped_s': gap_remainder_dropped_s,
             'dropped_dup': dropped_dup,
             'audio_length_diff_s': audio_length_diff_s,
             'skipped_short_chunks': len(skipped_short),
             'skipped_short_s': sum(c['end'] - c['start'] for c in skipped_short),
             'dropped_stock': len(stock_segments),
             'dropped_stock_examples': [{k: s[k] for k in ('start', 'end', 'text')} for s in stock_segments[:20]],
             'near_limit_chunks': sum(c.get('near_limit', False) for c in all_chunks),
             'max_prompt_tokens': max((c.get('prompt_tokens', 0) for c in all_chunks), default=0),
             'lang_decisions': lang_decisions,
             'gpu_name': (gpu.name or (names[0] if names else None)) if device == 'cuda' else None,
             'gpu_mem_peak_mb': max(peaks) if peaks and device == 'cuda' else None}
    if device == 'cuda' and gpu.error:
        stats['gpu_monitor_error'] = gpu.error
    result = {'file_id': args.file_id, 'source_file': session['source_file'],
              'session': session['session'], 'recorded_kst': session['recorded_kst'],
              'audio_duration_s': audio_duration, 'range': {'start_s': offset, 'duration_s': duration},
              'config': config, 'stats': stats, 'chunks': all_chunks, 'segments': all_segments}
    atomic_json(folder.parent / (args.file_id + '.json'), result)
    return result


if __name__ == '__main__':
    # An uncaught exception prints its full traceback and gives a nonzero exit.
    run(parse_args())
