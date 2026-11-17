"""Drive-side orchestration. Standard library only outside guarded Colab calls."""
import collections
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import traceback
import zipfile


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def write_status(base, stage, **info):
    path = Path(base) / 'out/STATUS.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    event = dict(stage=stage, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **info)
    atomic_json(path, event | {'history': previous.get('history', []) + [event]})


def run_items(control):
    pilots = list(control.get('pilot_runs', [])) if control.get('run_pilot') else []
    full = [dict(tag=control['full_tag'], file_id=file_id, audio=control.get('full_audio', 'raw'),
                 vad_audio=control.get('full_vad_audio', control.get('full_audio', 'raw')),
                 args=control.get('full_args', '')) for file_id in control.get('full_ids', [])] if control.get('run_full') else []
    return pilots, full


def component(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError(f'Unsafe path component: {value!r}')
    return value


def measure_volume(audio, ffmpeg):
    result = subprocess.run([str(ffmpeg), '-hide_banner', '-i', str(audio), '-vn', '-af',
                             'volumedetect', '-f', 'null', '-'], capture_output=True,
                            text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError('\n'.join(result.stderr.splitlines()[-20:]))
    values = {}
    for key in ('mean_volume', 'max_volume'):
        match = re.search(key + r':\s*(-?inf|[-\d.]+) dB', result.stderr)
        values[key] = float(match[1]) if match and 'inf' not in match[1] else (match[1] if match else None)
    return values


def prepare_audio(base, control, work, ffmpeg='ffmpeg', runs=None):
    """Build only requested file/variant pairs; report cache decisions and levels."""
    base, work = Path(base), Path(work)
    pilots, full = run_items(control)
    needed = {(r['file_id'], variant)
              for r in (runs if runs is not None else pilots + full)
              for variant in (r.get('audio', 'raw'), r.get('vad_audio', r.get('audio', 'raw')))}
    variants = dict(control.get('audio_variants', {}), raw='')
    results = {}
    raw_cache = {}
    for file_id, variant in sorted(needed):
        component(file_id)
        component(variant)
        try:
            if variant not in variants:
                raise ValueError(f'Unknown audio variant: {variant}')
            source = base / 'audio' / (file_id + '.m4a')
            raw = work / 'raw' / source.name
            raw.parent.mkdir(parents=True, exist_ok=True)
            if not source.is_file():
                raise FileNotFoundError(str(source))
            if file_id not in raw_cache:
                copied = not raw.exists() or raw.stat().st_mtime_ns < source.stat().st_mtime_ns or raw.stat().st_size != source.stat().st_size
                if copied:
                    shutil.copy2(source, raw)
                raw_cache[file_id] = dict(path=str(raw), rebuilt=copied, **measure_volume(raw, ffmpeg))
                write_status(base, 'audio_ready', file_id=file_id, audio='raw', **raw_cache[file_id])
            if variant == 'raw':
                results[(file_id, variant)] = raw_cache[file_id]
                continue
            target = work / variant / (file_id + '.wav')
            sidecar = target.with_suffix('.filter.txt')
            filter_string = variants[variant]
            skip = (target.exists() and target.stat().st_mtime_ns > raw.stat().st_mtime_ns
                    and sidecar.exists() and sidecar.read_text(encoding='utf-8') == filter_string)
            if not skip:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Invalidate the sidecar before ffmpeg can overwrite a formerly valid output.
                sidecar.write_text('', encoding='utf-8')
                command = [str(ffmpeg), '-y', '-i', str(raw), '-vn', '-ac', '1', '-ar', '16000',
                           '-af', filter_string, '-c:a', 'pcm_s16le', str(target)]
                child = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
                if child.returncode:
                    raise RuntimeError('\n'.join(child.stderr.splitlines()[-20:]))
                sidecar.write_text(filter_string, encoding='utf-8')
            result = dict(path=str(target), rebuilt=not skip, **measure_volume(target, ffmpeg))
            results[(file_id, variant)] = result
            write_status(base, 'audio_ready', file_id=file_id, audio=variant, **result)
        except Exception as exc:
            write_status(base, 'audio_failed', file_id=file_id, audio=variant,
                         last_20_lines=str(exc).splitlines()[-20:])
            raise
    return results


def cli_for(base, run, audio, core, sessions, python, vad_audio=None):
    return [str(python), '-u', str(core), '--audio', str(audio), '--file-id', run['file_id'],
            '--sessions', str(sessions), '--out-dir', str(Path(base) / 'out'), '--tag', run['tag'],
            '--start', str(run.get('start', 0)), '--duration', str(run.get('duration', 0))] + (
                ['--vad-audio', str(vad_audio)] if vad_audio is not None else []) + shlex.split(run.get('args', ''))


def run_core(base, run, audio, core, sessions, python, vad_audio=None):
    component(run['tag'])
    component(run['file_id'])
    command = cli_for(base, run, audio, core, sessions, python, vad_audio)
    log_path = Path(base) / 'out' / run['tag'] / (run['file_id'] + '.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = collections.deque(maxlen=30)
    write_status(base, 'running', tag=run['tag'], file_id=run['file_id'])
    code = -1
    with log_path.open('w', encoding='utf-8') as log:
        try:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding='utf-8', errors='replace', bufsize=1) as child:
                for line in child.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                    tail.append(line.rstrip())
                code = child.wait()
        except Exception:
            message = traceback.format_exc()
            log.write(message)
            tail.extend(message.splitlines())
    result = dict(tag=run['tag'], file_id=run['file_id'], exit_code=code)
    write_status(base, 'file_done' if code == 0 else 'subprocess_failed', **result,
                 last_30_lines=list(tail) if code else [])
    return result


def timestamp(seconds):
    seconds = int(seconds)
    return f'{seconds // 60:02d}:{seconds % 60:02d}'


def read_result(base, run, result):
    path = Path(base) / 'out' / run['tag'] / (run['file_id'] + '.json')
    return json.loads(path.read_text(encoding='utf-8')) if result['exit_code'] == 0 and path.exists() else None


def write_pilot_report(base, name, runs, results):
    lines = ['# Pilot report', '']
    for run, result in zip(runs, results):
        lines.append(f"## {run['tag']} | {run['file_id']} | {run.get('start', 0)}+{run.get('duration', 0)}s | {run.get('audio', 'raw')} | {run.get('args', '')}")
        data = read_result(base, run, result)
        if data is None:
            lines.extend([f"FAILED exit={result['exit_code']}; see log and STATUS.json", ''])
            continue
        stats, segments = data['stats'], data['segments']
        mean = statistics.mean(s['avg_logprob'] for s in segments) if segments else None
        fields = {k: stats.get(k) for k in ('rtf', 'speech_s', 'n_chunks', 'n_segments', 'dropped_dup', 'lang_seconds',
                                          'max_prompt_tokens', 'near_limit_chunks', 'skipped_short_chunks', 'dropped_stock')}
        fields['vad_audio'] = run.get('vad_audio', run.get('audio', 'raw'))
        tail_chunks = [c for c in data['chunks'] if c.get('tail_gap_s') is not None and c['tail_gap_s'] > 3]
        fields.update(hotword_tokens=data.get('config', {}).get('hotword_tokens'),
                      hotwords_dropped=len(data.get('config', {}).get('hotwords_dropped', [])),
                      tail_gap_chunks=len(tail_chunks))
        fields.update(fallback_count=sum(s['temperature'] > 0 for s in segments), mean_avg_logprob=mean,
                      gpu_peak_mb=stats.get('gpu_mem_peak_mb'),
                      context_used=sum(c.get('context_used', False) for c in data['chunks']))
        lines.append(' | '.join(f'{k}={json.dumps(v, ensure_ascii=False)}' for k, v in fields.items()))
        lines.extend(f"[{timestamp(s['start'])}][{s['language']}] {s['text'].strip()}" for s in segments)
        lines.extend(f"[{timestamp(c['start'])}–{timestamp(c['end'])}] tail_gap={c['tail_gap_s']:.3f} "
                     f"prompt={c['prompt_tokens']} output={c['output_tokens']}" for c in tail_chunks)
        lines.append('')
    path = Path(base) / 'out' / f'report_{name}.md'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def pack(base, name, runs, artifacts, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out = Path(base) / 'out'
    paths = {out / 'STATUS.json', *map(Path, artifacts)}
    for run in runs:
        paths.update(out / run['tag'] / (run['file_id'] + suffix) for suffix in ('.json', '.log'))
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.is_file():
                archive.write(path, path.relative_to(out).as_posix())
    shutil.copy2(destination, out / destination.name)
    return destination


def main(BASE, control, CODE, setup_info, *, ffmpeg='ffmpeg', audio_work=None,
         core_path=None, python_executable=None, zip_dir=None):
    base, code = Path(BASE), Path(CODE)
    name = control.get('_name', 'control_unknown')
    try:
        from google.colab import drive, files
    except ImportError:
        drive = files = None
    work = Path(audio_work) if audio_work else (Path('/content/audio') if drive else base / '_audio')
    zip_dir = Path(zip_dir) if zip_dir else (Path('/content') if drive else base / '_pack')
    core_path = Path(core_path) if core_path else code / 'stt_core.py'
    python = python_executable or sys.executable
    pilots, full = [], []
    runs, artifacts, results = [], [], []
    error = None
    zip_path = zip_dir / f'aiday_{name}.zip'
    write_status(base, 'starting', control=name, versions=setup_info.get('versions', {}),
                 download_triggered=False, colab=drive is not None)
    try:
        component(name)
        pilots, full = run_items(control)
        runs = pilots + full
        for run in runs:
            component(run['tag'])
            component(run['file_id'])
            try:
                prepared = prepare_audio(base, control, work, ffmpeg, runs=[run])
                audio = prepared[(run['file_id'], run.get('audio', 'raw'))]['path']
                vad_audio = prepared[(run['file_id'], run.get('vad_audio', run.get('audio', 'raw')))]['path']
                result = run_core(base, run, audio, core_path, code / 'sessions.json', python, vad_audio)
            except Exception:
                message = traceback.format_exc()
                log = base / 'out' / run['tag'] / (run['file_id'] + '.log')
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text(message, encoding='utf-8')
                result = dict(tag=run['tag'], file_id=run['file_id'], exit_code=-1)
                write_status(base, 'run_failed', **result, last_30_lines=message.splitlines()[-30:])
            results.append(result)
        if control.get('run_pilot'):
            artifacts.append(write_pilot_report(base, name, pilots, results[:len(pilots)]))
        if control.get('run_full'):
            summary = []
            for run, result in zip(full, results[len(pilots):]):
                data = read_result(base, run, result)
                summary.append(result | (dict(duration=data['range']['duration_s'], **data['stats']) if data else {}))
            path = base / 'out' / f'full_summary_{name}.json'
            atomic_json(path, summary)
            artifacts.append(path)
    except Exception:
        error = traceback.format_exc()
    finally:
        try:
            write_status(base, 'failed' if error else 'done', control=name,
                         traceback=error, failed_runs=sum(r['exit_code'] != 0 for r in results),
                         download_triggered=files is not None, colab=drive is not None)
            pack(base, name, runs, artifacts, zip_path)
            if files is not None:
                files.download(str(zip_path))
        except Exception:
            error = traceback.format_exc()
            write_status(base, 'failed', traceback=error, download_triggered=False)
            # Retain the final failure state in a package when possible.
            try:
                pack(base, name, runs, artifacts, zip_path)
            except Exception:
                pass
        finally:
            if drive is not None:
                try:
                    drive.flush_and_unmount()
                except Exception:
                    error = traceback.format_exc()
                    print(error, flush=True)
                    write_status(base, 'failed', traceback=error, download_triggered=files is not None)
                    pack(base, name, runs, artifacts, zip_path)
        if error:
            raise RuntimeError(error)
    return dict(results=results, zip=str(zip_path))
