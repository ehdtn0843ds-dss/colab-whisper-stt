"""Turn a stt_core.py result JSON into a readable transcript (.txt) and subtitles (.srt).

Usage:
    python export_text.py out/full_v3/01_lecture.json             # writes .txt and .srt next to it
    python export_text.py out/full_v3/*.json --formats txt         # only .txt
"""
import argparse
import json
from pathlib import Path


def clock(seconds, srt=False):
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    if srt:
        return f'{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}'
    return f'{hours:02d}:{minutes:02d}:{secs:02d}'


def to_txt(data):
    head = [f"# {data.get('session', '')}", f"# {data.get('source_file', '')} | {data.get('recorded_kst', '')}", '']
    body = [f"[{clock(s['start'])}] [{s['language']}] {s['text'].strip()}" for s in data['segments'] if s['text'].strip()]
    return '\n'.join(head + body) + '\n'


def to_srt(data):
    blocks = []
    segments = [s for s in data['segments'] if s['text'].strip()]
    for number, s in enumerate(segments, 1):
        blocks.append(f"{number}\n{clock(s['start'], True)} --> {clock(s['end'], True)}\n{s['text'].strip()}\n")
    return '\n'.join(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('results', nargs='+', type=Path)
    parser.add_argument('--formats', default='txt,srt')
    args = parser.parse_args()
    writers = {'txt': to_txt, 'srt': to_srt}
    for path in args.results:
        data = json.loads(path.read_text(encoding='utf-8'))
        for fmt in args.formats.split(','):
            target = path.with_suffix('.' + fmt)
            target.write_text(writers[fmt](data), encoding='utf-8')
            print('wrote', target)


if __name__ == '__main__':
    main()
