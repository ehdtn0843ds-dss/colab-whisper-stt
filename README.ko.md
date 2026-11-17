# colab-whisper-stt

[English](README.md) | **한국어**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ehdtn0843ds-dss/colab-whisper-stt/blob/main/notebook/colab_whisper_stt.ipynb)
[![Guide](https://img.shields.io/badge/guide-EN%20%7C%20%ED%95%9C%EA%B5%AD%EC%96%B4-76b900)](https://ehdtn0843ds-dss.github.io/colab-whisper-stt/?lang=ko)

한국어와 영어가 섞인 긴 발표 녹음을 무료 Colab T4 GPU에서 **faster-whisper large-v3**로 전사한다.
NVIDIA AI Day Seoul 2026에서 녹음한 세션들을 글로 옮기려고 만들었다. 사회는 한국어, 키노트는 영어였고, 부스 시연은 두 언어가 섞인 데다 제품명이 많이 나왔다.

📖 **전체 가이드 (블로그 형식, English / 한국어): https://ehdtn0843ds-dss.github.io/colab-whisper-stt/?lang=ko**

## 하는 일

| 기본 Whisper의 문제 | 이 파이프라인의 대응 |
|---|---|
| 세션 안에서 한국어와 영어가 바뀜 | 발화 조각마다 `ko`/`en`을 고르고, 짧아서 애매한 조각은 직전 언어를 이어받음 |
| 제품명이 틀리게 나옴 | 세션별 hotword 목록을 우선순위대로 120토큰 한도 안에서 넣음 |
| 음악·박수 구간에 없는 문장이 생김 | VAD 조각만 전사하고, 반복 세그먼트와 자막 크레딧류 문구를 지움 |
| 파일 도중에 Colab 연결이 끊김 | 약 10분 블록마다 Drive에 저장하고, 재시작하면 끝난 블록은 건너뜀 |

## 실제 실행 한 번 (T4, control_006)

| 실행 | 모델 | 오디오 | RTF | fallback | 평균 avg_logprob |
|---|---|---|---|---|---|
| `full_v3` | large-v3 | 79분 키노트 | 0.114 (542초) | 16 | −0.49 |
| `full_v2` | large-v2 | 같은 파일 | 0.182 (865초) | 53 | −0.59 |
| `patch_qna_en` | large-v3, `en` 고정 | Q&A 53초 구간 | 0.086 | 0 | −0.42 |

최대 GPU 메모리는 4.9 GB였다. 녹음 하나에서 나온 속도·신뢰도 수치이고, 단어 오류율(WER)은 재지 않았다.

## 빠른 시작

1. **Open in Colab**을 누르고 `런타임 → 런타임 유형 변경 → T4 GPU`로 바꾼다.
2. **SETUP**, **BOOTSTRAP** 순서로 실행한다. `MyDrive/aiday_stt/`가 만들어지고 코드와 템플릿이 복사된다.
3. 녹음을 `MyDrive/aiday_stt/audio/<id>.m4a`에 넣고 `code/sessions.json`에 설명을 적는다.
4. control 파일을 고친다 (파일럿부터 돌려 보고, 그다음 [`examples/control_002_full.json`](examples/control_002_full.json)).
5. **MOUNT_AND_LOAD**, **RUN**을 실행한다. 끝나면 전사 결과 zip이 내려받아진다.

```
MyDrive/aiday_stt/
├── control_001_pilot.json     # 이름순으로 가장 뒤의 control_*.json을 사용
├── audio/01_keynote.m4a       # 파일 이름 = sessions.json의 id
├── code/stt_core.py, runner.py, sessions.json
└── out/                       # 결과, 로그, 체크포인트, STATUS.json
```

결과를 일반 텍스트와 자막으로 바꾸려면:

```bash
python code/export_text.py out/full_v3/01_keynote.json   # .txt와 .srt 생성
```

## 저장소 구성

| 경로 | 내용 |
|---|---|
| [`notebook/colab_whisper_stt.ipynb`](notebook/colab_whisper_stt.ipynb) | Colab 노트북: SETUP → BOOTSTRAP → MOUNT_AND_LOAD → RUN |
| [`code/stt_core.py`](code/stt_core.py) | 전사: VAD 조각, gap filling, 조각별 언어, hotword + 문맥 프롬프트, 이어서 돌리기 |
| [`code/runner.py`](code/runner.py) | 전체 흐름 관리 (표준 라이브러리만 사용): 오디오 준비, 하위 프로세스, 로그, 리포트, zip |
| [`code/export_text.py`](code/export_text.py) | 결과 JSON → `.txt` / `.srt` |
| [`code/sessions.json`](code/sessions.json) | 녹음 설명 템플릿 |
| [`examples/`](examples/) | control 파일 템플릿, 실제 AI Day 세션·control 파일 |
| [`docs/index.html`](docs/index.html) | 한·영 가이드 (GitHub Pages) |
| [`tests/`](tests/) | GPU 없이 도는 테스트 (`python -m pytest -q tests`) |

## 고정 버전

faster-whisper 1.2.1 · CTranslate2 4.8.5 · PyAV 16.0.1 · CUDA 12 cuBLAS/cuDNN 휠 (SETUP 셀이 설치)

## 참고

- 녹음과 전사는 행사가 허용하는 범위에서만 한다. 이 저장소에는 오디오와 전사본이 없다.
- 코드는 [MIT 라이선스](LICENSE)를 따른다. [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2), [Silero VAD](https://github.com/snakers4/silero-vad)를 바탕으로 한다.
