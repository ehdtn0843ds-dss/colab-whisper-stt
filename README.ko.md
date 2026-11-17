# colab-whisper-stt

[English](README.md) | **한국어**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ehdtn0843ds-dss/colab-whisper-stt/blob/main/notebook/colab_whisper_stt.ipynb)
[![Guide](https://img.shields.io/badge/guide-EN%20%7C%20%ED%95%9C%EA%B5%AD%EC%96%B4-f26b1d)](https://ehdtn0843ds-dss.github.io/colab-whisper-stt/?lang=ko)

**한국어와 영어가 섞인 녹음을 위한 Colab Whisper.**

영어 용어가 가득한 강의, 말하다가 언어가 바뀌는 회의, 사회는 한국어인데 발표는 영어인 컨퍼런스, 인터뷰, 팟캐스트. 기본 Whisper는 이런 녹음에서 한 언어로만 받아 적거나, 전문 용어를 엉뚱하게 쓰거나, 조용한 구간에 없는 문장을 만들어 낸다. 이 노트북은 무료 Colab T4에서 **faster-whisper large-v3**를 돌리면서 이 문제들을 처리한다.

📖 **전체 가이드 (블로그 형식, English / 한국어): https://ehdtn0843ds-dss.github.io/colab-whisper-stt/?lang=ko**

## 이런 녹음에 맞는다

- 🎓 영어 전문 용어가 섞인 한국어 강의·세미나
- 💼 한국어와 영어가 오가는 회의
- 🎤 컨퍼런스 발표, 패널 토론, Q&A
- 🎙️ 두 언어가 섞인 인터뷰, 팟캐스트, 유튜브 영상

## 하는 일

| 기본 Whisper의 문제 | 이 노트북의 대응 |
|---|---|
| 한 파일 안에서 한국어와 영어가 바뀜 | **발화 조각마다** `ko`/`en`을 고르고, 짧아서 애매한 조각은 직전 언어를 이어받음 |
| 이름과 전문 용어가 틀리게 나옴 | 녹음별 **hotword 목록**을 우선순위대로 120토큰 한도 안에서 넣음 |
| 무음·음악·박수 구간에 없는 문장이 생김 | VAD로 잘라낸 발화 조각만 전사하고, 반복 세그먼트와 자막 크레딧류 문구를 지움 |
| 긴 파일 도중에 Colab 연결이 끊김 | 약 10분 블록마다 Drive에 저장하고, 재시작하면 **끝난 블록은 건너뜀** |

## 속도

무료 Colab의 Tesla T4로 79분짜리 컨퍼런스 녹음(영어 약 56분, 한국어 약 22분)을 돌린 결과다.

| 모델 | 디코딩 시간 | RTF | temperature fallback | 평균 avg_logprob |
|---|---|---|---|---|
| large-v3 | 542초 (약 9분) | 0.114 | 16 | −0.49 |
| large-v2 | 865초 | 0.182 | 53 | −0.59 |

최대 GPU 메모리는 15 GB 중 4.9 GB였다. 녹음 하나에서 나온 속도·신뢰도 수치이고, 단어 오류율(WER)은 재지 않았다. 실제로 쓴 세션·control 파일은 [`examples/real-run-conference/`](examples/real-run-conference/)에 있다.

## 빠른 시작

1. **Open in Colab**을 누르고 `런타임 → 런타임 유형 변경 → T4 GPU`로 바꾼다.
2. **SETUP**, **BOOTSTRAP** 순서로 실행한다. `MyDrive/whisper_stt/`가 만들어지고 코드와 템플릿이 복사된다.
3. 녹음을 `MyDrive/whisper_stt/audio/<id>.m4a`에 넣고 `code/sessions.json`에 녹음마다 주 언어와 hotword를 적는다.
4. 파일럿 control 파일(2분 구간, 설정 두 개)로 먼저 돌려 보고, 그다음 [`examples/control_002_full.json`](examples/control_002_full.json)으로 바꾼다.
5. **MOUNT_AND_LOAD**, **RUN**을 실행한다. 끝나면 전사 결과 zip이 내려받아진다.

```
MyDrive/whisper_stt/
├── control_001_pilot.json     # 이름순으로 가장 뒤의 control_*.json을 사용
├── audio/01_lecture.m4a       # 파일 이름 = sessions.json의 id
├── code/stt_core.py, runner.py, sessions.json
└── out/                       # 결과, 로그, 체크포인트, STATUS.json
```

`sessions.json` 항목 예시:

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

결과를 일반 텍스트와 자막으로 바꾸려면:

```bash
python code/export_text.py out/full_v3/01_lecture.json   # .txt와 .srt 생성
```

## 저장소 구성

| 경로 | 내용 |
|---|---|
| [`notebook/colab_whisper_stt.ipynb`](notebook/colab_whisper_stt.ipynb) | Colab 노트북: SETUP → BOOTSTRAP → MOUNT_AND_LOAD → RUN |
| [`code/stt_core.py`](code/stt_core.py) | 전사: VAD 조각, gap filling, 조각별 언어, hotword + 문맥 프롬프트, 이어서 돌리기 |
| [`code/runner.py`](code/runner.py) | 전체 흐름 관리 (표준 라이브러리만 사용): 오디오 준비, 하위 프로세스, 로그, 리포트, zip |
| [`code/export_text.py`](code/export_text.py) | 결과 JSON → `.txt` / `.srt` |
| [`code/sessions.json`](code/sessions.json) | 녹음 설명 템플릿 |
| [`examples/`](examples/) | control 파일 템플릿, 실제 컨퍼런스 실행 예시 |
| [`docs/index.html`](docs/index.html) | 한·영 가이드 (GitHub Pages) |
| [`tests/`](tests/) | GPU 없이 도는 테스트 (`python -m pytest -q tests`) |

## 고정 버전

faster-whisper 1.2.1 · CTranslate2 4.8.5 · PyAV 16.0.1 · CUDA 12 cuBLAS/cuDNN 휠 (SETUP 셀이 설치)

`--languages`에는 Whisper가 지원하는 언어를 넣을 수 있다(예: `ja,en`). 다만 기본값과 환각 필터는 한국어·영어 녹음 기준으로 맞춰 두었다.

## 참고

- 녹음할 권리가 있는 녹음만 전사한다. 이 저장소에는 오디오와 전사본이 없다.
- 코드는 [MIT 라이선스](LICENSE)를 따른다. [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2), [Silero VAD](https://github.com/snakers4/silero-vad)를 바탕으로 한다.
