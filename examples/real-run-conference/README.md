# Real run: a full day of conference recordings

These are the actual files from the run that produced the numbers in the main README and the guide: five recordings from a tech conference (NVIDIA AI Day Seoul 2026), with a Korean host, English keynotes, Korean booth demos full of English product names, and an expert Q&A in both languages.

| File | What it shows |
|---|---|
| `sessions.json` | Five sessions with `default_language` and hotword lists of 17 to 42 terms each |
| `control_004.json` | Every file with large-v3 and large-v2, side by side |
| `control_006.json` | The final run: one long file with both models, plus a 53-second passage re-run with the language forced to English |

They reference `code_v3` / `code_v4`, the code folder names used at the time. To reuse them, set `code_dir` to `code`.

---

# 실제 실행 예시: 컨퍼런스 하루치 녹음

메인 README와 가이드에 나온 수치를 만든 실행에서 실제로 쓴 파일이다. 기술 컨퍼런스(NVIDIA AI Day Seoul 2026) 녹음 다섯 개로, 한국어 사회, 영어 키노트, 영어 제품명이 섞인 한국어 부스 시연, 한·영 전문가 Q&A가 들어 있다.

| 파일 | 내용 |
|---|---|
| `sessions.json` | 세션 다섯 개의 `default_language`와 hotword 목록 (세션당 17~42개) |
| `control_004.json` | 모든 파일을 large-v3와 large-v2로 나란히 돌린 설정 |
| `control_006.json` | 마지막 실행: 긴 파일 하나를 두 모델로 돌리고, 53초 구간을 영어 고정으로 다시 돌린 설정 |

당시 코드 폴더 이름인 `code_v3` / `code_v4`를 가리키고 있으니, 다시 쓰려면 `code_dir`를 `code`로 바꾼다.
