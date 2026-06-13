# AGENTS.md — TurtleNeckDetector

## 프로젝트 개요

웹캠 기반 실시간 거북목 감지 앱. 시스템 트레이(Windows / macOS)에서 백그라운드로 동작하며, MediaPipe Pose로 자세를 판정하고 OS 알림을 발송한다. Firebase 인증 + Firestore 연동으로 로그인 사용자의 자세 통계를 클라우드에 저장한다.

자세한 내용은 아래 문서를 반드시 참고한다.

- 답변은 항상 토큰을 최적화해서 효율적으로 답변을 해준다.
- **[README.md](README.md)** — 빠른 시작 가이드
- **[DEVELOP.md](DEVELOP.md)** — 전체 아키텍처, 모듈 설명, 데이터 흐름, 개발 현황
- ** 항상 개발은 windows, MacOS 에서 둘 다 작동이 가능하게 구현할 것 **

## 주요 진입점

| 파일 | 역할 |
|---|---|
| `turtle_neck.py` | 앱 진입점. `AppState` + 스레드 오케스트레이션 |
| `src/detector.py` | MediaPipe 자세 점수 계산 + 히스테리시스 판정 |
| `src/auth.py` | Firebase Auth REST API + 세션 영속화 |
| `src/startup_window.py` | tkinter UI (StartupWindow / SettingsWindow / AuthWindow) |
| `src/tray_app.py` | pystray 트레이 아이콘 + OS 알림 |
| `src/logger.py` | JSON Lines 로컬 로그 저장 |
| `src/utils/firebase_uploader.py` | Firestore 업로드 |
| `src/utils/upload_queue.py` | JSONL 기반 오프라인 업로드 큐 |

## 스레드 구조

- **메인 스레드**: tkinter mainloop + `_poll()` (200ms 간격 `tk_queue` 소비)
- **Daemon 스레드 1**: `camera_loop()` — 웹캠 캡처 → 자세 판정 → 로그 저장
- **Daemon 스레드 2**: `upload_loop()` — 60초 간격 Firestore 업로드
- **Daemon 스레드 3**: pystray 트레이 루프

배경 스레드 → UI 호출은 반드시 `app.tk_queue`를 거쳐야 한다.

## 설정 파일

- `config.json` — 거북목 임계값(`delta_turtle`, `delta_ok`), 로그 flush 주기(`save_interval_seconds`)
- `.env` — `FIREBASE_API_KEY` (`.gitignore`에 포함, `.env.example` 참고)
- `firebase_key.json` — Firestore 서비스 계정 키 (`.gitignore`에 포함, 없으면 업로드 비활성)

## 런타임 환경

- Python 3.10+ 필수 (`float | None` 타입 힌트 사용)
- 의존성: `requirements.txt` 참고
- 생성 파일: `logs/` 디렉터리 자동 생성 (`.gitignore` 적용)
