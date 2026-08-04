# AGENTS.md — TurtleNeckDetector

## 프로젝트 개요

**CNN 전이학습 기반 실시간 거북목 감지 및 자세 교정 코칭 시스템** (디지털 헬스케어 AI 헬스테크)

웹캠 기반 실시간 거북목 감지 앱. 시스템 트레이(Windows / macOS)에서 백그라운드로 동작하며, MediaPipe Pose로 자세를 판정하고 OS 알림을 발송한다. 로그인 없이 로컬 전용으로 동작하며(Steam/MS Store/Google Play 배포를 염두에 두고 계정 기능은 의도적으로 제외), 로그는 `logs/local/`에 JSON Lines로 저장된다.

- **현재(v1.0)**: MediaPipe Pose 랜드마크 + 규칙 기반 임계값으로 거북목 판정
- **목표(v2.0)**: EfficientNet-B0 전이학습 모델로 판정 엔진 교체 + LLM 기반 코칭 피드백 추가

자세한 내용은 아래 문서를 반드시 참고한다.

- 답변은 항상 토큰을 최적화해서 효율적으로 답변을 해준다.
- **[README.md](README.md)** — 빠른 시작 가이드
- **[DEVELOP.md](DEVELOP.md)** — 전체 아키텍처, 모듈 설명, 데이터 흐름, 개발 현황, v2.0 로드맵
- **항상 개발은 Windows, macOS 에서 둘 다 작동이 가능하게 구현할 것**

## 주요 진입점

| 파일 | 역할 |
|---|---|
| `turtleCheck.py` | 앱 진입점. `AppState` + 스레드 오케스트레이션 |
| `src/detector.py` | MediaPipe 자세 점수 계산 + 히스테리시스 판정 |
| `src/startup_window.py` | tkinter UI (StartupWindow / SettingsWindow) |
| `src/tray_app.py` | pystray 트레이 아이콘 + OS 알림 |
| `src/logger.py` | JSON Lines 로컬 로그 저장 |
| `src/log_config.py` | 앱 전체 로깅 설정 |
| `src/utils/notifier.py` | OS별 알림 추상화 (Windows/macOS) |

## 스레드 구조

- **메인 스레드**: tkinter mainloop + `_poll()` (200ms 간격 `tk_queue` 소비)
- **Daemon 스레드 1**: `camera_loop()` — 웹캠 캡처 → 자세 판정 → 로그 저장
- **Daemon 스레드 2**: pystray 트레이 루프

배경 스레드 → UI 호출은 반드시 `app.tk_queue`를 거쳐야 한다.

## 설정 파일

- `config.json` — 거북목 임계값(`delta_turtle`, `delta_ok`), 로그 flush 주기(`save_interval_seconds`)
- `.env` — 환경 변수 (필요 시 사용, `.gitignore`에 포함, `.env.example` 참고)

## 로그인/계정 기능

의도적으로 제거된 상태다. Firebase → Supabase Auth 이전까지 시도했으나, 최종 배포 목표(Steam/MS Store/Google Play)를 고려해 핵심 감지 기능 우선으로 전면 제거했다. 관련 코드를 다시 추가하기 전에 `DEVELOP.md`의 [다음 과제](DEVELOP.md#다음-과제) 섹션을 먼저 확인할 것.

## 런타임 환경

- Python 3.10+ 필수 (`float | None` 타입 힌트 사용)
- 의존성: `requirements.txt` 참고
- 생성 파일: `logs/` 디렉터리 자동 생성 (`.gitignore` 적용)

## v2.0 개발 제약

v2.0 작업 시 반드시 준수해야 할 규칙.

| 규칙 | 내용 |
|---|---|
| MediaPipe 역할 | 전처리(ROI 크롭) 전용. 거북목 판단 로직 일체 포함 금지 |
| 판정 주체 | EfficientNet-B0 파인튜닝 모델만 판단 — 규칙 기반 임계값 혼용 금지 |
| 교체 범위 | `src/detector.py`의 `_calc_score()` 제거 후 모델 추론으로 대체. `camera_loop`, 트레이 구조는 유지 |
| 출력 클래스 | 정상 / 경증 거북목 / 중증 거북목 (3-class) |
| 실시간 성능 | 노트북 CPU에서 30fps 유지 — 무거운 전처리 추가 금지 |
| 데이터셋 경로 | `dataset/normal/`, `dataset/mild/`, `dataset/severe/` |
| 학습 환경 | Google Colab (GPU). 로컬 학습 코드 작성 불필요 |
