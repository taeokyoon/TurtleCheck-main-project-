# AGENTS.md — TurtleNeckDetector

## 프로젝트 개요

**규칙 기반 실시간 거북목 감지 + 데이터 기반 검증/AI 보조 지표 시스템** (디지털 헬스케어 AI 헬스테크)

웹캠 기반 실시간 거북목 감지 앱. 시스템 트레이(Windows / macOS)에서 백그라운드로 동작하며, MediaPipe Pose로 자세를 판정하고 OS 알림을 발송한다. 로그인 없이 로컬 전용으로 동작하며(Steam/MS Store/Google Play 배포를 염두에 두고 계정 기능은 의도적으로 제외), 로그는 `logs/local/`에 JSON Lines로 저장된다.

- **현재(v1.0)**: MediaPipe Pose 랜드마크 + 규칙 기반 임계값으로 거북목 판정
- **검증 + AI 보조 지표**: `tools/`에서 직접 라벨링한 데이터셋으로 로지스틱 회귀를 학습시켜,
  규칙 기반 판정이 실제 라벨과 얼마나 일치하는지 정량 검증했다(`dataset/validation_report.md`).
  학습된 가중치(`dataset/model_weights.json`)는 `src/ai_advisor.py`를 통해 앱에서도 실시간으로
  로드되어, 트레이 툴팁에 "AI 판단이 규칙 기반과 일치하는지"를 보조 정보로 보여준다. **최종
  판정 권한은 여전히 규칙 기반(`src/detector.py`)에 있다** — AI는 참고용 보조 신호일 뿐 판정을
  대체하지 않는다. 배경은
  [docs/superpowers/specs/2026-08-04-rule-validation-model-design.md](docs/superpowers/specs/2026-08-04-rule-validation-model-design.md) 참고.

자세한 내용은 아래 문서를 반드시 참고한다.

- 답변은 항상 토큰을 최적화해서 효율적으로 답변을 해준다.
- **[README.md](README.md)** — 빠른 시작 가이드
- **[DEVELOP.md](DEVELOP.md)** — 전체 아키텍처, 모듈 설명, 데이터 흐름, 개발 현황, 규칙 기반 판정 검증
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
| `src/ai_advisor.py` | 학습된 가중치로 실시간 AI 보조 판단 계산 (판정 로직에는 관여 안 함) |

## 스레드 구조

- **메인 스레드**: tkinter mainloop + `_poll()` (200ms 간격 `tk_queue` 소비)
- **Daemon 스레드 1**: `camera_loop()` — 웹캠 캡처 → 자세 판정 → 로그 저장
- **Daemon 스레드 2**: pystray 트레이 루프

배경 스레드 → UI 호출은 반드시 `app.tk_queue`를 거쳐야 한다.

## 설정 파일

- `config.json` — 거북목 임계값(`delta_turtle`, `delta_ok`), 로그 flush 주기(`save_interval_seconds`)

## 로그인/계정 기능

의도적으로 제거된 상태다. Firebase → Supabase Auth 이전까지 시도했으나, 최종 배포 목표(Steam/MS Store/Google Play)를 고려해 핵심 감지 기능 우선으로 전면 제거했다. 관련 코드를 다시 추가하기 전에 `DEVELOP.md`의 [다음 과제](DEVELOP.md#다음-과제) 섹션을 먼저 확인할 것.

## 런타임 환경

- Python 3.10+ 필수 (`float | None` 타입 힌트 사용)
- 의존성: `requirements.txt` 참고
- 생성 파일: `logs/` 디렉터리 자동 생성 (`.gitignore` 적용)

## 검증 도구 (`tools/`)

| 파일 | 역할 |
|---|---|
| `tools/collect_labels.py` | 라벨링 데이터 수집 (개발 전용, 배포 앱과 무관) |
| `tools/train_model.py` | 세션 단위 train/test 학습·검증 리포트 생성 + 전체 데이터로 재학습해 `dataset/model.pkl`/`model_weights.json` 내보내기 |
| `tools/test_train_model.py` | `train_model.py`의 순수 로직 함수 pytest (`cd tools && pytest`) |
| `tools/requirements.txt` | 학습 도구 전용 의존성(pandas, scikit-learn, pytest) — 배포용 `requirements.txt`와 분리 |

판정 로직(`src/detector.py`)은 그대로 유지되며, `PostureDetector.last_avg_features` 속성만
AI 보조 지표가 읽어갈 수 있도록 추가됐다(판정 로직 자체는 변경 없음). scikit-learn은 학습
도구(`tools/`)에서만 쓰이고, 배포 앱(`src/ai_advisor.py`)은 가중치 숫자만 읽어 순수 파이썬으로
시그모이드를 계산한다(무거운 의존성 없음).

**AI 판단 기준 확률은 표준값 50%를 그대로 쓴다.** 한때 "규칙 기반과 더 잘 맞도록" 34.4%로
낮춰봤지만, 그러면 AI의 판단이 규칙 기반을 뒤따라가도록 억지로 맞춘 것이 되어 "AI가 독립적으로
검증해준다"는 취지에서 벗어나므로 되돌렸다 — 자세한 경위는 아래 DEVELOP.md 참고.

상세 설계는
[docs/superpowers/specs/2026-08-04-rule-validation-model-design.md](docs/superpowers/specs/2026-08-04-rule-validation-model-design.md) 참고.
