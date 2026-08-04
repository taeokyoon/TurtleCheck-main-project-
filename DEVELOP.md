# TurtleNeckDetector — 개발 문서

> 웹캠 하나로 실시간 거북목 자세를 감지하고, 시스템 트레이(Windows / macOS)에서 조용히 동작하며 경고 알림을 보내주는 크로스플랫폼 백그라운드 애플리케이션입니다.
>
> **프로젝트 정보**: 대학교 캡스톤 가상 창업 기업 | 업태: 정보통신업 / 소프트웨어 개발 및 공급업 | 사업분야: 디지털 헬스케어 (AI 헬스테크)

---

## 목차

1. [기능 목록](#기능-목록)
2. [시스템 아키텍처](#시스템-아키텍처)
3. [폴더 구조](#폴더-구조)
4. [모듈 설명](#모듈-설명)
5. [기술 스택 및 의존성](#기술-스택-및-의존성)
6. [설치 및 실행 방법](#설치-및-실행-방법)
7. [사용 방법](#사용-방법)
8. [빌드 및 배포](#빌드-및-배포)
9. [개발 단계 현황](#개발-단계-현황)
10. [v2.0 고도화 로드맵](#v20-고도화-로드맵)
11. [다음 과제](#다음-과제)

---

## 기능 목록

- **실시간 자세 감지** — MediaPipe Pose로 코·어깨 좌표(x·y·z)를 추출하여 거북목 여부 판정. z축 보조 가중치(`_Z_WEIGHT`)로 오탐 감소
- **캘리브레이션** — 사용자의 정상 자세를 기준값으로 설정, 개인 체형·카메라 위치 차이 자동 보정
- **히스테리시스 판정** — 진입/해제 임계값을 다르게 설정해 상태 떨림(flickering) 방지
- **시스템 트레이 아이콘** — 창 없이 백그라운드 동작, 아이콘 색상으로 상태 즉시 확인
  - 회색: 캘리브레이션 대기 중
  - 초록: 자세 정상
  - 빨강: 거북목 감지됨
- **OS 알림** — 거북목 감지 시 10초 쿨다운으로 반복 알림 방지 (Windows: 토스트 알림 / macOS: 데스크탑 알림)
- **JSON Lines 로그** — 60초마다 `logs/local/posture_log.jsonl`에 자동 저장 (로컬 전용, 클라우드 업로드 없음)
- **구조화 로깅** — `logs/app.log`에 회전 파일 로그 자동 기록

> 로그인·클라우드 통계·계정 동기화는 현재 지원하지 않습니다. 핵심 감지 기능에 집중하기 위해 의도적으로 제외했습니다 (배경은 [다음 과제](#다음-과제) 참고).

---

## 시스템 아키텍처

### 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      turtleCheck.py                          │
│                                                                │
│  AppState (공유 상태)                                          │
│  ├── detector       logger                                    │
│  ├── tk_queue       frame_queue   show_visual                 │
│  └── stop_event     tray_icon     last_save                   │
│                                                                │
│  [메인 스레드]              [백그라운드 스레드]                  │
│  tkinter mainloop()        camera_loop(app)                   │
│  _poll() — tk_queue        자세 감지 · 판정 · 로그 저장          │
│  트레이 콜백 실행                                                │
└─────────────────────────────────────────────────────────────┘
```

> **스레드 설계 원칙**
> - tkinter는 반드시 메인 스레드에서 실행 (pystray는 별도 daemon 스레드)
> - 백그라운드 스레드 → 메인 스레드 UI 호출은 `app.tk_queue`를 통해 직렬화

---

### 앱 실행 흐름

```
AppState 초기화
    │  PostureDetector, PostureLogger(logs/local)
    ▼
StartupWindow.run()
    │  카메라 피드 + 캘리브레이션 UI
    │  on_done() 호출 시 트레이 모드 전환
    ▼
build_tray() + _make_callbacks(app)
    │  트레이 아이콘 생성 (콜백이 app을 클로저로 캡처)
    ▼
스레드 시작
    ├── camera_loop(app)   — daemon
    └── tray_icon.run()    — daemon
    ▼
tkinter mainloop()  ← _poll()이 200ms마다 tk_queue 소비
```

---

### 자세 판정 데이터 흐름

```
웹캠 프레임
    │
    ▼
MediaPipe Pose
    │  NOSE, LEFT/RIGHT_SHOULDER, LEFT/RIGHT_EAR 랜드마크 추출
    ▼
_calc_features()
    │  y_score       = (shoulder_y_avg - nose_y) / shoulder_width
    │  z_forward     = (nose_z - shoulder_z_avg) / shoulder_width      (고개 숙임 시 z 기여 억제)
    │  head_tilt     = (left_ear_y - right_ear_y) / shoulder_width
    │  ear_z_offset  = (ear_z_avg - shoulder_z_avg) / shoulder_width
    │  → PostureFeatures(y_score, z_forward, head_tilt, ear_z_offset) 로 4개 지표를 분리 보존
    ▼
EMA 스무딩 (_ema, alpha=0.3) — 프레임간 흔들림 완화
    ▼
슬라이딩 윈도우 (deque, maxlen=200, 유효기간 1초)
    │  최근 1초 평균 피처 벡터 계산 (최소 5샘플)
    ▼
update() — 1초마다 판정
    │  score     = _combine(avg_features)   ※ 현재는 y_score/z_forward만 반영 (v1 규칙 기반 유지)
    │  deviation = score - baseline_score
    │  deviation < -0.10  →  is_turtle = True   (거북목 진입)
    │  deviation > -0.05  →  is_turtle = False  (정상 복귀)
    ▼
┌─────────────────┬──────────────────┬────────────────────┐
트레이 아이콘 갱신  Windows 알림        PostureLogger.tick()
set_tray_state()  (10초 쿨다운)       60초마다 flush → posture_log.jsonl
```

---

### 로그 저장 구조

```
logs/
├── app.log                   ← 구조화 로그 (RotatingFileHandler, 1MB × 3)
└── local/
    └── posture_log.jsonl     ← 분 단위 자세 판정 기록
```

#### `posture_log.jsonl` 레코드 형식

```jsonl
{"timestamp": "2026-04-24T09:01:00", "status": 0, "turtle_seconds": 3, "total_seconds": 60}
{"timestamp": "2026-04-24T09:02:00", "status": 1, "turtle_seconds": 38, "total_seconds": 59}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `timestamp` | ISO 8601 string | 기록 시각 |
| `status` | 0 or 1 | 구간 다수결 결과 (0: 정상, 1: 거북목) |
| `turtle_seconds` | int | 거북목 판정 초 수 |
| `total_seconds` | int | 유효 측정 초 수 |

---

## 폴더 구조

```
TurtleNeckDetector/
│
├── src/
│   ├── __init__.py
│   ├── detector.py          ← MediaPipe 자세 점수 계산 + 히스테리시스 판정
│   ├── log_config.py        ← 앱 전체 로깅 설정 (콘솔 + 회전 파일)
│   ├── logger.py            ← JSON Lines 분 단위 로컬 로그 저장
│   ├── startup_window.py    ← tkinter 시작창(StartupWindow) / 설정창(SettingsWindow)
│   ├── tray_app.py          ← pystray 트레이 아이콘 + 알림
│   └── utils/
│       └── notifier.py      ← OS별 알림 추상화 (Windows/macOS/기타)
│
├── assets/
│   └── mascot.png           ← UI 마스코트 이미지
│
├── logs/                    ← 런타임 자동 생성 (.gitignore)
│   ├── app.log
│   └── local/posture_log.jsonl
│
├── turtleCheck.py           ← 진입점: AppState + 스레드 오케스트레이션
├── config.json               ← 임계값·저장 주기 설정
├── .env                      ← 환경 변수 (필요 시 사용, .gitignore)
├── .env.example               ← 팀원용 환경 변수 템플릿
│
├── AGENTS.md                 ← Claude Code 등 AI 에이전트용 컨텍스트 파일
├── DEVELOP.md                 ← 이 파일
├── README.md
├── requirements.txt
├── TurtleNeckDetector.spec    ← PyInstaller 빌드 스펙
└── .gitignore
```

---

## 모듈 설명

### `turtleCheck.py` — 진입점

| 구성요소 | 설명 |
|---|---|
| `AppState` | 앱 전체 공유 상태 클래스. 전역 변수 대신 단일 인스턴스로 관리 |
| `_make_callbacks(app)` | 트레이 메뉴 콜백 팩토리. `app`을 클로저로 캡처해 딕셔너리 반환 |
| `camera_loop(app)` | 백그라운드 스레드: 프레임 수집 → 점수 계산 → 판정 → 로그 저장 |

**주요 상수**

| 상수 | 값 | 설명 |
|---|---|---|
| `SAVE_INTERVAL` | `config.json` | 로그 flush 주기 (기본 60초) |
| `NOTIFY_COOLDOWN` | `10.0` | 거북목 알림 재발송 최소 간격 (초) |
| `POLL_INTERVAL_MS` | `200` | tkinter 이벤트 큐 폴링 간격 (ms) |

---

### `src/detector.py` — 자세 감지

| 상수 | 값 | 설명 |
|---|---|---|
| `_WINDOW_MAXLEN` | `200` | 슬라이딩 윈도우 최대 샘플 수 |
| `_MIN_VISIBILITY` | `0.5` | 랜드마크 신뢰도 하한 |
| `_MIN_SHOULDER_W` | `0.05` | 어깨 너비 최솟값 (측면 촬영 필터) |
| `_EVAL_INTERVAL` | `1.0` | 판정 주기 (초) |
| `_MIN_SCORES` | `5` | 판정에 필요한 최소 샘플 수 |
| `_Z_WEIGHT` | `0.3` | z축 보조 가중치 (0 = y축 전용) |
| `_Z_GATE_Y` | `0.15` | y 변화량이 이 값 이상이면 z 기여 점진 억제 |
| `_EMA_ALPHA` | `0.3` | 프레임간 지수이동평균 계수 (작을수록 부드럽게) |

`PostureFeatures = namedtuple(..., ["y_score", "z_forward", "head_tilt", "ear_z_offset"])` — 자세 지표를 하나의 점수로 합치지 않고 벡터로 보존. v1 규칙 기반 판정은 `_combine()`에서 `y_score`/`z_forward`만 사용하지만, 4개 지표 전부 슬라이딩 윈도우에 쌓여 향후 이상탐지 모델(OCSVM 등) 학습 데이터로 재사용 가능.

| 메서드 | 반환 | 설명 |
|---|---|---|
| `process_frame(frame)` | `PostureFeatures \| None` | BGR 프레임 → 자세 피처 벡터 |
| `process_frame_visual(frame)` | `(features, rgb)` | 피처 벡터 + 랜드마크 오버레이 이미지 |
| `update(features)` | `(did_evaluate, state_changed)` | EMA 스무딩 → 윈도우 갱신 → 1초마다 히스테리시스 판정 |
| `calibrate()` | `float \| None` | 현재 윈도우 평균 피처를 `baseline_features`/`baseline_score`로 설정 |

---

### `src/log_config.py` — 로깅 설정

`setup_logging(log_dir)` 를 앱 시작 시 **한 번만** 호출합니다.

- **콘솔 핸들러**: `INFO` 이상 출력
- **파일 핸들러**: `DEBUG` 이상 → `logs/app.log` (1MB × 3개 회전)
- 각 모듈에서 `logging.getLogger(__name__)` 으로 로거 획득

---

### `src/logger.py` — 로컬 로그 저장

`PostureLogger(user_dir)` — 초 단위 판정 결과를 누적하다가 `flush()` 호출 시 한 줄의 JSON Lines 레코드로 저장.

| 메서드 | 설명 |
|---|---|
| `tick(is_turtle)` | 1초 판정 결과 누적 |
| `flush()` / `flush_with_record()` | 누적 데이터를 파일에 기록하고 카운터 초기화 |

---

### `src/startup_window.py` — UI 창

| 구성요소 | 설명 |
|---|---|
| `StartupWindow` | 앱 실행 시 메인 UI. 마스코트·카메라 피드·캘리브레이션 |
| `SettingsWindow` | 트레이 "설정 화면 열기" 클릭 시 설정 창. 마스코트·캘리브레이션·카메라 포함 |
| `_load_mascot()` | `assets/mascot.png` 로드 (실패 시 graceful fallback) |

---

### `src/tray_app.py` — 트레이 아이콘

| 함수 | 설명 |
|---|---|
| `build_tray(on_open_gui, on_quit)` | 트레이 아이콘 생성. 메뉴: 설정 화면 열기 \| 종료 |
| `set_tray_state(icon, baseline, is_turtle)` | 상태에 따른 아이콘 색상·툴팁 갱신 |
| `notify(title, msg)` | OS 알림 발송 (`src/utils/notifier.py` 위임) |

---

## 기술 스택 및 의존성

| 라이브러리 | 용도 |
|---|---|
| `opencv-python` | 웹캠 프레임 캡처 |
| `mediapipe` | Pose 랜드마크 추출 |
| `pystray` | 시스템 트레이 아이콘 |
| `Pillow` | 트레이 아이콘 이미지 생성 + 카메라 피드 변환 |
| `customtkinter` | 현대적 UI 테마 |
| `python-dotenv` | `.env` 파일에서 환경 변수 로드 |
| `winotify` | Windows 토스트 알림 |
| `plyer` | macOS 알림 |
| `pyinstaller` | .exe 빌드 |

**Python 버전:** 3.10 이상 (`float | None` 타입 힌트 사용)

---

## 설치 및 실행 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd <클론된 폴더명>
```

### 2. 가상환경 생성 및 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 3. 실행

```bash
python turtleCheck.py
```

> 로그인·API 키 설정 없이 바로 실행 가능합니다.

---

## 사용 방법

### 1단계: 시작 창

앱 실행 시 시작 창이 열립니다. 카메라 미리보기가 바로 표시됩니다.

### 2단계: 캘리브레이션

1. **바른 자세로 웹캠 앞에 앉는다**
2. 시작 창 또는 트레이 메뉴 → **캘리브레이션 시작** 클릭
3. 완료 후 "계속하기" 클릭 → 트레이 아이콘이 초록색으로 전환

> 카메라 위치가 바뀌거나 자리를 옮기면 재캘리브레이션 권장.

### 3단계: 백그라운드 모니터링

- 시작 창이 닫히면 트레이 모드로 전환, 창이 뜨지 않음
- 트레이 아이콘 색상으로 실시간 자세 확인
- 거북목 감지 시 OS 알림 팝업 (10초 쿨다운)

### 트레이 메뉴

| 메뉴 | 설명 |
|---|---|
| 설정 화면 열기 | 카메라 피드 + 재캘리브레이션 창 |
| 종료 | 앱 종료 |

### 로그 직접 조회

```python
import json

with open("logs/local/posture_log.jsonl", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

print(f"총 {len(records)}건 기록")
```

---

## 빌드 및 배포

### PyInstaller 빌드 명령

```bash
python -m pyinstaller TurtleNeckDetector.spec
```

또는 spec 없이 처음부터 생성할 경우:

```bat
python -m pyinstaller ^
  --noconsole ^
  --onedir ^
  --name TurtleNeckDetector ^
  --collect-all mediapipe ^
  --hidden-import pystray._win32 ^
  --hidden-import customtkinter ^
  --add-data "config.json;." ^
  --add-data "assets;assets" ^
  turtleCheck.py
```

### 배포 패키지 구성

```
TurtleNeckDetector/
├── TurtleNeckDetector.exe
├── config.json
└── logs/                    ← 런타임 자동 생성
```

**빌드 결과물 크기:** MediaPipe 포함으로 약 300~500MB

---

## 개발 단계 현황

| 단계 | 상태 | 비고 |
|---|---|---|
| 핵심 감지 엔진 | ✅ 완료 | MediaPipe Pose + 히스테리시스 판정, z축 오탐 감소 |
| 로컬 로깅 | ✅ 완료 | `PostureLogger` — JSON Lines 분 단위 저장 |
| 트레이/알림 UI | ✅ 완료 | pystray 아이콘 + OS 알림 (Windows/macOS) |
| 로그인·클라우드 연동 | ❌ 제거됨 | Firebase → Supabase 이전을 시도했으나, 스토어 배포 목표에 맞춰 핵심 기능 우선으로 판단해 전면 제거 (배경: [다음 과제](#다음-과제)) |
| exe 배포 | 🔶 진행 중 | 빌드 스펙 정리 완료, 신규 환경 실행 검증 필요 |
| 운영 안정화 | ✅ 완료 | logging 모듈 도입 (log_config.py), app.log 자동 기록 |

---

## v2.0 고도화 로드맵

### 핵심 전환: 규칙 기반 → CNN 전이학습

기존 `_calc_score()` 규칙 기반 판정을 EfficientNet-B0 파인튜닝 모델로 전면 교체합니다.
`src/detector.py`만 교체하고, 기존 `camera_loop`, 트레이 구조는 최대한 유지합니다.

### v2.0 기술 아키텍처

```
웹캠 프레임
  ↓
[전처리] MediaPipe Pose
  - 어깨/목/머리 랜드마크로 상체 ROI 영역 크롭
  - 판단 역할 없음, 순수 전처리 도구
  ↓
[AI 판단] EfficientNet-B0 파인튜닝 모델
  - 입력: 크롭된 상체 이미지 (픽셀 데이터)
  - 출력: 정상 / 경증 거북목 / 중증 거북목 (3-class 분류)
  - Google Colab에서 자체 수집 데이터셋으로 학습
  ↓
[지능형 피드백] LLM API (Claude / GPT-4o)
  - 입력: 심각도 + 지속시간 + 사용자 세션 패턴
  - 출력: 맞춤형 스트레칭 추천, 자세 교정 피드백 (자연어)
```

### 데이터셋 전략

- **직접 수집**: 웹캠으로 정상/경증/중증 자세 촬영 및 라벨링
- **저장 구조**: `dataset/normal/`, `dataset/mild/`, `dataset/severe/`
- **수집 도구**: `collect_data.py` (신규 개발)

### v2.0 기술 스택 추가

| 항목 | 내용 |
|---|---|
| 모델 | EfficientNet-B0 (PyTorch + torchvision) |
| 학습 환경 | Google Colab (GPU) |
| 추론 환경 | 노트북 CPU — 30fps 유지 목표 |
| LLM 연동 | Claude API / GPT-4o |

### 5개월 추진 일정

| 월차 | 주요 작업 |
|---|---|
| 1개월차 | 데이터 수집 모듈(`collect_data.py`) 개발 + 데이터셋 구축 + 라벨링 |
| 2개월차 | EfficientNet-B0 파인튜닝 + 정확도 검증 + `detector.py` 규칙 기반 교체 |
| 3개월차 | LLM API 연동 + 스트레칭 추천 엔진 + 코칭 UI 구현 |
| 4개월차 | 주간 리포트 + 사용자 테스트 + 성능 최적화 |
| 5개월차 | 최종 패키징 + 발표 자료 + 베타 배포 |

---

## 다음 과제

### 로그인/계정 기능 — 의도적으로 보류

Supabase Auth + Google OAuth까지 구현했으나, 최종 목표가 **Steam / MS Store / Google Play 배포**로 정해지면서 제거했습니다.

- 로그인 없는 로컬 전용 유틸리티로도 스토어 출시가 충분히 가능
- 계정 기능은 복잡도(OAuth 콜백, 세션 갱신, 클라우드 동기화) 대비 지금 단계의 핵심 가치(감지 정확도) 증명에 기여하지 않음
- Google Play는 계정 기능을 넣는 순간 **계정 삭제 기능**과 **개인정보처리방침**을 요구하므로, 필요성이 명확해지기 전까지는 미루는 것이 유리
- 추후 필요해지면(기기 간 동기화, 구독 등) 커스텀 URI 스킴 등 더 간결한 방식으로 재도입 검토

### v1.0 마무리

- 실제 신규 PC에서 exe 실행 검증 (캘리브레이션·알림 전체)

### v1.0 추후 검토

- 자동 업데이트 (GitHub Releases 연동)
- 주간/월간 통계 대시보드 (로컬 데이터 기반)
- Windows 시작 프로그램 자동 등록

---

## 라이선스

```
MIT License — Copyright (c) 2026 TurtleNeckDetector Contributors
```
