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
10. [규칙 기반 판정 검증](#규칙-기반-판정-검증)
11. [로드맵 및 진행 현황](#로드맵-및-진행-현황)

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

> 로그인·클라우드 통계·계정 동기화는 현재 지원하지 않습니다. 핵심 감지 기능에 집중하기 위해 의도적으로 제외했습니다 (배경은 [로드맵 및 진행 현황](#로드맵-및-진행-현황) 참고).

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
    │  deviation < -0.13  →  is_turtle = True   (거북목 진입)   ※ config.json delta_turtle
    │  deviation > -0.07  →  is_turtle = False  (정상 복귀)   ※ config.json delta_ok
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

`PostureFeatures = namedtuple(..., ["y_score", "z_forward", "head_tilt", "ear_z_offset"])` — 자세 지표를 하나의 점수로 합치지 않고 벡터로 보존. 판정은 `_combine()`에서 `y_score`/`z_forward`만 사용하며, `head_tilt`/`ear_z_offset`은 계속 계산·로깅만 해서 추후 임계값 튜닝이나 디버깅 참고용으로 남겨둔다.

히스테리시스는 `baseline_score`와의 거리(`deviation`)로 판정한다: `deviation < -delta_turtle`이면 거북목 진입, `deviation > -delta_ok`이면 정상 복귀. `delta_turtle(0.13) > delta_ok(0.07)` (`config.json`)로 진입 임계값이 복귀보다 더 크게(멀게) 설정돼 있어 두 조건의 경계가 겹치지 않고, 상태가 매 틱 뒤집히는 플래핑이 구조적으로 발생하지 않는다.

| 메서드 | 반환 | 설명 |
|---|---|---|
| `process_frame(frame)` | `PostureFeatures \| None` | BGR 프레임 → 자세 피처 벡터 |
| `process_frame_with_landmarks(frame)` | `(features, landmarks)` | 피처 + MediaPipe 원시 랜드마크 33개. 수집 도구 전용, 판정엔 미사용 |
| `process_frame_visual(frame)` | `(features, rgb)` | 피처 벡터 + 랜드마크 오버레이 이미지 |
| `update(features)` | `(did_evaluate, state_changed)` | EMA 스무딩 → 윈도우 갱신 → 1초마다 히스테리시스 판정 |
| `calibrate()` | `float \| None` | 윈도우 평균을 baseline으로 즉시 설정 (카운트다운 없이 클릭 즉시 완료) |

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
| 로그인·클라우드 연동 | ❌ 제거됨 | Firebase → Supabase 이전을 시도했으나, 스토어 배포 목표에 맞춰 핵심 기능 우선으로 판단해 전면 제거 (배경: [로드맵 및 진행 현황](#로드맵-및-진행-현황)) |
| exe 배포 | 🔶 진행 중 | 빌드 스펙 정리 완료, 신규 환경 실행 검증 필요 |
| 운영 안정화 | ✅ 완료 | logging 모듈 도입 (log_config.py), app.log 자동 기록 |
| 규칙 기반 판정 데이터 검증 | ✅ 완료 | 라벨링 데이터셋(662행) 수집 + 로지스틱 회귀 학습·검증 (`tools/`) |
| 앱 내 AI 보조 지표 | ✅ 완료 | `src/ai_advisor.py` — 트레이 툴팁·알림에 AI 참고 의견 표시 |
| 랜드마크 기반 PyTorch 모델 | ✅ 완료 | `tools/train_torch.py` — MLP·GRU vs 규칙 기반 3자 비교, ONNX 내보내기 (앱 통합은 Phase 3) |

---

## 규칙 기반 판정 검증

### 배경

과거 이 섹션은 "규칙 기반 → EfficientNet-B0 CNN 전이학습 교체"를 v2.0 목표로 잡았었다.
검토 결과 팀 인력상 본인 1인 데이터만 가능해 이미지 데이터셋의 다양성을 확보할 수 없고,
과적합 위험과 실시간 추론 최적화 부담이 커서 기각했다. 대신 **판정 로직(`src/detector.py`)은
그대로 유지**하고, 별도로 라벨링한 데이터셋으로 로지스틱 회귀를 학습시켜 "규칙 기반 판정이
실제 라벨과 얼마나 일치하는가"를 정량 검증하는 방향으로 전환했다. 자세한 검토 과정은
[docs/superpowers/specs/2026-08-04-rule-validation-model-design.md](docs/superpowers/specs/2026-08-04-rule-validation-model-design.md) 참고.

### 도구 구성

| 파일 | 역할 |
|---|---|
| `tools/collect_labels.py` | 웹캠 + 키보드로 라벨링 데이터 수집 v2 → `dataset/posture_v2.csv` (개발 전용, 아래 "데이터셋 v2" 참고) |
| `tools/train_model.py` | 세션 단위 train/test 분리 → 로지스틱 회귀 학습 → 규칙 기반 판정과 비교 리포트 생성 |

`turtleCheck.py`, `src/detector.py` 등 배포 코드는 이 작업으로 전혀 수정되지 않았다. 학습된
모델(`dataset/model.pkl`)은 배포되지 않으며, 검증 근거로만 쓰인다.

### 검증 결과

`dataset/posture_labels.csv`(직접 수집, 5세션 662행)를 세션 단위로 train(406행)/test(256행)
분리해 학습·검증한 결과(`dataset/validation_report.md`):

| | Accuracy | Precision | Recall |
|---|---|---|---|
| 학습된 로지스틱 회귀 모델 | 87.9% | 71.3% | 100% |
| 규칙 기반 판정 (현재 배포 로직) | 98.8% | 97.4% | 98.7% |

규칙 기반 판정이 학습된 모델보다 더 높은 정확도를 보였다 — 기존에 튜닝해둔 판정 로직이
데이터로 검증해도 신뢰할 만하다는 근거로 활용한다. (재현율/정밀도 트레이드오프 관점에서,
학습 모델은 재현율 100%인 대신 정밀도가 낮아 "예민하게 오탐하는" 패턴을 보였고, 규칙
기반은 두 지표가 고르게 높아 어느 한쪽으로 치우치지 않았다.)

**주의**: 위 검증 지표는 데이터의 80%(3세션)만 학습에 쓴 `eval_model` 기준이다. 실제
`dataset/model.pkl`/`model_weights.json`(앱에 배포되는 가중치)은 검증이 끝난 뒤 **5세션
662행 전체**로 다시 학습한 별도의 최종 모델이다 — 검증은 held-out으로 공정하게 하되, 배포
모델은 가진 데이터를 아끼지 않기 위함이다.

### 규칙과 AI가 실제로 갈렸던 사례

검증(test) 256행 중 규칙 기반이 실제로 틀렸던 건 3건(오탐 2건, 누락 1건)뿐이었는데, 이 3건
전부에서 학습된 모델은 정답 쪽으로 판단했다:

| 실제 라벨 | 규칙 판정 | AI 확률 | AI 판정 |
|---|---|---|---|
| 정상 | 거북목(오탐) | 1.7% | 정상 (정답) |
| 정상 | 거북목(오탐) | 1.0% | 정상 (정답) |
| 거북목 | 정상(누락) | 58.7% | 거북목 (정답) |

규칙이 애매하게 걸친 경계 사례에서 AI가 보정해준다는 정황 증거다. 다만 표본이 3건뿐이라
"AI가 항상 규칙의 오류를 잡아준다"는 통계적 증명은 아니고, 규칙 자체가 워낙 정확해서(98.8%)
비교할 오류 사례 자체가 적다는 한계가 있다.

### 앱 내 AI 보조 지표

학습된 가중치(`dataset/model_weights.json`)는 `src/ai_advisor.py`를 통해 앱 실행 중에도 실시간
로드된다. `camera_loop`에서 판정 상태가 바뀔 때마다 AI 모델의 실시간 판단을 계산해:

- 트레이 툴팁에 `거북목 감지됨! (AI: 거북목 · 일치)`처럼 표시
- 거북목 알림 팝업 본문에도 `자세를 바로잡아 주세요. (AI 모델도 동의: 일치)`처럼 덧붙임

**최종 판정 권한은 여전히 규칙 기반에 있고, AI는 참고용 보조 신호다.** scikit-learn 전체를
배포하지 않기 위해 학습된 가중치 4개+절편만 JSON으로 내보내고, 앱에서는 순수 파이썬으로
시그모이드를 직접 계산한다(`src/ai_advisor.py`).

**AI 판단 기준 확률(50%)을 낮추지 않기로 한 이유**: 처음엔 실제 사용 중 "불일치"만 계속
떠서, "규칙 기반 판정과 가장 잘 일치하는 확률(34.4%)"을 데이터에서 찾아 적용해봤다. 그런데
이는 "AI가 규칙과 다르게 판단할 자유"를 없애고 규칙을 그대로 따라가게 만드는 것과 같아서,
"AI로 독립적으로 검증한다"는 원래 취지와 순환 논리에 빠지는 문제가 있었다. 그래서 표준
기본값 50%로 되돌렸고, 그 결과 실제 사용 중 "불일치"가 다시 자주 뜬다 — 이는 AI가 규칙보다
더 큰 편차가 있어야 확신하는, 정직하고 독립적인 참고 의견이다.

### 데이터셋 v2 — 학습용 원시 랜드마크 수집

v1 데이터(`posture_labels.csv`, 662행)는 1명·1일·5세션이고 세션마다 라벨이 하나뿐이라
일반화 성능을 검증할 수 없었다. v2 수집 도구는 이를 보완한다 (v1 파일과 리포트는 그대로 보존):

```bash
python tools/collect_labels.py --person <이름> [--camera 0]
# n=정상  t=거북목  p=일시정지(기록 안 함)  q=종료
```

| 항목 | v1 | v2 |
|---|---|---|
| 파일 | `dataset/posture_labels.csv` | `dataset/posture_v2.csv` |
| 세션 | 토글마다 새 세션 (세션 = 단일 라벨) | 1실행 = 1세션, 안에서 라벨 전환 (정상↔거북목 전환 구간 포함) |
| 기록 주기 | 1Hz | 5Hz (`_RECORD_INTERVAL = 0.2`) |
| 컬럼 | 피처 4 + baseline 4 + label | + `person_id`, `timestamp`(유닉스 초), 상체 랜드마크 13개 × (x,y,z,v) = 64컬럼 |

랜드마크는 MediaPipe Pose 0~12번(코·눈·귀·입·어깨)만 저장한다 — 책상 앞에서 하체는 거의
보이지 않는다. 기존 피처 4개도 같이 남겨서, 같은 v2 데이터로 규칙 기반 판정과 학습 모델을
공정하게 비교할 수 있다. 피처는 EMA 스무딩 전 원시값이다(스무딩은 `update()` 안에서 일어남).

**수집 가이드**: 사람마다 `--person`을 다르게 주고 3명 이상 모은다. 한 세션 안에서
정상→거북목→정상을 여러 번 오가되, 자세를 바꾸는 순간엔 `p`로 잠깐 멈춰 라벨 노이즈를 막는다.

### 데이터 품질 — 라벨 노이즈

662행 중 6행(약 0.9%)은 "거북목(1)"으로 기록됐지만 baseline 대비 편차가 거의 0(정상처럼
보임)이었다. 확인해보니 세션 시작이 아니라 중간(예: 3분 녹화 중 48~70번째 행)에서 발생했다
— 거북목 자세를 유지하며 녹화하던 도중 잠깐 자세를 고쳐 앉은 순간이 그대로 "거북목" 라벨로
같이 기록된 것으로 보인다. 662행 중 0.9% 수준은 로지스틱 회귀가 감당할 수 있는 범위라
학습에 큰 영향은 없었지만, 향후 데이터를 더 모을 때는 녹화 중 자세를 크게 바꾸면 토글을
잠깐 꺼두는 것이 라벨 정확도에 도움이 된다.

---

### 랜드마크 기반 PyTorch 모델 (Phase 2)

8월 검증에서 로지스틱 회귀가 규칙 기반보다 못했던 건 "AI가 못해서"가 아니라, 사람이 설계한
피처 4개를 선형 결합만 할 수 있어 규칙의 비선형 게이트(`z_gate`)를 재현하지 못했기 때문이었다.
Phase 2는 피처 설계를 사람이 하지 않고 **원시 랜드마크(상체 13개)에서 모델이 스스로 특징을
배우게** 한다. 설계: [docs/superpowers/specs/2026-09-22-torch-model-design.md](docs/superpowers/specs/2026-09-22-torch-model-design.md)

```bash
pip install -r tools/requirements.txt      # torch(CPU)·onnx·onnxruntime·matplotlib
python tools/train_torch.py                # 약 20초, CPU 로 충분 (1천 행대 × 파라미터 수천 개)
```

| 항목 | 내용 |
|---|---|
| 입력 | 랜드마크 13개를 어깨 중심 원점·어깨 너비 1 로 정규화 → (x,y,z) 39 + visibility 13 = 52차원. **캘리브레이션 불필요** |
| MLP | 단일 프레임 52 → 32 → 16 → 1 (≈2.2K 파라미터) |
| GRU | 최근 10스텝(2초) 시퀀스 → GRU(32) → 16 → 1 (≈9K 파라미터) |
| 학습 | BCE, Adam 1e-3, early stopping(patience 10), seed 42 |
| 검증 | 세션 단위 Leave-One-Out. 표준화는 train 폴드 기준 |
| 출력 | `dataset/torch_report.md`, `torch_report/*.png`, `mlp_model.onnx`(9.7KB), `gru_model.onnx`(37KB), `feature_norm.json` |
| 보고서용 그래프 | `python tools/plot_report_figures.py` → `dataset/report_figures/fig_comparison.png`, `fig_confusion.png` (한글·% 라벨, 재학습 없이 리포트 숫자만 읽음). 손실 그래프(`*_loss.png`)는 학습 진단용이라 보고서엔 쓰지 않는다 — fold1 MLP val 과적합, GRU fold0 val 누수가 그대로 보임 |

**결과** (`dataset/torch_report.md`, 세션 2개 LOSO 평균, 1,156행):

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **MLP (프레임)** | **0.953** | 0.928 | 0.976 | **0.949** |
| GRU (윈도우) | 0.849 | 0.795 | 1.000 | 0.871 |
| 규칙 기반 | 0.916 | 0.988 | 0.816 | 0.894 |

**해석**:
- **MLP가 규칙 기반을 앞섰다** (95.3% vs 91.6%). 규칙 기반은 거북목 505건 중 93건을 놓쳤고(Recall
  0.816), MLP는 12건만 놓쳤다. 원시 랜드마크 + 비선형 모델이면 사람이 설계한 피처·임계값보다 나을
  수 있다는 Phase 2 가설이 확인됐다. 알림 앱에선 "놓침"이 "오탐"보다 치명적이라 MLP 성향이 제품에 맞다.
- **규칙 기반이 8월(98.8%)보다 낮은 이유**: v2 데이터엔 정상↔거북목 **전환 구간**이 들어 있다.
  규칙은 편차가 -0.13을 넘어야 거북목으로 보므로 "살짝 숙인" 초반 구간을 전부 놓친다. v1 데이터엔
  전환 구간이 없어서 점수가 높았다.
- **GRU는 폴드에 따라 69.8% / 100%로 불안정**. 원인은 데이터 수다 — 학습 세션이 1개뿐이면
  early stopping 용 val 이 같은 세션의 뒤 20%로 잡혀 사실상 답을 보고 검증하는 셈이 되고(폴드0:
  100 epoch 내내 val_loss 감소, train_loss 0.000), 과적합을 못 막는다. 세션이 3개 이상이면 val 이
  별도 세션이 되어 해소된다. **세션 5개 이상 모은 뒤 재실행이 필요하다.**
- **한계**: 1명·2세션. 각 폴드가 세션 하나로만 학습하므로 위 숫자는 "파이프라인이 작동하고 방향이
  맞다"는 증거이지 최종 성능이 아니다. 다인 데이터가 들어오면 `leave_one_group_out(person_id)`로
  사람 단위 검증으로 바꾸면 된다.

**앱 통합 시 주의**: 앱은 `feature_norm.json`의 mean/std 로 **똑같이** 표준화한 뒤 ONNX 에 넣어야
한다. 이걸 빠뜨리면 학습 땐 잘 되고 앱에선 엉망이 된다. `onnxruntime`만 추가하면 되고 torch 는
배포하지 않는다 (Phase 3).

**문제 기록 — protobuf 충돌 (2026-09-22)**:
- 증상: `tools/requirements.txt`에 `onnx>=1.15.0`을 추가해 설치한 뒤 `test_detector.py`가
  `AttributeError: 'FieldDescriptor' object has no attribute 'label'`로 실패하고, 배포 앱도 같은
  이유로 실행 불가.
- 원인: onnx 최신(1.23)이 `protobuf 7.36`을 끌어와, mediapipe 0.10.9가 요구하는 `protobuf<4`가
  깨짐. 배포 `requirements.txt`엔 핀이 있었지만 tools 쪽엔 없었다.
- 해결: `tools/requirements.txt`에 `onnx>=1.15.0,<1.17` + `protobuf<4` 명시 → protobuf 3.20.3 /
  onnx 1.16.2 로 복구, 테스트 24개 전부 통과. onnxruntime 1.30은 메타데이터상 protobuf 4를 요구한다고
  경고하지만 C++ 내부에 자체 protobuf 를 포함해 파이썬 protobuf 버전과 무관하게 추론이 동작한다
  (ONNX 동일성 검증 1e-9 로 확인).
- 교훈: 상한 없는 `>=` 핀은 시간이 지나면 다른 패키지를 깨뜨릴 수 있다. mediapipe 0.10.9 를 쓰는 한
  protobuf 를 건드리는 패키지는 상한을 같이 잡는다.

---

## 로드맵 및 진행 현황

> 마지막 갱신: 2026-09-22. 비중은 예상 작업량 기준.

| Phase | 내용 | 비중 | 상태 |
|---|---|---|---|
| 0 | 문서 정합성 (임계값 0.13/0.07) + 검증 강화 (LOSO) | 5% | ✅ 완료 (Phase 1·2에 흡수) |
| 1 | 라벨링 데이터 수집 도구 v2 (`tools/collect_labels.py`) | 20% | ✅ 완료 — 단, 확보 데이터는 1명·2세션(1,156행)으로 목표(3명·5세션+)의 20~30% |
| 2 | 랜드마크 기반 PyTorch 모델 + ONNX 내보내기 (`tools/train_torch.py`) | 30% | ✅ 완료 — MLP 95.3% > 규칙 91.6% |
| 3 | 앱 통합 — onnxruntime 추론, AI 판정 모드, 알림 피드백 라벨 수집 | 25% | ⬜ 미착수 |
| 4 | MediaPipe Solutions → Tasks API 이전 | 20% | ⬜ 미착수 |
| | **합계** | 100% | **55%** |

```
[■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□] 55%
```

"AI가 실제로 거북목을 판정하는 앱" 관점에서는 약 40% — Phase 3이 끝나야 사용자가 체감하는
변화가 생기고, 데이터가 1명·2세션이라 모델 신뢰도도 아직 낮다.

### 지금 바로 할 수 있는 것 — 데이터 세션 추가 수집

도구는 완성돼 있어 실행만 하면 된다. 세션 3개 이상 더 모으면 GRU 불안정(early stopping 용 val 이
같은 세션에서 나오는 문제)이 해소되고, 5개 이상이면 지표를 믿을 만해진다.

```bash
python tools/collect_labels.py --person taeok      # 카메라 위치·조명을 바꿔가며 2~3분씩
python tools/train_torch.py                        # 재실행하면 리포트·ONNX 자동 갱신
```

다른 사람 데이터는 `--person <이름>` 으로 구분한다. 3명 이상 모이면 `cross_validate()` 의 그룹
키를 `person_id` 로 바꿔 **사람 단위** 검증으로 전환한다 (`leave_one_group_out` 은 그룹 열만 바꾸면 됨).

### Phase 3 — 앱 통합 (다음 작업)

목표: 학습된 MLP 를 앱에서 실시간 추론하고, 사용자가 "AI 판정 모드"를 켜면 AI 가 판정 권한을 갖는다.

| 항목 | 설계 방향 |
|---|---|
| 추론 모듈 | `src/onnx_advisor.py` (신규) — `onnxruntime` 으로 `dataset/mlp_model.onnx` 로드, `feature_norm.json` 의 mean/std 로 **학습과 동일하게** 표준화 후 추론. torch 는 배포하지 않는다 |
| 피처 변환 | `tools/train_torch.py` 의 `normalize_landmarks()` 와 같은 계산을 앱에서 재현 (랜드마크 13개 → 52차원). 공통 함수로 뽑아 두 곳이 같은 코드를 쓰게 한다 |
| 판정 모드 | `config.json` 에 `"judge_mode": "rule" \| "ai"` 추가. 기본은 `rule`(검증된 v1 동작 유지). `ai` 모드에서도 랜드마크 인식 실패·모델 로드 실패 시 규칙 기반으로 fallback |
| 히스테리시스 | AI 확률에도 진입/복귀 기준을 다르게(예: 0.6 진입 / 0.4 복귀) 둬 플래핑 방지 — 규칙 기반과 같은 원리 |
| **알림 피드백 라벨 수집** | 거북목 알림에 "맞아요 / 아니에요" 버튼. 응답이 있을 때만 그 시점 랜드마크 + 사용자 라벨을 `dataset/feedback_labels.csv` 에 저장. **판정 결과 자체를 라벨로 쓰지 않는다** — 그건 판정기의 오류를 그대로 학습하는 순환 논리다. 사람이 준 라벨만 쌓는다 |
| 트레이 표시 | 툴팁에 `판정: AI 87% (규칙: 정상)` 처럼 두 판정을 나란히 |
| 의존성 | 배포 `requirements.txt` 에 `onnxruntime` 추가 (~10MB). PyInstaller spec 에 `.onnx`·`feature_norm.json` 포함 |

### Phase 4 — MediaPipe Tasks API 이전

`mp.solutions.pose` 는 Google 이 legacy 로 둔 API 다 (mediapipe 0.10.9 고정, `protobuf<4` 핀도 이 때문).
`mediapipe.tasks.python.vision.PoseLandmarker` 로 옮기면 최신 모델·GPU delegate·유지보수를 받을
수 있다. `.task` 모델 파일 로드, `LIVE_STREAM` 모드 콜백, 랜드마크 접근 방식이 달라 `src/detector.py`
와 PyInstaller spec 을 함께 손봐야 한다. AI 작업과 독립적이라 별도 단계로 잡는다.

### 보류·추후 검토

- **로그인/계정 기능** — 의도적으로 보류. Supabase Auth + Google OAuth 까지 구현했으나 Steam/MS
  Store/Google Play 배포 목표에 맞춰 제거. 로컬 전용으로도 출시 가능하고, Google Play 는 계정 기능을
  넣는 순간 계정 삭제 기능·개인정보처리방침을 요구한다. 필요해지면 커스텀 URI 스킴 등으로 재검토.
- **데이터 증강 실험** — Phase 2 기준선이 나왔으므로 "보간 증강 있음 vs 없음" 비교 가능. 합성 행은
  `session_id` 에 `aug_` 접두어로 구분하고 학습에만 쓴다(검증은 실제 행만).
- v1.0 마무리: 신규 PC 에서 exe 실행 검증 / 자동 업데이트(GitHub Releases) / 주간·월간 통계 대시보드 /
  Windows 시작 프로그램 등록

---

## 라이선스

```
MIT License — Copyright (c) 2026 TurtleNeckDetector Contributors
```
