# TurtleNeckDetector — 개발 문서

> 웹캠 하나로 실시간 거북목 자세를 감지하고, 시스템 트레이(Windows / macOS)에서 조용히 동작하며 경고 알림을 보내주는 크로스플랫폼 백그라운드 애플리케이션입니다.

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
10. [다음 과제](#다음-과제)

---

## 기능 목록

- **실시간 자세 감지** — MediaPipe Pose로 코·어깨 좌표(x·y·z)를 추출하여 거북목 여부 판정. z축 보조 가중치(`_Z_WEIGHT`)로 오탐 감소
- **캘리브레이션** — 사용자의 정상 자세를 기준값으로 설정, 개인 체형·카메라 위치 차이 자동 보정
- **히스테리시스 판정** — 진입/해제 임계값을 다르게 설정해 상태 떨림(flickering) 방지
- **로그인 / 비로그인 모드** — 로그인 없이도 즉시 탐지 가능, 로그인 시 Firebase 통계 연동
- **시스템 트레이 아이콘** — 창 없이 백그라운드 동작, 아이콘 색상으로 상태 즉시 확인
  - 회색: 캘리브레이션 대기 중
  - 초록: 자세 정상
  - 빨강: 거북목 감지됨
- **OS 알림** — 거북목 감지 시 10초 쿨다운으로 반복 알림 방지 (Windows: 토스트 알림 / macOS: 데스크탑 알림)
- **JSON Lines 로그** — 60초마다 `posture_log.jsonl`에 자동 저장
- **Firestore 업로드** — 로그인 사용자 전용, 오프라인 큐로 네트워크 단절 복구 지원
- **Firebase 누적 통계** — Firestore 직접 쿼리로 오늘·7일·30일 누적 통계 조회 (모바일 앱 데이터 포함)
- **구조화 로깅** — `logs/app.log`에 회전 파일 로그 자동 기록

---

## 시스템 아키텍처

### 전체 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      turtle_neck.py                         │
│                                                             │
│  AppState (공유 상태)                                        │
│  ├── auth_manager   detector    uploader                    │
│  ├── logger         upload_queue                            │
│  ├── tk_queue       frame_queue   show_visual               │
│  └── stop_event     tray_icon     last_save                 │
│                                                             │
│  [메인 스레드]          [백그라운드 스레드 1]  [스레드 2]      │
│  tkinter mainloop()    camera_loop(app)   upload_loop(app)  │
│  _poll() — tk_queue    자세 감지·판정·로그  Firestore 업로드  │
│  트레이 콜백 실행        frame_queue 공급                     │
└─────────────────────────────────────────────────────────────┘
```

> **스레드 설계 원칙**
> - tkinter는 반드시 메인 스레드에서 실행 (pystray는 별도 daemon 스레드)
> - 백그라운드 스레드 → 메인 스레드 UI 호출은 `app.tk_queue`를 통해 직렬화
> - `logger`·`upload_queue` 교체 시 `app.logger_lock`으로 보호

---

### 앱 실행 흐름

```
AppState 초기화
    │  AuthManager, PostureDetector, FirebaseUploader
    │  load_session() → switch_logger(uid)
    ▼
StartupWindow.run()
    │  카메라 피드 + 로그인/캘리브레이션 UI
    │  on_done() 호출 시 트레이 모드 전환
    ▼
build_tray() + _make_callbacks(app)
    │  트레이 아이콘 생성 (콜백이 app을 클로저로 캡처)
    ▼
스레드 시작
    ├── camera_loop(app)   — daemon
    ├── upload_loop(app)   — daemon
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
    │  NOSE, LEFT/RIGHT_SHOULDER 랜드마크 추출
    ▼
_calc_score()
    │  y_score    = (shoulder_y - nose_y)
    │  z_forward  = nose_z - shoulder_avg_z   (고개 숙임 시 z 기여 억제)
    │  score      = (y_score + Z_WEIGHT * z_forward * z_gate) / shoulder_width
    ▼
슬라이딩 윈도우 (deque, maxlen=200, 유효기간 1초)
    │  최근 1초 평균 score 계산 (최소 5샘플)
    ▼
update() — 1초마다 판정
    │  deviation = avg - baseline_score
    │  deviation < -0.10  →  is_turtle = True   (거북목 진입)
    │  deviation > -0.05  →  is_turtle = False  (정상 복귀)
    ▼
┌─────────────────┬──────────────────┬────────────────────┐
트레이 아이콘 갱신  Windows 알림        PostureLogger.tick()
set_tray_state()  (10초 쿨다운)       60초마다 flush → enqueue
```

---

### 로그 저장 구조

```
logs/
├── app.log                   ← 구조화 로그 (RotatingFileHandler, 1MB × 3)
├── session.json              ← 로그인 세션 (uid, email, logged_in_at)
├── anonymous/                ← 비로그인 데이터
│   └── posture_log.jsonl
└── {uid}/                    ← 로그인 사용자별
    ├── posture_log.jsonl
    └── upload_queue.jsonl    ← 업로드 대기/완료/실패 레코드
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

#### Firestore 경로

```
hour/
└── {uid}/
    └── {YYYY-MM-DD}/
        └── {H}~{H+1}/          ← 시간 단위 문서
            ├── total_tracked_seconds
            ├── total_turtle_seconds
            ├── bad_posture_count
            ├── log_data[]
            └── uploaded_at
```

---

## 폴더 구조

```
TurtleNeckDetector/
│
├── src/
│   ├── __init__.py
│   ├── auth.py              ← Firebase Auth REST API + 세션 관리
│   ├── detector.py          ← MediaPipe 자세 점수 계산 + 히스테리시스 판정
│   ├── log_config.py        ← 앱 전체 로깅 설정 (콘솔 + 회전 파일)
│   ├── logger.py            ← JSON Lines 분 단위 로컬 로그 저장
│   ├── startup_window.py    ← tkinter 시작창(StartupWindow) / 설정창(SettingsWindow) / 인증창(AuthWindow)
│   ├── tray_app.py          ← pystray 트레이 아이콘 + 알림
│   └── utils/
│       ├── firebase_uploader.py  ← Firestore 업로드
│       ├── notifier.py           ← OS별 알림 추상화 (Windows/macOS/기타)
│       └── upload_queue.py       ← JSONL 기반 오프라인 업로드 큐
│
├── assets/
│   └── mascot.png           ← UI 마스코트 이미지
│
├── logs/                    ← 런타임 자동 생성 (.gitignore)
│
├── turtle_neck.py           ← 진입점: AppState + 스레드 오케스트레이션
├── config.json              ← 임계값·저장 주기 설정
├── .env                     ← 시크릿 (Firebase API 키, .gitignore)
├── .env.example             ← 팀원용 환경 변수 템플릿
├── firebase_key.json        ← Firestore 서비스 계정 키 (.gitignore)
│
├── CLAUDE.md                ← Claude Code 컨텍스트 파일
├── DEVELOP.md               ← 이 파일
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 모듈 설명

### `turtle_neck.py` — 진입점

| 구성요소 | 설명 |
|---|---|
| `AppState` | 앱 전체 공유 상태 클래스. 전역 변수 대신 단일 인스턴스로 관리 |
| `AppState.switch_logger(uid)` | 로그인/로그아웃 시 `logger`·`upload_queue` 경로 교체 |
| `AppState.get_user_dir(uid)` | `logs/{uid}` 또는 `logs/anonymous` 경로 반환 |
| `_make_callbacks(app)` | 트레이 메뉴 콜백 팩토리. `app`을 클로저로 캡처해 딕셔너리 반환 |
| `camera_loop(app)` | 백그라운드 스레드: 프레임 수집 → 점수 계산 → 판정 → 로그 저장 |
| `upload_loop(app)` | 백그라운드 스레드: 60초 간격 Firestore 업로드 |
| `_show_stats(app)` | 로컬 JSONL 기반 오늘 통계 집계 → tk_queue로 팝업 전달 |

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

| 메서드 | 반환 | 설명 |
|---|---|---|
| `process_frame(frame)` | `float \| None` | BGR 프레임 → head_forward_score |
| `process_frame_visual(frame)` | `(score, rgb)` | 점수 + 랜드마크 오버레이 이미지 |
| `update(score)` | `(did_evaluate, state_changed)` | 윈도우 갱신 + 1초마다 히스테리시스 판정 |
| `calibrate()` | `float \| None` | 현재 윈도우 평균을 `baseline_score`로 설정 |

---

### `src/auth.py` — 인증

Firebase Auth REST API 기반 Google OAuth 로그인. 모든 메서드는 예외를 외부로 던지지 않습니다.

| 메서드 | 설명 |
|---|---|
| `login_with_google(client_secret_path)` | Google OAuth → uid 반환 (실패 시 None) |
| `logout()` | 세션 파일 삭제 + 상태 초기화 |
| `load_session()` / `save_session()` | `logs/session.json` 세션 영속화 |
| `get_valid_token()` | 유효한 ID 토큰 반환 (만료 시 자동 갱신) |
| `get_uid()` / `get_email()` / `is_logged_in()` | 상태 조회 |

---

### `src/log_config.py` — 로깅 설정

`setup_logging(log_dir)` 를 앱 시작 시 **한 번만** 호출합니다.

- **콘솔 핸들러**: `INFO` 이상 출력
- **파일 핸들러**: `DEBUG` 이상 → `logs/app.log` (1MB × 3개 회전)
- 각 모듈에서 `logging.getLogger(__name__)` 으로 로거 획득

---

### `src/startup_window.py` — UI 창

| 구성요소 | 설명 |
|---|---|
| `StartupWindow` | 앱 실행 시 메인 UI. 마스코트·카메라 피드·로그인·캘리브레이션 |
| `SettingsWindow` | 트레이 "설정 화면 열기" 클릭 시 설정 창. 마스코트·인증·캘리브레이션·카메라 포함 |
| `AuthWindow` | 트레이 "로그인" 클릭 시 컴팩트 로그인 폼 |
| `_open_signup_dialog()` | `StartupWindow`·`SettingsWindow`·`AuthWindow`에서 공용 사용하는 회원가입 Toplevel |
| `_load_mascot()` | `assets/mascot.png` 로드 (실패 시 graceful fallback) |

---

### `src/utils/upload_queue.py` — 업로드 큐

JSONL 파일 기반 영속 큐. 각 항목 구조:

```json
{
  "id": "<uuid>",
  "status": "pending | done | failed",
  "queued_at": "<ISO8601>",
  "record": { "<posture record>" }
}
```

| 메서드 | 설명 |
|---|---|
| `enqueue(record)` | pending 상태로 append |
| `get_pending()` | pending 항목 목록 반환 |
| `get_all_records(hour_prefix)` | done+pending 전체 반환 (시간대 필터 지원) |
| `mark_done(ids)` / `mark_failed(ids)` | 상태 갱신 |
| `retry_failed()` | failed → pending 으로 복원 |

---

## 기술 스택 및 의존성

| 라이브러리 | 용도 |
|---|---|
| `opencv-python` | 웹캠 프레임 캡처 |
| `mediapipe` | Pose 랜드마크 추출 |
| `pystray` | 시스템 트레이 아이콘 |
| `Pillow` | 트레이 아이콘 이미지 생성 + 카메라 피드 변환 |
| `firebase-admin` | Firestore 업로드 |
| `requests` | Firebase Auth REST API |
| `python-dotenv` | `.env` 파일에서 환경 변수 로드 |
| `winotify` | Windows 토스트 알림 |
| `plyer` | macOS 알림 |

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

### 3. 환경 변수 설정 (필수)

```bash
cp .env.example .env
```

`.env` 파일을 열어 실제 Firebase Web API 키를 입력합니다:

```
FIREBASE_API_KEY=여기에_실제_키_입력
```

> Firebase 콘솔 → 프로젝트 설정 → 일반 탭 → 웹 API 키

### 4. Firebase 서비스 계정 키 배치

Firestore 업로드를 사용하려면 `firebase_key.json`을 프로젝트 루트에 배치합니다.

> 없어도 앱은 실행됩니다 (업로드 기능만 비활성).

### 5. 실행

```bash
python turtle_neck.py
```

---

## 사용 방법

### 1단계: 시작 창

앱 실행 시 시작 창이 열립니다.

- **로그인 (선택)**: 이메일/비밀번호 입력 후 로그인 또는 회원가입
- **비로그인으로 시작**: 로그인 없이 탐지 기능만 사용

### 2단계: 캘리브레이션

1. **바른 자세로 웹캠 앞에 앉는다**
2. 시작 창 또는 트레이 메뉴 → **캘리브레이션 시작** 클릭
3. 완료 알림 후 트레이 아이콘이 초록색으로 전환

> 카메라 위치가 바뀌거나 자리를 옮기면 재캘리브레이션 권장.

### 3단계: 백그라운드 모니터링

- 시작 창이 닫히면 트레이 모드로 전환, 창이 뜨지 않음
- 트레이 아이콘 색상으로 실시간 자세 확인
- 거북목 감지 시 Windows 알림 팝업 (10초 쿨다운)

### 트레이 메뉴

| 메뉴 | 비로그인 | 로그인 |
|---|---|---|
| 캘리브레이션 | ✅ | ✅ |
| 통계 보기 | — | ✅ |
| 로그인 | ✅ | — |
| 로그아웃 | — | ✅ |
| 종료 | ✅ | ✅ |

### 로그 직접 조회

```python
import json

with open("logs/anonymous/posture_log.jsonl", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]

print(f"총 {len(records)}건 기록")
```

---

## 빌드 및 배포

> **현재 상태**: 빌드 스크립트(`build.bat`) 미작성 — exe 배포는 4단계 작업 진행 중.

### PyInstaller 빌드 명령

```bat
python -m pyinstaller ^
  --noconsole ^
  --onedir ^
  --name TurtleNeckDetector ^
  --collect-all mediapipe ^
  --hidden-import pystray._win32 ^
  --hidden-import firebase_admin ^
  --hidden-import google.cloud.firestore ^
  --add-data "config.json;." ^
  --add-data "firebase_key.json;." ^
  --add-data "assets;assets" ^
  turtle_neck.py
```

> `.env` 파일은 빌드 결과물에 포함되지 않습니다. 배포 시 별도로 동봉하거나 환경 변수로 주입하세요.

### 배포 패키지 구성

```
TurtleNeckDetector/
├── TurtleNeckDetector.exe
├── config.json
├── firebase_key.json        ← 배포 시 별도 제공
├── .env                     ← 배포 시 별도 제공
└── logs/                    ← 런타임 자동 생성
```

**빌드 결과물 크기:** MediaPipe 포함으로 약 300~500MB

---

## 개발 단계 현황

| 단계 | 상태 | 비고 |
|---|---|---|
| 1단계: 모드 분리·인증 | ✅ 완료 | AuthManager, Google OAuth, 세션 지속 |
| 2단계: 데이터 연동 파이프라인 | ✅ 완료 | uid 경로 분리, UploadQueue, Firestore 업로드 |
| 3단계: 통계 조회 | ✅ 완료 | `src/stats.py` 구현 — 로컬 오늘/7일 + Firestore 클라우드 통계 병합 |
| 4단계: exe 배포 | 🔶 진행 중 | `build.bat` 작성 완료, 신규 환경 실행 검증 필요 |
| 5단계: 운영 안정화 | ✅ 완료 | logging 모듈 도입 (log_config.py), app.log 자동 기록 |
| z축 자세 판정 | ✅ 완료 | `_Z_WEIGHT` + `_Z_GATE_Y` 로 고개 숙임 오탐 감소 |
| 스레드 안전성 | ✅ 완료 | `PostureDetector._lock` 도입, 복수 스레드 동시 접근 안전 |
| 리팩토링 | ✅ 완료 | AppState 캡슐화, 시크릿 .env 분리, 상수 추출, UI 공통 위젯 추출, 버그 수정 |

### 리팩토링에서 해결된 항목

| 항목 | 해결 방법 |
|---|---|
| 전역 변수 11개 산재 | `AppState` 클래스로 캡슐화 |
| Firebase API 키 config.json 노출 | `.env` + `python-dotenv`로 분리 |
| `print()` 남발 | `logging` 모듈로 전환 (`log_config.py`) |
| 매직 넘버 산재 | `NOTIFY_COOLDOWN`, `POLL_INTERVAL_MS`, `_WINDOW_MAXLEN` 등 상수화 |
| `camera_loop()` 리소스 누수 | `try/finally`로 `detector.close()`, `cap.release()` 보장 |
| `firebase_uploader.py` datetime 버그 | `datetime.now()` → `datetime.datetime.now()` 수정 |
| Linux 알림 미처리 | `notifier.py`에 fallback 경고 로그 추가 |
| 타입 힌트 누락 | 전체 모듈 함수 시그니처 통일 |
| `StartupWindow._on_logout` 중복 정의 | 중복 메서드 제거 |
| `AuthWindow._close` 존재하지 않는 변수 참조 | 죽은 변수 할당 제거 |
| `StartupWindow`·`SettingsWindow` 인증 UI 중복 | `_build_auth_section` / `_refresh_auth_ui` 공통 헬퍼로 추출 |

---

## 다음 과제

### 우선순위 높음

- **4단계 완성**: 실제 신규 PC에서 exe 실행 검증 (로그인·캘리브레이션·알림 전체)

### 추후 검토

- 자동 업데이트 (GitHub Releases 연동)
- 주간/월간 통계 대시보드
- Windows 시작 프로그램 자동 등록

---

## 라이선스

```
MIT License — Copyright (c) 2026 TurtleNeckDetector Contributors
```
