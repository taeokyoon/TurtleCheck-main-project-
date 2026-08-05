# 규칙 기반 거북목 판정 검증 모델 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 배포 중인 규칙 기반 거북목 판정(`src/detector.py`)은 그대로 두고, 본인이 직접 라벨링한
데이터셋으로 로지스틱 회귀를 학습시켜 "규칙 기반 판정이 실제 라벨과 얼마나 일치하는가"를
정량적으로 검증하는 개발 도구 2개(`tools/collect_labels.py`, `tools/train_model.py`)를 추가한다.

**Architecture:** `src/detector.py`의 `PostureDetector`/`PostureFeatures`/`_combine()`을 그대로
재사용하는 두 개의 독립 스크립트. 하나는 웹캠 + 키보드 토글로 라벨링 데이터를 CSV에 쌓고, 다른
하나는 그 CSV를 세션 단위로 train/test 분리해 로지스틱 회귀를 학습시키고, 규칙 기반 판정과
비교한 마크다운 리포트를 생성한다.

**Tech Stack:** Python 3.10+, OpenCV(`cv2`, 기존 의존성), `pandas`, `scikit-learn`, `pytest`
(신규 — `tools/requirements.txt`에 분리, 배포용 `requirements.txt`는 건드리지 않음)

## Global Constraints

- Python 3.10+ 문법 사용 (`float | None` 타입 힌트 등, `AGENTS.md` 기준)
- Windows/macOS 양쪽에서 동작해야 함 — 플랫폼 종속 API 금지
- 배포 코드(`turtleCheck.py`, `src/detector.py`, `src/tray_app.py`, `src/startup_window.py` 등)는
  **일체 수정하지 않는다**
- `src/detector.py`의 `PostureFeatures`, `_combine()`, `PostureDetector`를 재사용한다 — 판정 공식을
  재구현하지 않는다
- `delta_turtle` 값은 하드코딩하지 않고 `config.json`에서 로드한다
- 자동화 테스트는 순수 로직 함수(`tools/train_model.py`의 헬퍼들)에만 작성한다.
  `tools/collect_labels.py`는 웹캠/키보드 상호작용이라 자동화 테스트를 만들지 않고 수동
  실행으로 검증한다 (`docs/superpowers/specs/2026-08-04-rule-validation-model-design.md`의
  "테스트 방침" 참고)
- CSV 컬럼은 원시 피처 + baseline 원시값을 그대로 저장한다 (편차를 미리 계산해 저장하지 않는다
  — `_combine()`의 비선형 `z_gate` 때문에 편차 재현이 틀어짐)

---

### Task 1: 라벨링 데이터 수집 스크립트 (`tools/collect_labels.py`)

**Files:**
- Create: `tools/collect_labels.py`

**Interfaces:**
- Consumes: `src.detector.PostureDetector(delta_turtle, delta_ok)` — 생성자, `.process_frame(frame) -> PostureFeatures | None`, `.update(features) -> tuple[bool, bool]`, `.calibrate() -> float | None`, `.baseline_features -> PostureFeatures | None`, `.close()`
- Produces: `dataset/posture_labels.csv` (헤더: `session_id,y_score,z_forward,head_tilt,ear_z_offset,baseline_y_score,baseline_z_forward,baseline_head_tilt,baseline_ear_z_offset,label`) — Task 2/3에서 이 포맷을 그대로 읽는다.

- [ ] **Step 1: 스크립트 작성**

`tools/collect_labels.py` 전체 내용:

```python
"""
collect_labels.py — 거북목 판정 검증용 라벨링 데이터 수집 도구

실행: 프로젝트 루트에서 `python tools/collect_labels.py`

사용법:
  1. 카메라 앞에 정상 자세로 앉아 기다리면 자동으로 캘리브레이션된다.
  2. n 키: "정상" 라벨 기록 시작/종료 토글
  3. t 키: "거북목" 라벨 기록 시작/종료 토글
  4. q 키: 종료

배포 앱(turtleCheck.py)과 무관한 개발 전용 스크립트다. src/detector.py의
PostureDetector를 그대로 재사용해, 판정 로직과 100% 동일한 방식으로 피처를 계산한다.
"""
import csv
import json
import os
import sys
import time

import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.detector import PostureDetector  # noqa: E402

_CSV_PATH = os.path.join(_ROOT, "dataset", "posture_labels.csv")
_RECORD_INTERVAL = 1.0        # 초당 1행 기록
_CALIBRATION_SECONDS = 3.0

_CSV_COLUMNS = [
    "session_id", "y_score", "z_forward", "head_tilt", "ear_z_offset",
    "baseline_y_score", "baseline_z_forward", "baseline_head_tilt", "baseline_ear_z_offset",
    "label",
]


def _load_config() -> dict:
    with open(os.path.join(_ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _ensure_csv() -> None:
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)
    if not os.path.exists(_CSV_PATH):
        with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_CSV_COLUMNS)


def _calibrate(detector: PostureDetector, cap: cv2.VideoCapture) -> bool:
    print(f"캘리브레이션 중... {_CALIBRATION_SECONDS:.0f}초간 정상 자세로 가만히 있어주세요.")
    start = time.time()
    while time.time() - start < _CALIBRATION_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue
        detector.update(detector.process_frame(frame))
        cv2.imshow("collect_labels", frame)
        cv2.waitKey(1)
    baseline = detector.calibrate()
    if baseline is None:
        print("캘리브레이션 실패 — 카메라에 얼굴/어깨가 잘 보이는지 확인 후 다시 실행하세요.")
        return False
    print(f"캘리브레이션 완료 (baseline_score={baseline:.4f})")
    return True


def main() -> None:
    _ensure_csv()
    cfg = _load_config()
    detector = PostureDetector(cfg["delta_turtle"], cfg["delta_ok"])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    try:
        if not _calibrate(detector, cap):
            return
        baseline = detector.baseline_features

        print("n=정상 기록 토글, t=거북목 기록 토글, q=종료")
        recording_label: "int | None" = None
        session_id: "str | None" = None
        last_write = 0.0

        with open(_CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                features = detector.process_frame(frame)

                status = "대기 중"
                if recording_label is not None:
                    status = f"기록 중 (label={recording_label}, session={session_id})"
                display = frame.copy()
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("collect_labels", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("n"):
                    if recording_label == 0:
                        recording_label = None
                    elif recording_label is None:
                        recording_label = 0
                        session_id = time.strftime("%Y%m%d%H%M%S")
                elif key == ord("t"):
                    if recording_label == 1:
                        recording_label = None
                    elif recording_label is None:
                        recording_label = 1
                        session_id = time.strftime("%Y%m%d%H%M%S")

                now = time.time()
                if (recording_label is not None and features is not None
                        and now - last_write >= _RECORD_INTERVAL):
                    last_write = now
                    writer.writerow([
                        session_id,
                        features.y_score, features.z_forward, features.head_tilt, features.ear_z_offset,
                        baseline.y_score, baseline.z_forward, baseline.head_tilt, baseline.ear_z_offset,
                        recording_label,
                    ])
                    csv_file.flush()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 수동 실행 검증**

프로젝트 루트에서 실행:

```bash
python tools/collect_labels.py
```

확인할 것:
1. 캘리브레이션 메시지가 뜨고 3초 후 "캘리브레이션 완료" 로그가 찍힌다.
2. `n` 키를 누르면 화면에 "기록 중 (label=0, ...)"이 표시되고, 1초 간격으로 `dataset/posture_labels.csv`에 행이 추가된다 (파일을 열어 직접 확인).
3. `n`을 다시 누르면 "대기 중"으로 돌아가고 더 이상 행이 추가되지 않는다.
4. `t` 키로 label=1 기록도 동일하게 동작한다.
5. `q`로 종료 시 창이 닫히고 프로세스가 정상 종료된다.
6. `dataset/posture_labels.csv`를 열어 헤더와 각 행의 컬럼 수가 10개(`_CSV_COLUMNS` 개수)로 일치하는지 확인한다.

Expected: 위 6가지 모두 통과.

- [ ] **Step 3: Commit**

```bash
git add tools/collect_labels.py
git commit -m "feat: 라벨링 데이터 수집 스크립트 추가"
```

---

### Task 2: 검증 스크립트 순수 로직 (`tools/train_model.py` 헬퍼 함수, TDD)

**Files:**
- Create: `tools/requirements.txt`
- Create: `tools/train_model.py` (이 태스크에서는 헬퍼 함수만 — `main()`은 Task 3)
- Create: `tools/test_train_model.py`

**Interfaces:**
- Consumes: Task 1이 정의한 CSV 컬럼 포맷, `src.detector.PostureFeatures`, `src.detector._combine`
- Produces:
  - `split_by_session(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `deviation_features(df: pd.DataFrame) -> pd.DataFrame` (컬럼: `y_score, z_forward, head_tilt, ear_z_offset`)
  - `rule_based_predict(df: pd.DataFrame, delta_turtle: float) -> pd.Series`
  - `compute_metrics(y_true, y_pred) -> dict` (키: `accuracy, precision, recall, confusion_matrix`)
  - `format_report(model_metrics: dict, rule_metrics: dict, n_train: int, n_test: int) -> str`
  - Task 3의 `main()`이 이 5개 함수를 그대로 가져다 쓴다.

- [ ] **Step 1: `tools/requirements.txt` 작성**

```text
# tools/ 개발 전용 의존성 — 배포용 requirements.txt에는 포함하지 않는다
pandas>=2.0.0
scikit-learn>=1.3.0
pytest>=7.4.0
```

- [ ] **Step 2: 설치**

```bash
pip install -r tools/requirements.txt
```

- [ ] **Step 3: 실패하는 테스트 작성**

`tools/test_train_model.py` 전체 내용:

```python
import pandas as pd
import pytest

from train_model import (
    compute_metrics,
    deviation_features,
    format_report,
    rule_based_predict,
    split_by_session,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "session_id":            ["s1", "s1", "s2", "s2", "s3", "s3"],
        "y_score":                [-0.40, -0.55, -0.42, -0.30, -0.60, -0.20],
        "z_forward":               [0.0] * 6,
        "head_tilt":               [0.0] * 6,
        "ear_z_offset":            [0.0] * 6,
        "baseline_y_score":       [-0.40] * 6,
        "baseline_z_forward":      [0.0] * 6,
        "baseline_head_tilt":      [0.0] * 6,
        "baseline_ear_z_offset":   [0.0] * 6,
        "label":                   [0, 1, 0, 0, 1, 0],
    })


def test_split_by_session_holds_out_whole_sessions():
    df = _sample_df()
    train_df, test_df = split_by_session(df, test_frac=1 / 3)

    train_sessions = set(train_df["session_id"])
    test_sessions = set(test_df["session_id"])
    assert train_sessions.isdisjoint(test_sessions)
    assert test_sessions == {"s3"}
    assert train_sessions == {"s1", "s2"}


def test_deviation_features_subtracts_baseline():
    df = _sample_df()
    dev = deviation_features(df)
    assert dev["y_score"].tolist() == pytest.approx([0.0, -0.15, -0.02, 0.10, -0.20, 0.20])
    assert dev["z_forward"].tolist() == [0.0] * 6


def test_rule_based_predict_flags_large_negative_deviation():
    df = _sample_df()
    pred = rule_based_predict(df, delta_turtle=0.10)
    assert pred.tolist() == [0, 1, 0, 0, 1, 0]


def test_compute_metrics_perfect_prediction():
    m = compute_metrics([0, 1, 0, 1], [0, 1, 0, 1])
    assert m["accuracy"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]


def test_format_report_contains_key_numbers():
    m = {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "confusion_matrix": [[1, 0], [0, 1]]}
    report = format_report(m, m, n_train=10, n_test=5)
    assert "0.900" in report
    assert "10행" in report
    assert "5행" in report
```

- [ ] **Step 4: 테스트 실패 확인**

```bash
cd tools && python -m pytest test_train_model.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'train_model'` (아직 파일 없음)

- [ ] **Step 5: `tools/train_model.py` 헬퍼 함수 구현**

`tools/train_model.py` 전체 내용 (이 태스크에서는 헬퍼 함수 + import까지만; `main()`은 Task 3에서 추가):

```python
"""
train_model.py — 라벨링 데이터로 로지스틱 회귀를 학습시키고, 규칙 기반 판정과
비교하는 검증 리포트를 생성한다.

실행: 프로젝트 루트에서 `python tools/train_model.py`

학습된 모델은 배포되지 않는다. 규칙 기반 판정(src/detector.py)이 실제 라벨과
얼마나 일치하는지를 검증하는 근거 자료(dataset/validation_report.md)를 만드는 것이 목적이다.
"""
import os
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.detector import PostureFeatures, _combine  # noqa: E402

_FEATURE_COLS = ["y_score", "z_forward", "head_tilt", "ear_z_offset"]


def split_by_session(df: pd.DataFrame, test_frac: float = 0.2) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """session_id 단위로 train/test를 나눈다. 행 단위 무작위 분리는 하지 않는다 —
    같은 세션의 행들은 거의 동일해서 무작위 분리 시 데이터 누수가 발생한다."""
    session_ids = sorted(df["session_id"].unique())
    n_test = max(1, round(len(session_ids) * test_frac))
    test_sessions = set(session_ids[-n_test:])
    test_df = df[df["session_id"].isin(test_sessions)]
    train_df = df[~df["session_id"].isin(test_sessions)]
    return train_df, test_df


def deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """원시 피처에서 baseline을 뺀 편차 피처를 반환한다 (모델 학습 입력용)."""
    return pd.DataFrame({col: df[col] - df[f"baseline_{col}"] for col in _FEATURE_COLS})


def rule_based_predict(df: pd.DataFrame, delta_turtle: float) -> pd.Series:
    """src/detector.py와 동일한 공식(_combine)으로 규칙 기반 예측을 재현한다."""
    preds = []
    for row in df.itertuples():
        raw = PostureFeatures(row.y_score, row.z_forward, row.head_tilt, row.ear_z_offset)
        baseline = PostureFeatures(
            row.baseline_y_score, row.baseline_z_forward,
            row.baseline_head_tilt, row.baseline_ear_z_offset,
        )
        deviation = _combine(raw) - _combine(baseline)
        preds.append(1 if deviation < -delta_turtle else 0)
    return pd.Series(preds, index=df.index)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def format_report(model_metrics: dict, rule_metrics: dict, n_train: int, n_test: int) -> str:
    def _section(name: str, m: dict) -> str:
        return (
            f"### {name}\n\n"
            f"- Accuracy: {m['accuracy']:.3f}\n"
            f"- Precision: {m['precision']:.3f}\n"
            f"- Recall: {m['recall']:.3f}\n"
            f"- Confusion Matrix (행=실제, 열=예측, [0,1] 순서): {m['confusion_matrix']}\n"
        )

    return (
        "# 거북목 판정 검증 리포트\n\n"
        f"- 학습 데이터 수: {n_train}행\n"
        f"- 검증(test) 데이터 수: {n_test}행\n\n"
        f"{_section('학습된 로지스틱 회귀 모델', model_metrics)}\n"
        f"{_section('규칙 기반 판정 (src/detector.py)', rule_metrics)}\n"
    )
```

- [ ] **Step 6: 테스트 통과 확인**

```bash
cd tools && python -m pytest test_train_model.py -v
```

Expected: PASS (5개 테스트 모두)

- [ ] **Step 7: Commit**

```bash
git add tools/requirements.txt tools/train_model.py tools/test_train_model.py
git commit -m "feat: 검증 스크립트 순수 로직 추가 (TDD)"
```

---

### Task 3: `main()` 오케스트레이션 + 통합 검증

**Files:**
- Modify: `tools/train_model.py` (Task 2가 만든 파일에 `main()` 추가)

**Interfaces:**
- Consumes: Task 2의 `split_by_session`, `deviation_features`, `rule_based_predict`, `compute_metrics`, `format_report`
- Produces: `dataset/validation_report.md`, `dataset/model.pkl` (실행 시 생성, 배포에는 쓰이지 않음)

- [ ] **Step 1: `main()` 추가**

`tools/train_model.py` 최상단 import 블록을 다음으로 교체한다:

```python
import json
import os
import pickle
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
```

그리고 파일 끝(마지막 함수 `format_report` 뒤)에 아래 내용을 추가한다:

```python
_CSV_PATH = os.path.join(_ROOT, "dataset", "posture_labels.csv")
_REPORT_PATH = os.path.join(_ROOT, "dataset", "validation_report.md")
_MODEL_PATH = os.path.join(_ROOT, "dataset", "model.pkl")


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} 에 데이터가 없습니다. collect_labels.py로 먼저 데이터를 모아주세요.")
    if df["label"].nunique() < 2:
        raise ValueError("정상(0)/거북목(1) 라벨이 둘 다 있어야 합니다.")
    return df


def main() -> None:
    with open(os.path.join(_ROOT, "config.json"), encoding="utf-8") as f:
        delta_turtle = json.load(f)["delta_turtle"]

    df = load_dataset(_CSV_PATH)
    train_df, test_df = split_by_session(df)

    X_train, y_train = deviation_features(train_df), train_df["label"]
    X_test, y_test = deviation_features(test_df), test_df["label"]

    model = LogisticRegression()
    model.fit(X_train, y_train)
    model_metrics = compute_metrics(y_test, model.predict(X_test))

    rule_pred = rule_based_predict(test_df, delta_turtle)
    rule_metrics = compute_metrics(y_test, rule_pred)

    report = format_report(model_metrics, rule_metrics, len(train_df), len(test_df))
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(report)
    print(f"리포트 저장: {_REPORT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 작은 가짜 데이터셋으로 통합 검증**

프로젝트 루트에서 임시 CSV를 만들어 `main()`이 끝까지 동작하는지 확인한다 (실제 웹캠 데이터가
아직 없어도 되는 배관 검증용):

```bash
python -c "
import pandas as pd
df = pd.DataFrame({
    'session_id': ['s1']*10 + ['s2']*10 + ['s3']*10 + ['s4']*10,
    'y_score': [-0.40]*10 + [-0.55]*10 + [-0.42]*10 + [-0.58]*10,
    'z_forward': [0.0]*40,
    'head_tilt': [0.0]*40,
    'ear_z_offset': [0.0]*40,
    'baseline_y_score': [-0.40]*40,
    'baseline_z_forward': [0.0]*40,
    'baseline_head_tilt': [0.0]*40,
    'baseline_ear_z_offset': [0.0]*40,
    'label': [0]*10 + [1]*10 + [0]*10 + [1]*10,
})
import os
os.makedirs('dataset', exist_ok=True)
df.to_csv('dataset/posture_labels.csv', index=False)
"
python tools/train_model.py
```

Expected:
1. 콘솔에 리포트가 출력되고 `리포트 저장: .../dataset/validation_report.md` 로그가 뜬다.
2. `dataset/validation_report.md`가 생성되고, accuracy/precision/recall 숫자와 confusion
   matrix가 들어있다.
3. `dataset/model.pkl`이 생성된다.

검증이 끝나면 이 임시 CSV는 실제 라벨링 데이터로 덮어써질 것이므로 지우지 않아도 된다
(Task 1에서 만든 진짜 데이터로 다시 실행하면 자동으로 덮어써진다).

- [ ] **Step 3: Commit**

```bash
git add tools/train_model.py
git commit -m "feat: 검증 스크립트 main() 오케스트레이션 추가"
```

---

### Task 4: `AGENTS.md` 문서 갱신

**Files:**
- Modify: `AGENTS.md:9-10`
- Modify: `AGENTS.md:53-65` (기존 "v2.0 개발 제약" 섹션 전체 교체)

**Interfaces:** 없음 (문서 전용 변경)

- [ ] **Step 1: 프로젝트 개요의 목표 문구 교체**

`AGENTS.md:9-10`의 아래 두 줄을:

```markdown
- **현재(v1.0)**: MediaPipe Pose 랜드마크 + 규칙 기반 임계값으로 거북목 판정
- **목표(v2.0)**: EfficientNet-B0 전이학습 모델로 판정 엔진 교체 + LLM 기반 코칭 피드백 추가
```

아래로 교체:

```markdown
- **현재(v1.0)**: MediaPipe Pose 랜드마크 + 규칙 기반 임계값으로 거북목 판정
- **검증**: `tools/`에서 직접 라벨링한 데이터셋으로 로지스틱 회귀를 학습시켜, 규칙 기반 판정이
  실제 라벨과 얼마나 일치하는지 정량 검증한다(`dataset/validation_report.md`). 판정 로직 자체는
  교체하지 않는다 — 배경은
  [docs/superpowers/specs/2026-08-04-rule-validation-model-design.md](docs/superpowers/specs/2026-08-04-rule-validation-model-design.md) 참고.
```

- [ ] **Step 2: "v2.0 개발 제약" 섹션을 "검증 도구" 섹션으로 교체**

`AGENTS.md:53-65`의 아래 블록 전체를:

```markdown
## v2.0 개발 제약

v2.0 작업 시 반드시 준수해야 할 규칙.

| 규칙 | 내용 |
|---|---|
| MediaPipe 역할 | 전처리(ROI 크롭) 전용. 거북목 판단 로직 일체 포함 금지 |
| 판정 주체 | EfficientNet-B0 파인튜닝 모델만 판단 — 규칙 기반 임계값 혼용 금지 |
| 교체 범위 | `src/detector.py`의 `_calc_features()`/`_combine()` 규칙 기반 판정 제거 후 모델 추론으로 대체. `camera_loop`, 트레이 구조는 유지 |
| 출력 클래스 | 정상 / 경증 거북목 / 중증 거북목 (3-class) |
| 실시간 성능 | 노트북 CPU에서 30fps 유지 — 무거운 전처리 추가 금지 |
| 데이터셋 경로 | `dataset/normal/`, `dataset/mild/`, `dataset/severe/` |
| 학습 환경 | Google Colab (GPU). 로컬 학습 코드 작성 불필요 |
```

아래로 교체:

```markdown
## 검증 도구 (`tools/`)

| 파일 | 역할 |
|---|---|
| `tools/collect_labels.py` | 라벨링 데이터 수집 (개발 전용, 배포 앱과 무관) |
| `tools/train_model.py` | 로지스틱 회귀 학습 + 규칙 기반 판정과의 비교 리포트 생성 |

배포 코드(`turtleCheck.py`, `src/*`)는 이 도구들과 무관하게 그대로 유지된다. 학습된 모델은
배포되지 않으며, `dataset/validation_report.md`는 규칙 기반 판정의 정확도를 뒷받침하는
근거 자료로만 쓰인다. 상세 설계는
[docs/superpowers/specs/2026-08-04-rule-validation-model-design.md](docs/superpowers/specs/2026-08-04-rule-validation-model-design.md) 참고.
```

- [ ] **Step 3: Commit**

```bash
git add AGENTS.md
git commit -m "docs: v2.0 CNN 로드맵을 검증 도구 방향으로 갱신"
```
