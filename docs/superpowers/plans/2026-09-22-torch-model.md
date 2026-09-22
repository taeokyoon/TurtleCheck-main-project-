# 랜드마크 기반 PyTorch 거북목 판정 모델 — 구현 계획 (Phase 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `dataset/posture_v2.csv`의 원시 랜드마크로 MLP·GRU를 학습시키고, 규칙 기반 판정과 3자 비교 리포트를 만든 뒤 ONNX로 내보낸다.

**Architecture:** 모든 로직은 `tools/train_torch.py` 한 파일에 "순수 함수(피처·시퀀스·분할·표준화) → 모델 클래스 → 학습 루프 → main()" 순으로 둔다. 순수 함수와 모델 forward는 pytest로, 학습 루프는 실제 실행으로 검증한다. 기존 `tools/train_model.py`의 `rule_based_predict()`·`compute_metrics()`를 import해 규칙 기반 비교를 재현한다.

**Tech Stack:** Python 3.11, PyTorch(CPU), numpy, pandas, scikit-learn(지표), matplotlib(그래프), onnx + onnxruntime(내보내기·검증), pytest

**Spec:** `docs/superpowers/specs/2026-09-22-torch-model-design.md`

## Global Constraints

- 배포 코드(`turtleCheck.py`, `src/`)와 배포 `requirements.txt`는 수정하지 않는다.
- `dataset/posture_v2.csv`는 읽기만 한다.
- 입력 피처 = 어깨 기준 정규화 랜드마크 13개 × (x,y,z) 39 + visibility 13 = **52차원**. 캘리브레이션(baseline) 미사용.
- GRU 윈도우 `SEQ_LEN = 10`, 스트라이드 1, 같은 세션·같은 라벨 연속 구간 안에서만.
- 검증 = 세션 단위 Leave-One-Out. 표준화 mean/std는 **train 폴드에서만** 계산.
- 판정 기준 확률 50% 고정. 재현성 `torch.manual_seed(42)`, `np.random.seed(42)`.
- 모든 pytest는 `cd tools && pytest`로 실행한다 (기존 테스트가 이 위치 기준).
- **git 작업(add/commit)은 사용자가 직접 한다.** 각 Task 끝의 "Save 기준"은 저장 시점 안내일 뿐, 실행자가 커밋하지 않는다.
- 코드 스타일: 기존 `tools/train_model.py`처럼 한국어 docstring·주석, 직관적 이름, 초보자가 읽을 수 있게.

## 파일 구조

| 파일 | 역할 |
|---|---|
| `tools/train_torch.py` (신규) | 피처 변환 → 시퀀스 → 폴드 → 표준화 → 모델 → 학습 → 리포트 → ONNX. `main()`으로 실행 |
| `tools/test_train_torch.py` (신규) | 순수 함수 + 모델 forward pytest |
| `tools/requirements.txt` (수정) | torch(CPU), onnx, onnxruntime, matplotlib 추가 |
| `dataset/torch_report.md`, `dataset/torch_report/*.png`, `dataset/mlp_model.onnx`, `dataset/gru_model.onnx`, `dataset/feature_norm.json` | 실행 결과물 (Task 5·6에서 생성) |
| `AGENTS.md`, `DEVELOP.md` (수정) | Task 7 |

---

### Task 1: 의존성 설치 + 어깨 기준 정규화 `normalize_landmarks()`

**Files:**
- Modify: `tools/requirements.txt`
- Create: `tools/train_torch.py`
- Create: `tools/test_train_torch.py`

**Interfaces:**
- Produces: `LANDMARK_COUNT = 13`, `FEATURE_DIM = 52`, `landmark_xyz_columns() -> list[str]`, `filter_valid_rows(df) -> pd.DataFrame`, `normalize_landmarks(df: pd.DataFrame) -> np.ndarray` (shape `(N, 52)`, dtype float32)

- [ ] **Step 1: `tools/requirements.txt`에 의존성 추가**

파일 끝에 추가:

```
# ── Phase 2: PyTorch 학습 (tools/train_torch.py) ──
# torch는 CPU 전용 휠을 받는다 — 이 프로젝트 데이터 규모엔 GPU가 불필요하고, CPU 휠이 훨씬 작다
--extra-index-url https://download.pytorch.org/whl/cpu
torch>=2.2.0
onnx>=1.15.0
onnxruntime>=1.17.0
matplotlib>=3.8.0
```

- [ ] **Step 2: 설치**

Run: `cd C:\Users\taeok\desktop\turtleCheck && .venv\Scripts\python.exe -m pip install -r tools\requirements.txt`
Expected: 오류 없이 종료. 확인: `.venv\Scripts\python.exe -c "import torch, onnx, onnxruntime, matplotlib; print(torch.__version__)"` → 버전 출력.

- [ ] **Step 3: 실패하는 테스트 작성**

`tools/test_train_torch.py`:

```python
import numpy as np
import pandas as pd

# train_torch 가 import 시 프로젝트 루트를 sys.path 에 넣으므로 먼저 import 한다
from train_torch import (  # noqa: I001
    FEATURE_DIM,
    LANDMARK_COUNT,
    filter_valid_rows,
    normalize_landmarks,
)


def _frame_df(points, vis=1.0) -> pd.DataFrame:
    """13개 (x,y,z) 튜플을 받아 posture_v2.csv 와 같은 컬럼 이름의 1행 DataFrame 을 만든다."""
    row = {}
    for i, (x, y, z) in enumerate(points):
        row[f"lm{i:02d}_x"] = x
        row[f"lm{i:02d}_y"] = y
        row[f"lm{i:02d}_z"] = z
        row[f"lm{i:02d}_v"] = vis
    return pd.DataFrame([row])


def _sample_points():
    """코는 (0.5, 0.3, -0.1), 어깨는 (0.3, 0.5, 0)/(0.7, 0.5, 0), 나머지는 (0.5, 0.4, 0)."""
    pts = [(0.5, 0.4, 0.0)] * LANDMARK_COUNT
    pts[0] = (0.5, 0.3, -0.1)     # NOSE
    pts[11] = (0.3, 0.5, 0.0)     # LEFT_SHOULDER
    pts[12] = (0.7, 0.5, 0.0)     # RIGHT_SHOULDER
    return pts


def test_normalize_puts_shoulder_center_at_origin_and_width_one():
    feats = normalize_landmarks(_frame_df(_sample_points()))
    assert feats.shape == (1, FEATURE_DIM)
    assert feats.dtype == np.float32
    # 컬럼 순서: x0,y0,z0, x1,y1,z1, ... → 11번 어깨는 index 33~35, 12번은 36~38
    left, right = feats[0, 33:36], feats[0, 36:39]
    np.testing.assert_allclose((left + right) / 2, [0, 0, 0], atol=1e-6)   # 중심이 원점
    np.testing.assert_allclose(abs(left[0] - right[0]), 1.0, atol=1e-6)     # 너비가 1
    # 코: (0.5-0.5)/0.4, (0.3-0.5)/0.4, (-0.1-0)/0.4
    np.testing.assert_allclose(feats[0, 0:3], [0.0, -0.5, -0.25], atol=1e-6)
    # visibility 13개는 맨 뒤
    np.testing.assert_allclose(feats[0, 39:], 1.0)


def test_normalize_is_invariant_to_camera_shift_and_zoom():
    """카메라를 옮기거나(평행이동) 가까이 가도(확대) 같은 자세면 같은 피처가 나와야 한다."""
    base = normalize_landmarks(_frame_df(_sample_points()))
    moved = [(2 * x + 0.1, 2 * y + 0.2, 2 * z + 0.05) for x, y, z in _sample_points()]
    np.testing.assert_allclose(normalize_landmarks(_frame_df(moved)), base, atol=1e-6)


def test_filter_valid_rows_drops_zero_shoulder_width():
    good = _frame_df(_sample_points())
    pts = _sample_points()
    pts[11] = pts[12] = (0.5, 0.5, 0.0)      # 어깨가 겹침 → 너비 0
    bad = _frame_df(pts)
    df = pd.concat([good, bad], ignore_index=True)
    assert len(filter_valid_rows(df)) == 1
```

- [ ] **Step 4: 테스트 실패 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest test_train_torch.py -q`
Expected: `ModuleNotFoundError: No module named 'train_torch'`

- [ ] **Step 5: `tools/train_torch.py` 작성 (Task 1 범위)**

```python
"""
train_torch.py — posture_v2.csv 의 원시 랜드마크로 MLP·GRU 를 학습시키고,
규칙 기반 판정과 3자 비교 리포트를 만든 뒤 ONNX 로 내보낸다.

실행: 프로젝트 루트에서 `python tools/train_torch.py`

흐름:
  1. CSV 로드 → 어깨 기준 정규화 (프레임당 52차원)
  2. GRU 용 시퀀스 생성 (같은 세션·같은 라벨 구간에서 10스텝 윈도우)
  3. 세션 단위 Leave-One-Out 교차검증: MLP / GRU / 규칙 기반 지표 비교
  4. 전체 데이터로 최종 재학습 → ONNX + feature_norm.json 내보내기
  5. dataset/torch_report.md + 그래프 저장

배포 코드(src/)는 수정하지 않는다. 설계: docs/superpowers/specs/2026-09-22-torch-model-design.md
"""
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

LANDMARK_COUNT = 13          # MediaPipe Pose 0~12번 (코·눈·귀·입·어깨)
FEATURE_DIM = LANDMARK_COUNT * 3 + LANDMARK_COUNT   # xyz 39 + visibility 13 = 52
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_MIN_SHOULDER_WIDTH = 0.01   # 이보다 좁으면 측면 촬영으로 보고 제외


# ── 1. 피처 변환 ─────────────────────────────────────────────────────────────

def landmark_xyz_columns() -> list[str]:
    """lm00_x, lm00_y, lm00_z, lm01_x, ... 순서의 컬럼 이름 (visibility 제외)."""
    return [f"lm{i:02d}_{axis}" for i in range(LANDMARK_COUNT) for axis in ("x", "y", "z")]


def _shoulder_width(df: pd.DataFrame) -> np.ndarray:
    return np.abs(df[f"lm{_LEFT_SHOULDER:02d}_x"].to_numpy() - df[f"lm{_RIGHT_SHOULDER:02d}_x"].to_numpy())


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """어깨 너비가 0에 가까운 행(측면 촬영·인식 오류)을 제외한다."""
    return df[_shoulder_width(df) > _MIN_SHOULDER_WIDTH].reset_index(drop=True)


def normalize_landmarks(df: pd.DataFrame) -> np.ndarray:
    """어깨 중심을 원점, 어깨 너비를 1로 만드는 정규화. (N, 52) float32 반환.

    카메라 위치(평행이동)와 거리(확대·축소)에 무관한 값이 되어, 캘리브레이션 없이도
    다른 환경에서 찍은 데이터와 같은 잣대로 비교할 수 있다."""
    xyz = df[landmark_xyz_columns()].to_numpy(dtype=np.float64).reshape(len(df), LANDMARK_COUNT, 3)
    center = (xyz[:, _LEFT_SHOULDER] + xyz[:, _RIGHT_SHOULDER]) / 2          # (N, 3)
    width = _shoulder_width(df)                                               # (N,)
    normalized = (xyz - center[:, None, :]) / width[:, None, None]            # (N, 13, 3)
    visibility = df[[f"lm{i:02d}_v" for i in range(LANDMARK_COUNT)]].to_numpy(dtype=np.float64)
    return np.concatenate([normalized.reshape(len(df), -1), visibility], axis=1).astype(np.float32)
```

- [ ] **Step 6: 테스트 통과 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest test_train_torch.py -q`
Expected: `3 passed`

---

### Task 2: GRU용 시퀀스 생성 `make_sequences()`

**Files:**
- Modify: `tools/train_torch.py` (Task 1 코드 아래에 추가)
- Modify: `tools/test_train_torch.py`

**Interfaces:**
- Produces: `SEQ_LEN = 10`, `make_sequences(features: np.ndarray, labels: np.ndarray, sessions: np.ndarray, seq_len: int = SEQ_LEN) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]` → `(X (M, seq_len, D), y (M,), groups (M,), last_idx (M,))`. `last_idx[k]`는 k번째 윈도우의 마지막 프레임이 원본 배열에서 몇 번째 행인지 — 규칙 기반과 같은 프레임으로 비교하기 위해 쓴다. 입력은 세션·시간순으로 정렬돼 있다고 가정한다.

- [ ] **Step 1: 실패하는 테스트 추가**

`tools/test_train_torch.py` 상단 import에 `SEQ_LEN, make_sequences` 추가하고 파일 끝에:

```python
def _index_features(n: int, dim: int = 2) -> np.ndarray:
    """0번 열에 행 번호를 넣어, 윈도우가 어느 행들로 만들어졌는지 추적할 수 있게 한다."""
    feats = np.zeros((n, dim), dtype=np.float32)
    feats[:, 0] = np.arange(n)
    return feats


def test_make_sequences_never_crosses_label_boundary():
    labels = np.array([0] * 12 + [1] * 12)
    sessions = np.array(["s1"] * 24)
    X, y, groups, last_idx = make_sequences(_index_features(24), labels, sessions, seq_len=SEQ_LEN)
    # 각 라벨 구간 12행 → 12-10+1 = 3개 윈도우, 총 6개
    assert X.shape == (6, SEQ_LEN, 2)
    assert list(y) == [0, 0, 0, 1, 1, 1]
    # 윈도우 안의 모든 행이 윈도우 라벨과 같아야 한다 (경계를 넘지 않았다는 뜻)
    rows_in_window = X[:, :, 0].astype(int)
    assert (labels[rows_in_window] == y[:, None]).all()
    # last_idx 는 윈도우의 마지막 행 번호
    assert list(last_idx) == [9, 10, 11, 21, 22, 23]
    assert (groups == "s1").all()


def test_make_sequences_never_crosses_session_boundary():
    labels = np.zeros(20, dtype=int)
    sessions = np.array(["a"] * 10 + ["b"] * 10)
    X, y, groups, _ = make_sequences(_index_features(20), labels, sessions, seq_len=SEQ_LEN)
    assert X.shape == (2, SEQ_LEN, 2)          # 세션마다 딱 1개
    assert list(groups) == ["a", "b"]


def test_make_sequences_drops_runs_shorter_than_seq_len():
    labels = np.array([0] * 5 + [1] * 15)
    sessions = np.array(["s1"] * 20)
    X, y, _, _ = make_sequences(_index_features(20), labels, sessions, seq_len=SEQ_LEN)
    assert X.shape == (6, SEQ_LEN, 2)          # 5행 구간은 버려지고 15행 구간에서 6개
    assert (y == 1).all()
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest test_train_torch.py -q`
Expected: `ImportError: cannot import name 'SEQ_LEN'`

- [ ] **Step 3: 구현 추가**

`tools/train_torch.py`의 `_MIN_SHOULDER_WIDTH` 아래에 상수, 피처 변환 섹션 아래에 함수:

```python
SEQ_LEN = 10                 # GRU 윈도우 길이 (5Hz × 2초)
```

```python
# ── 2. 시퀀스 생성 (GRU 용) ──────────────────────────────────────────────────

def make_sequences(features: np.ndarray, labels: np.ndarray, sessions: np.ndarray,
                   seq_len: int = SEQ_LEN):
    """같은 세션·같은 라벨이 이어지는 구간 안에서만 seq_len 길이 윈도우를 1칸씩 밀며 자른다.

    라벨이 바뀌는 지점을 걸치는 윈도우는 라벨을 하나로 정할 수 없으므로 만들지 않는다.
    Returns:
        X        (M, seq_len, D)  윈도우 피처
        y        (M,)             윈도우 라벨 (= 구간 라벨)
        groups   (M,)             윈도우가 속한 세션 (폴드 분할용)
        last_idx (M,)             윈도우 마지막 프레임의 원본 행 번호 (규칙 기반 비교용)
    """
    X, y, groups, last_idx = [], [], [], []
    n = len(labels)
    run_start = 0
    for i in range(1, n + 1):
        run_ended = i == n or labels[i] != labels[run_start] or sessions[i] != sessions[run_start]
        if not run_ended:
            continue
        # [run_start, i) 구간이 끝났다 → 이 안에서 윈도우를 자른다
        for start in range(run_start, i - seq_len + 1):
            X.append(features[start:start + seq_len])
            y.append(labels[run_start])
            groups.append(sessions[run_start])
            last_idx.append(start + seq_len - 1)
        run_start = i
    dim = features.shape[1]
    return (
        np.array(X, dtype=np.float32).reshape(-1, seq_len, dim),
        np.array(y),
        np.array(groups),
        np.array(last_idx, dtype=int),
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest test_train_torch.py -q`
Expected: `6 passed`

---

### Task 3: 폴드 분할 `leave_one_group_out()` + 표준화

**Files:**
- Modify: `tools/train_torch.py`
- Modify: `tools/test_train_torch.py`

**Interfaces:**
- Produces:
  - `leave_one_group_out(groups: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]` — `(train_idx, test_idx)` 목록, 그룹 이름 정렬순
  - `split_train_val(train_idx: np.ndarray, groups: np.ndarray, val_frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]` — early stopping용. train 안에 그룹이 2개 이상이면 마지막 그룹이 val, 1개면 뒤 20%
  - `fit_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]` → `(mean (D,), std (D,))`. X는 2D `(N, D)` 또는 3D `(N, T, D)`
  - `apply_standardizer(X, mean, std) -> np.ndarray`

- [ ] **Step 1: 실패하는 테스트 추가**

import에 `apply_standardizer, fit_standardizer, leave_one_group_out, split_train_val` 추가하고 파일 끝에:

```python
def test_leave_one_group_out_holds_out_each_group_once():
    groups = np.array(["a"] * 3 + ["b"] * 2 + ["c"] * 4)
    folds = leave_one_group_out(groups)
    assert len(folds) == 3
    held_out = []
    for train_idx, test_idx in folds:
        assert len(set(train_idx) & set(test_idx)) == 0          # 겹침 없음
        assert len(train_idx) + len(test_idx) == len(groups)     # 빠진 행 없음
        assert len(np.unique(groups[test_idx])) == 1             # test 는 그룹 딱 하나
        held_out.append(groups[test_idx][0])
    assert held_out == ["a", "b", "c"]


def test_split_train_val_uses_last_group_when_multiple():
    groups = np.array(["a"] * 3 + ["b"] * 2 + ["c"] * 4)
    train_idx = np.arange(9)
    tr, val = split_train_val(train_idx, groups)
    assert set(groups[val]) == {"c"}
    assert set(groups[tr]) == {"a", "b"}


def test_split_train_val_uses_tail_when_single_group():
    groups = np.array(["a"] * 10)
    tr, val = split_train_val(np.arange(10), groups, val_frac=0.2)
    assert list(val) == [8, 9]
    assert list(tr) == list(range(8))


def test_standardizer_gives_zero_mean_unit_std_on_2d_and_3d():
    rng = np.random.default_rng(0)
    X2 = rng.normal(5, 3, size=(50, 4)).astype(np.float32)
    mean, std = fit_standardizer(X2)
    Z2 = apply_standardizer(X2, mean, std)
    np.testing.assert_allclose(Z2.mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(Z2.std(axis=0), 1, atol=1e-4)

    X3 = rng.normal(-2, 0.5, size=(20, 10, 4)).astype(np.float32)
    mean, std = fit_standardizer(X3)
    assert mean.shape == (4,) and std.shape == (4,)
    Z3 = apply_standardizer(X3, mean, std)
    np.testing.assert_allclose(Z3.reshape(-1, 4).mean(axis=0), 0, atol=1e-5)
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest test_train_torch.py -q`
Expected: `ImportError: cannot import name 'apply_standardizer'`

- [ ] **Step 3: 구현 추가**

`tools/train_torch.py`의 시퀀스 섹션 아래:

```python
# ── 3. 폴드 분할 + 표준화 ────────────────────────────────────────────────────

def leave_one_group_out(groups: np.ndarray) -> list:
    """그룹(세션 또는 사람)을 하나씩 test 로 빼는 폴드 목록. 그룹 이름 정렬순."""
    folds = []
    for name in sorted(np.unique(groups)):
        test_idx = np.where(groups == name)[0]
        train_idx = np.where(groups != name)[0]
        folds.append((train_idx, test_idx))
    return folds


def split_train_val(train_idx: np.ndarray, groups: np.ndarray, val_frac: float = 0.2):
    """early stopping 용 검증 세트. train 안에 그룹이 2개 이상이면 마지막 그룹을 통째로,
    1개뿐이면 뒤쪽 val_frac 만큼을 val 로 쓴다 (행 단위 무작위 분리는 누수가 생기므로 안 함)."""
    train_groups = sorted(np.unique(groups[train_idx]))
    if len(train_groups) >= 2:
        is_val = groups[train_idx] == train_groups[-1]
        return train_idx[~is_val], train_idx[is_val]
    cut = int(len(train_idx) * (1 - val_frac))
    return train_idx[:cut], train_idx[cut:]


def fit_standardizer(X: np.ndarray):
    """마지막 차원(피처) 기준 mean/std. 2D (N, D) 와 3D (N, T, D) 모두 지원."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8    # 상수 피처(std=0)로 0 나눗셈 방지
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest test_train_torch.py -q`
Expected: `10 passed`

**Save 기준 1** — 여기까지가 spec의 "피처·시퀀스·분할·표준화 함수 + 테스트 통과". 사용자에게 저장 시점임을 알린다.

---

### Task 4: 모델 클래스 `MlpModel` / `GruModel`

**Files:**
- Modify: `tools/train_torch.py`
- Modify: `tools/test_train_torch.py`

**Interfaces:**
- Produces: `MlpModel(nn.Module)` — forward `(B, 52) -> (B, 1)` 로짓; `GruModel(nn.Module)` — forward `(B, T, 52) -> (B, 1)` 로짓. 둘 다 인자 없이 생성 가능 (`input_dim=FEATURE_DIM` 기본값).

- [ ] **Step 1: 실패하는 테스트 추가**

import에 `GruModel, MlpModel` 추가, 파일 상단에 `import torch` 추가, 파일 끝에:

```python
def test_mlp_forward_shape():
    model = MlpModel()
    out = model(torch.zeros(7, FEATURE_DIM))
    assert out.shape == (7, 1)


def test_gru_forward_shape():
    model = GruModel()
    out = model(torch.zeros(7, SEQ_LEN, FEATURE_DIM))
    assert out.shape == (7, 1)


def test_models_are_small():
    """1,156행에 큰 모델은 암기만 한다 — 파라미터 수 상한을 테스트로 고정."""
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(MlpModel()) < 5_000
    assert count(GruModel()) < 15_000
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest test_train_torch.py -q`
Expected: `ImportError: cannot import name 'GruModel'`

- [ ] **Step 3: 구현 추가**

`tools/train_torch.py` 상단 import에 `import torch`, `from torch import nn` 추가. 표준화 섹션 아래:

```python
# ── 4. 모델 ──────────────────────────────────────────────────────────────────
# 둘 다 일부러 작게 잡았다 — 데이터가 1천 행대라 큰 모델은 암기(과적합)만 한다.

class MlpModel(nn.Module):
    """단일 프레임 52차원 → 거북목 로짓 1개."""

    def __init__(self, input_dim: int = FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


class GruModel(nn.Module):
    """(T, 52) 시퀀스 → 마지막 스텝의 hidden → 거북목 로짓 1개."""

    def __init__(self, input_dim: int = FEATURE_DIM, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        out, _ = self.gru(x)          # out: (B, T, hidden)
        return self.head(out[:, -1])  # 마지막 스텝만 사용
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest test_train_torch.py -q`
Expected: `13 passed`

---

### Task 5: 학습 루프 + 교차검증 + 리포트·그래프 (`main()` 1차)

**Files:**
- Modify: `tools/train_torch.py`

**Interfaces:**
- Consumes: Task 1~4 전부, `train_model.rule_based_predict(df, delta_turtle)`, `train_model.compute_metrics(y_true, y_pred)`
- Produces: `train_model_with_early_stopping(model, X_tr, y_tr, X_val, y_val, ...) -> dict(history)`, `predict_proba(model, X) -> np.ndarray`, `metrics_with_f1(y_true, y_pred) -> dict`, `main()`. 실행 결과물 `dataset/torch_report.md`, `dataset/torch_report/{mlp_loss,gru_loss,comparison,confusion}.png`

- [ ] **Step 1: 학습·예측·지표 함수 추가**

`tools/train_torch.py` 상단 import에 추가:

```python
import json

import matplotlib
matplotlib.use("Agg")            # 창 없이 파일로만 저장
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from train_model import compute_metrics, rule_based_predict  # noqa: E402  (같은 tools/ 폴더)
```

경로 상수 (`SEQ_LEN` 아래):

```python
_CSV_PATH = os.path.join(_ROOT, "dataset", "posture_v2.csv")
_REPORT_PATH = os.path.join(_ROOT, "dataset", "torch_report.md")
_PLOT_DIR = os.path.join(_ROOT, "dataset", "torch_report")
_MLP_ONNX = os.path.join(_ROOT, "dataset", "mlp_model.onnx")
_GRU_ONNX = os.path.join(_ROOT, "dataset", "gru_model.onnx")
_NORM_PATH = os.path.join(_ROOT, "dataset", "feature_norm.json")
_SEED = 42
```

모델 섹션 아래:

```python
# ── 5. 학습 루프 ─────────────────────────────────────────────────────────────

def train_model_with_early_stopping(model: nn.Module, X_tr, y_tr, X_val, y_val,
                                    max_epochs: int = 100, patience: int = 10,
                                    lr: float = 1e-3, batch_size: int = 64) -> dict:
    """검증 손실이 patience epoch 동안 안 줄면 멈추고, 가장 좋았던 가중치를 복원한다.
    Returns: {"train_loss": [...], "val_loss": [...], "best_epoch": int}"""
    torch.manual_seed(_SEED)
    X_tr_t, y_tr_t = torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    X_val_t, y_val_t = torch.tensor(X_val), torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
    best_val, best_state, epochs_without_improvement = float("inf"), None, 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t))
        epoch_loss = 0.0
        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(X_tr_t[idx]), y_tr_t[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history["train_loss"].append(epoch_loss / len(perm))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val, epochs_without_improvement = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return history


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """거북목 확률 (N,). 판정은 50% 기준 — 규칙에 맞추려 조정하지 않는다."""
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).squeeze(1).numpy()


def metrics_with_f1(y_true, y_pred) -> dict:
    m = compute_metrics(y_true, y_pred)
    m["f1"] = f1_score(y_true, y_pred, zero_division=0)
    return m
```

- [ ] **Step 2: 데이터 로드 + 교차검증 함수 추가**

```python
# ── 6. 데이터 로드 + 교차검증 ────────────────────────────────────────────────

def load_dataset() -> pd.DataFrame:
    """세션·시간순 정렬 + 어깨 너비 필터. make_sequences() 가 정렬을 전제로 하므로 필수."""
    df = pd.read_csv(_CSV_PATH)
    df = df.sort_values(["session_id", "timestamp"]).reset_index(drop=True)
    return filter_valid_rows(df)


def _fit_and_eval(model_cls, X, y, groups, train_idx, test_idx):
    """한 폴드: train 안에서 val 분리 → 표준화(train 기준) → 학습 → test 예측.
    Returns: (y_pred (test 순서), history)"""
    tr_idx, val_idx = split_train_val(train_idx, groups)
    mean, std = fit_standardizer(X[tr_idx])
    model = model_cls()
    history = train_model_with_early_stopping(
        model,
        apply_standardizer(X[tr_idx], mean, std), y[tr_idx],
        apply_standardizer(X[val_idx], mean, std), y[val_idx],
    )
    proba = predict_proba(model, apply_standardizer(X[test_idx], mean, std))
    return (proba >= 0.5).astype(int), history


def cross_validate(df: pd.DataFrame, delta_turtle: float) -> dict:
    """세션 단위 Leave-One-Out. 폴드마다 MLP / GRU / 규칙 기반 지표를 계산한다.
    Returns: {"folds": [ {session, mlp, gru, rule_frame, rule_window}, ... ],
              "histories": {"mlp": [...], "gru": [...]}}"""
    X_frame = normalize_landmarks(df)
    y_frame = df["label"].to_numpy()
    sessions = df["session_id"].astype(str).to_numpy()
    X_seq, y_seq, seq_groups, last_idx = make_sequences(X_frame, y_frame, sessions)

    results = {"folds": [], "histories": {"mlp": [], "gru": []}}
    for train_idx, test_idx in leave_one_group_out(sessions):
        session = sessions[test_idx][0]
        # MLP: 프레임 단위
        mlp_pred, mlp_hist = _fit_and_eval(MlpModel, X_frame, y_frame, sessions, train_idx, test_idx)
        # GRU: 윈도우 단위 — 같은 세션의 윈도우를 test 로
        seq_train = np.where(seq_groups != session)[0]
        seq_test = np.where(seq_groups == session)[0]
        gru_pred, gru_hist = _fit_and_eval(GruModel, X_seq, y_seq, seq_groups, seq_train, seq_test)
        # 규칙 기반: 프레임 단위(MLP 비교용) + 윈도우 마지막 프레임 단위(GRU 비교용)
        rule_frame = rule_based_predict(df.iloc[test_idx], delta_turtle).to_numpy()
        rule_window = rule_based_predict(df.iloc[last_idx[seq_test]], delta_turtle).to_numpy()

        results["folds"].append({
            "session": session,
            "mlp": metrics_with_f1(y_frame[test_idx], mlp_pred),
            "gru": metrics_with_f1(y_seq[seq_test], gru_pred),
            "rule_frame": metrics_with_f1(y_frame[test_idx], rule_frame),
            "rule_window": metrics_with_f1(y_seq[seq_test], rule_window),
        })
        results["histories"]["mlp"].append(mlp_hist)
        results["histories"]["gru"].append(gru_hist)
        print(f"[fold {session}] mlp_acc={results['folds'][-1]['mlp']['accuracy']:.3f} "
              f"gru_acc={results['folds'][-1]['gru']['accuracy']:.3f} "
              f"rule_acc={results['folds'][-1]['rule_frame']['accuracy']:.3f}")
    return results
```

- [ ] **Step 3: 리포트·그래프 함수 추가**

```python
# ── 7. 리포트 + 그래프 ───────────────────────────────────────────────────────

_METRIC_KEYS = ["accuracy", "precision", "recall", "f1"]


def _mean_metrics(folds: list, key: str) -> dict:
    return {m: float(np.mean([f[key][m] for f in folds])) for m in _METRIC_KEYS}


def _metrics_row(name: str, m: dict) -> str:
    return f"| {name} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"


def format_report(df: pd.DataFrame, results: dict, onnx_check: "dict | None") -> str:
    folds = results["folds"]
    lines = [
        "# 랜드마크 기반 PyTorch 모델 — 3자 비교 리포트", "",
        f"- 데이터: {len(df)}행, 세션 {df['session_id'].nunique()}개, "
        f"사람 {df['person_id'].nunique()}명, 라벨 분포 {df['label'].value_counts().to_dict()}",
        f"- 검증: 세션 단위 Leave-One-Out ({len(folds)}폴드). 표준화는 train 폴드 기준.",
        f"- 입력: 어깨 정규화 랜드마크 {FEATURE_DIM}차원. GRU 윈도우 {SEQ_LEN}스텝.", "",
        "## 폴드별", "",
    ]
    header = "| 모델 | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|"
    for f in folds:
        lines += [f"### test 세션 = {f['session']}", "", header,
                  _metrics_row("MLP (프레임)", f["mlp"]),
                  _metrics_row("규칙 기반 (프레임)", f["rule_frame"]),
                  _metrics_row("GRU (윈도우)", f["gru"]),
                  _metrics_row("규칙 기반 (윈도우 마지막 프레임)", f["rule_window"]), ""]
    lines += ["## 평균", "", header,
              _metrics_row("MLP (프레임)", _mean_metrics(folds, "mlp")),
              _metrics_row("규칙 기반 (프레임)", _mean_metrics(folds, "rule_frame")),
              _metrics_row("GRU (윈도우)", _mean_metrics(folds, "gru")),
              _metrics_row("규칙 기반 (윈도우)", _mean_metrics(folds, "rule_window")), ""]
    lines += ["## 혼동행렬 (행=실제, 열=예측, [0,1] 순서) — 폴드 합산", ""]
    for key, name in [("mlp", "MLP"), ("gru", "GRU"), ("rule_frame", "규칙 기반")]:
        total = np.sum([f[key]["confusion_matrix"] for f in folds], axis=0).tolist()
        lines.append(f"- {name}: {total}")
    lines += ["", "## ONNX 내보내기", ""]
    if onnx_check is None:
        lines.append("- (아직 실행 안 됨)")
    else:
        for name, diff in onnx_check.items():
            status = "통과" if diff <= 1e-5 else "실패"
            lines.append(f"- {name}: PyTorch 대비 최대 확률 차이 {diff:.2e} → {status}")
    lines += ["", "그래프: `dataset/torch_report/` (mlp_loss, gru_loss, comparison, confusion)", ""]
    return "\n".join(lines)


def save_plots(results: dict) -> None:
    os.makedirs(_PLOT_DIR, exist_ok=True)
    folds = results["folds"]

    # 학습 곡선 — 폴드별 train/val loss
    for key in ("mlp", "gru"):
        plt.figure(figsize=(6, 4))
        for i, h in enumerate(results["histories"][key]):
            plt.plot(h["train_loss"], label=f"fold{i} train")
            plt.plot(h["val_loss"], "--", label=f"fold{i} val")
        plt.title(f"{key.upper()} loss"); plt.xlabel("epoch"); plt.ylabel("BCE"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(_PLOT_DIR, f"{key}_loss.png")); plt.close()

    # 3자 비교 막대
    names = ["MLP", "GRU", "Rule"]
    means = [_mean_metrics(folds, k) for k in ("mlp", "gru", "rule_frame")]
    x = np.arange(len(_METRIC_KEYS)); width = 0.25
    plt.figure(figsize=(7, 4))
    for i, (name, m) in enumerate(zip(names, means)):
        plt.bar(x + (i - 1) * width, [m[k] for k in _METRIC_KEYS], width, label=name)
    plt.xticks(x, _METRIC_KEYS); plt.ylim(0, 1.05); plt.legend(); plt.title("MLP vs GRU vs Rule (mean over folds)")
    plt.tight_layout(); plt.savefig(os.path.join(_PLOT_DIR, "comparison.png")); plt.close()

    # 혼동행렬 히트맵 3개
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, key, name in zip(axes, ("mlp", "gru", "rule_frame"), names):
        cm = np.sum([f[key]["confusion_matrix"] for f in folds], axis=0)
        ax.imshow(cm, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, int(cm[r, c]), ha="center", va="center")
        ax.set_title(name); ax.set_xlabel("pred"); ax.set_ylabel("true")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    plt.tight_layout(); plt.savefig(os.path.join(_PLOT_DIR, "confusion.png")); plt.close()
```

- [ ] **Step 4: `main()` 1차 (ONNX 제외) 추가**

```python
# ── 8. 실행 ──────────────────────────────────────────────────────────────────

def main() -> None:
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    with open(os.path.join(_ROOT, "config.json"), encoding="utf-8") as f:
        delta_turtle = json.load(f)["delta_turtle"]

    df = load_dataset()
    print(f"데이터 {len(df)}행, 세션 {df['session_id'].nunique()}개")
    if df["session_id"].nunique() < 2:
        print("세션이 2개 이상 있어야 Leave-One-Out 검증이 가능합니다.")
        return

    results = cross_validate(df, delta_turtle)
    save_plots(results)

    onnx_check = None   # Task 6 에서 채운다
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(format_report(df, results, onnx_check))
    print(f"리포트 저장: {_REPORT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 기존 테스트가 여전히 통과하는지 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest -q`
Expected: `24 passed` (기존 11 + 신규 13). matplotlib import 경고 없이.

- [ ] **Step 6: 실제 실행 (첫 학습)**

Run: `cd C:\Users\taeok\desktop\turtleCheck && .venv\Scripts\python.exe tools\train_torch.py`
Expected: 폴드 2개의 `[fold ...] mlp_acc=... gru_acc=... rule_acc=...` 출력, 1분 이내 종료, `dataset/torch_report.md` + `dataset/torch_report/` 안에 PNG 4개 생성. 리포트를 열어 표가 채워졌는지 확인.

**Save 기준 2** — "모델·학습 루프 + 로컬 실행으로 리포트·그래프 생성". 사용자에게 저장 시점임을 알린다. 리포트 결과 수치를 사용자에게 보여주고 해석을 함께 논의한다.

---

### Task 6: 최종 재학습 + ONNX 내보내기 + 동일성 확인

**Files:**
- Modify: `tools/train_torch.py`
- Modify: `tools/test_train_torch.py`

**Interfaces:**
- Produces: `export_onnx(model, example_input: np.ndarray, path: str) -> None`, `verify_onnx(model, path, X: np.ndarray) -> float` (최대 확률 차이), `train_final_and_export(df) -> dict` (`{"mlp": diff, "gru": diff}`). 결과물 `dataset/mlp_model.onnx`, `dataset/gru_model.onnx`, `dataset/feature_norm.json`

- [ ] **Step 1: 실패하는 테스트 추가**

import에 `export_onnx, verify_onnx` 추가, 파일 끝에:

```python
def test_onnx_export_matches_pytorch(tmp_path):
    """내보낸 ONNX 가 PyTorch 와 같은 확률을 내야 앱(onnxruntime)에서 학습 결과가 재현된다."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for model, X in [
        (MlpModel(), rng.normal(size=(5, FEATURE_DIM)).astype(np.float32)),
        (GruModel(), rng.normal(size=(5, SEQ_LEN, FEATURE_DIM)).astype(np.float32)),
    ]:
        path = str(tmp_path / f"{type(model).__name__}.onnx")
        export_onnx(model, X[:1], path)
        assert verify_onnx(model, path, X) <= 1e-5
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest test_train_torch.py -q`
Expected: `ImportError: cannot import name 'export_onnx'`

- [ ] **Step 3: 구현 추가**

상단 import에 `import onnxruntime as ort` 추가. 리포트 섹션 아래(실행 섹션 위):

```python
# ── 9. 최종 재학습 + ONNX ────────────────────────────────────────────────────

def export_onnx(model: nn.Module, example_input: np.ndarray, path: str) -> None:
    """입력 이름 "input", 출력 이름 "logit", 배치 차원은 동적."""
    model.eval()
    torch.onnx.export(
        model, torch.tensor(example_input), path,
        input_names=["input"], output_names=["logit"],
        dynamic_axes={"input": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def verify_onnx(model: nn.Module, path: str, X: np.ndarray) -> float:
    """같은 입력에 대한 PyTorch 확률과 onnxruntime 확률의 최대 차이를 돌려준다."""
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    logit = session.run(["logit"], {"input": X})[0]
    onnx_proba = 1 / (1 + np.exp(-logit.squeeze(1)))
    return float(np.max(np.abs(onnx_proba - predict_proba(model, X))))


def train_final_and_export(df: pd.DataFrame) -> dict:
    """교차검증이 끝난 뒤 전체 데이터로 다시 학습해 내보낸다 (v1 train_model.py 와 같은 원칙:
    검증은 held-out 으로 공정하게, 배포 모델은 데이터를 아끼지 않는다).
    Returns: {"mlp": 최대 확률 차이, "gru": 최대 확률 차이}"""
    X_frame = normalize_landmarks(df)
    y_frame = df["label"].to_numpy()
    sessions = df["session_id"].astype(str).to_numpy()
    X_seq, y_seq, seq_groups, _ = make_sequences(X_frame, y_frame, sessions)

    # 표준화는 전체 프레임 기준 하나만 만들어 MLP·GRU 가 공유한다 (앱에서도 하나만 적용하면 됨)
    mean, std = fit_standardizer(X_frame)
    with open(_NORM_PATH, "w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(),
                   "seq_len": SEQ_LEN, "landmark_count": LANDMARK_COUNT}, f, indent=2)

    checks = {}
    for name, model_cls, X, y, groups, path in [
        ("mlp", MlpModel, X_frame, y_frame, sessions, _MLP_ONNX),
        ("gru", GruModel, X_seq, y_seq, seq_groups, _GRU_ONNX),
    ]:
        all_idx = np.arange(len(y))
        tr_idx, val_idx = split_train_val(all_idx, groups)
        Xz = apply_standardizer(X, mean, std)
        model = model_cls()
        train_model_with_early_stopping(model, Xz[tr_idx], y[tr_idx], Xz[val_idx], y[val_idx])
        export_onnx(model, Xz[:1], path)
        checks[name] = verify_onnx(model, path, Xz[:64])
        print(f"[{name}] ONNX 저장: {path} (최대 확률 차이 {checks[name]:.2e})")
    return checks
```

`main()`의 `onnx_check = None   # Task 6 에서 채운다` 줄을 다음으로 교체:

```python
    onnx_check = train_final_and_export(df)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck\tools && ..\.venv\Scripts\python.exe -m pytest -q`
Expected: `25 passed`. `torch.onnx.export`가 `dynamo` 인자를 모른다는 `TypeError`가 나면(오래된 torch) 해당 인자만 제거한다.

- [ ] **Step 5: 전체 실행**

Run: `cd C:\Users\taeok\desktop\turtleCheck && .venv\Scripts\python.exe tools\train_torch.py`
Expected: 폴드 출력 → `[mlp] ONNX 저장 ... (최대 확률 차이 ...e-0x)` → `[gru] ...` → 리포트 저장. `dataset/`에 `mlp_model.onnx`, `gru_model.onnx`, `feature_norm.json` 생성. `torch_report.md`의 "ONNX 내보내기" 항목이 둘 다 "통과".

**Save 기준 3** — "ONNX 내보내기 + 동일성 확인 통과". 사용자에게 저장 시점임을 알린다.

---

### Task 7: 문서 갱신

**Files:**
- Modify: `AGENTS.md` (검증 도구 표)
- Modify: `DEVELOP.md` (개발 단계 현황 표, 규칙 기반 판정 검증 섹션 뒤에 신규 섹션)
- Modify: `.gitignore` 확인 — `dataset/torch_report/` PNG는 발표 자료이므로 **커밋 대상**. 제외하지 않는다.

- [ ] **Step 1: `AGENTS.md` 검증 도구 표에 행 추가**

`| \`tools/test_collect_labels.py\`, \`tools/test_detector.py\` | ... |` 행 아래에:

```
| `tools/train_torch.py` | 원시 랜드마크 → 어깨 정규화 52차원 → MLP·GRU 학습, 세션 단위 LOSO 로 규칙 기반과 3자 비교, ONNX 내보내기 (`dataset/torch_report.md`, `*.onnx`, `feature_norm.json`) |
| `tools/test_train_torch.py` | 피처·시퀀스·분할·표준화·모델 shape·ONNX 동일성 pytest |
```

- [ ] **Step 2: `DEVELOP.md` 개발 단계 현황 표에 행 추가**

`| 앱 내 AI 보조 지표 | ✅ 완료 | ... |` 행 아래에:

```
| 랜드마크 기반 PyTorch 모델 | ✅ 완료 | `tools/train_torch.py` — MLP·GRU vs 규칙 기반 3자 비교, ONNX 내보내기 (앱 통합은 Phase 3) |
```

- [ ] **Step 3: `DEVELOP.md`에 새 섹션 추가**

"### 데이터 품질 — 라벨 노이즈" 섹션 **바로 뒤**, "## 다음 과제" **앞**에 삽입. `<...>` 자리는 실제 `torch_report.md` 수치로 채운다:

```markdown
### 랜드마크 기반 PyTorch 모델 (Phase 2)

8월 검증에서 로지스틱 회귀가 규칙 기반보다 못했던 건 "AI가 못해서"가 아니라, 사람이 설계한
피처 4개를 선형 결합만 할 수 있어 규칙의 비선형 게이트(`z_gate`)를 재현하지 못했기 때문이었다.
Phase 2는 피처 설계를 사람이 하지 않고 **원시 랜드마크(상체 13개)에서 모델이 스스로 특징을
배우게** 한다. 설계: [docs/superpowers/specs/2026-09-22-torch-model-design.md](docs/superpowers/specs/2026-09-22-torch-model-design.md)

```bash
pip install -r tools/requirements.txt      # torch(CPU)·onnx·onnxruntime·matplotlib
python tools/train_torch.py                # 수십 초, CPU 로 충분 (1천 행대 × 파라미터 수천 개)
```

| 항목 | 내용 |
|---|---|
| 입력 | 랜드마크 13개를 어깨 중심 원점·어깨 너비 1 로 정규화 → (x,y,z) 39 + visibility 13 = 52차원. **캘리브레이션 불필요** |
| MLP | 단일 프레임 52 → 32 → 16 → 1 (≈2.2K 파라미터) |
| GRU | 최근 10스텝(2초) 시퀀스 → GRU(32) → 16 → 1 (≈9K 파라미터) |
| 학습 | BCE, Adam 1e-3, early stopping(patience 10), seed 42 |
| 검증 | 세션 단위 Leave-One-Out. 표준화는 train 폴드 기준 |
| 출력 | `dataset/torch_report.md`, `torch_report/*.png`, `mlp_model.onnx`, `gru_model.onnx`, `feature_norm.json` |

**결과** (`dataset/torch_report.md`, 세션 2개 LOSO 평균):

| | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| MLP (프레임) | <mlp_acc> | <mlp_prec> | <mlp_rec> | <mlp_f1> |
| GRU (윈도우) | <gru_acc> | <gru_prec> | <gru_rec> | <gru_f1> |
| 규칙 기반 | <rule_acc> | <rule_prec> | <rule_rec> | <rule_f1> |

<결과를 보고 사람이 쓰는 해석 2~4문장: 규칙 대비 우열, GRU 가 MLP 보다 나은지, 전환 구간에서의 차이,
1명·2세션이라는 한계>

**앱 통합 시 주의**: 앱은 `feature_norm.json`의 mean/std 로 **똑같이** 표준화한 뒤 ONNX 에 넣어야
한다. 이걸 빠뜨리면 학습 땐 잘 되고 앱에선 엉망이 된다. `onnxruntime`만 추가하면 되고 torch 는
배포하지 않는다 (Phase 3).
```

- [ ] **Step 4: 문서 렌더링 확인**

Run: `cd C:\Users\taeok\desktop\turtleCheck && grep -n "train_torch\|Phase 2" AGENTS.md DEVELOP.md`
Expected: AGENTS.md 2행, DEVELOP.md에 현황 표 행 + 새 섹션 제목이 보임. `<...>` 자리표시자가 남아 있지 않은지 `grep -n "<mlp_\|<gru_\|<rule_\|<결과" DEVELOP.md` → 출력 없음.

**Save 기준 4** — 문서 갱신 완료. Phase 2 종료. 사용자에게 저장 시점임을 알린다.

---

## Self-Review 결과

- **Spec 커버리지**: 피처(T1) · 시퀀스(T2) · LOSO/표준화(T3) · 모델(T4) · 학습/규칙 비교/리포트/그래프(T5) · 최종 재학습/ONNX/동일성/feature_norm(T6) · 문서(T7). 누락 없음.
- **자리표시자**: T7 Step 3의 `<...>`는 실행 결과로 채우는 자리이며 Step 4에서 남아 있지 않은지 검사한다.
- **이름 일관성**: `leave_one_group_out`, `split_train_val`, `fit_standardizer`/`apply_standardizer`, `make_sequences`(4개 반환), `train_model_with_early_stopping`, `predict_proba`, `metrics_with_f1`, `export_onnx`, `verify_onnx`, `train_final_and_export` — 정의와 호출이 일치.
