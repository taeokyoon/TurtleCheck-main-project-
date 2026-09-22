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
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")            # 창 없이 파일로만 저장
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score
from torch import nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from train_model import compute_metrics, rule_based_predict  # noqa: E402  (같은 tools/ 폴더)

LANDMARK_COUNT = 13          # MediaPipe Pose 0~12번 (코·눈·귀·입·어깨)
FEATURE_DIM = LANDMARK_COUNT * 3 + LANDMARK_COUNT   # xyz 39 + visibility 13 = 52
_LEFT_SHOULDER, _RIGHT_SHOULDER = 11, 12
_MIN_SHOULDER_WIDTH = 0.01   # 이보다 좁으면 측면 촬영으로 보고 제외
SEQ_LEN = 10                 # GRU 윈도우 길이 (5Hz × 2초)
_SEED = 42

_CSV_PATH = os.path.join(_ROOT, "dataset", "posture_v2.csv")
_REPORT_PATH = os.path.join(_ROOT, "dataset", "torch_report.md")
_PLOT_DIR = os.path.join(_ROOT, "dataset", "torch_report")
_MLP_ONNX = os.path.join(_ROOT, "dataset", "mlp_model.onnx")
_GRU_ONNX = os.path.join(_ROOT, "dataset", "gru_model.onnx")
_NORM_PATH = os.path.join(_ROOT, "dataset", "feature_norm.json")


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


# ── 2. 시퀀스 생성 (GRU 용) ──────────────────────────────────────────────────

def make_sequences(features: np.ndarray, labels: np.ndarray, sessions: np.ndarray,
                   seq_len: int = SEQ_LEN):
    """같은 세션·같은 라벨이 이어지는 구간 안에서만 seq_len 길이 윈도우를 1칸씩 밀며 자른다.

    라벨이 바뀌는 지점을 걸치는 윈도우는 라벨을 하나로 정할 수 없으므로 만들지 않는다.
    입력은 세션·시간순으로 정렬돼 있다고 가정한다 (load_dataset 이 보장).
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
        # 학습: 미니배치로 한 바퀴
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

        # 검증 손실 측정
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        history["val_loss"].append(val_loss)

        # early stopping: 최저 검증 손실 시점의 가중치를 기억해 둔다
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


# ── 9. 최종 재학습 + ONNX ────────────────────────────────────────────────────

def export_onnx(model: nn.Module, example_input: np.ndarray, path: str) -> None:
    """입력 이름 "input", 출력 이름 "logit", 배치 차원은 동적.
    dynamo=False: torch 2.x 기본값(dynamo 익스포터)은 onnxscript 등 추가 의존이 필요하고
    GRU 에서 불안정할 수 있어, 검증된 TorchScript 경로를 명시적으로 쓴다."""
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

    onnx_check = train_final_and_export(df)
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(format_report(df, results, onnx_check))
    print(f"리포트 저장: {_REPORT_PATH}")


if __name__ == "__main__":
    main()
