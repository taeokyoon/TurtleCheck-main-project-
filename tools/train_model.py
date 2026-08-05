"""
train_model.py — 라벨링 데이터로 로지스틱 회귀를 학습시키고, 규칙 기반 판정과
비교하는 검증 리포트를 생성한다.

실행: 프로젝트 루트에서 `python tools/train_model.py`

학습된 모델은 배포되지 않는다. 규칙 기반 판정(src/detector.py)이 실제 라벨과
얼마나 일치하는지를 검증하는 근거 자료(dataset/validation_report.md)를 만드는 것이 목적이다.
"""
import json
import os
import pickle
import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.detector import PostureFeatures, _combine  # noqa: E402

_FEATURE_COLS = ["y_score", "z_forward", "head_tilt", "ear_z_offset"]
_CSV_PATH = os.path.join(_ROOT, "dataset", "posture_labels.csv")
_REPORT_PATH = os.path.join(_ROOT, "dataset", "validation_report.md")
_MODEL_PATH = os.path.join(_ROOT, "dataset", "model.pkl")
_WEIGHTS_PATH = os.path.join(_ROOT, "dataset", "model_weights.json")


def split_by_session(df: pd.DataFrame, test_frac: float = 0.2) -> "tuple[pd.DataFrame, pd.DataFrame]":
    """session_id 단위로 train/test를 나눈다. 행 단위 무작위 분리는 하지 않는다 —
    같은 세션의 행들은 거의 동일해서 무작위 분리 시 데이터 누수가 발생한다.

    라벨(클래스)별로 나눠서 holdout한다 — 세션 수가 적을 때 전체를 뭉쳐서 나누면
    test에 한 클래스만 남는 경우가 생기기 때문이다. 클래스의 세션이 1개뿐이면
    holdout하지 않고 train에 전부 남긴다(test로 빼면 그 클래스가 train에서 사라짐)."""
    test_sessions = set()
    for _, group in df.groupby("label"):
        session_ids = sorted(group["session_id"].unique())
        if len(session_ids) < 2:
            continue
        n_test = max(1, round(len(session_ids) * test_frac))
        test_sessions.update(session_ids[-n_test:])
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
        "앱에 배포되는 AI 보조 지표는 이 모델과 동일한 가중치를, 판단 기준 확률 50%(로지스틱 "
        "회귀의 표준 기본값) 그대로 사용한다 — 규칙 기반과 더 잘 맞도록 별도로 보정하지 않는다. "
        "그래야 앱에서 보이는 'AI 일치/불일치' 표시가 규칙 기반 판정에 대한 독립적인 참고 의견으로 "
        "남는다.\n"
    )


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

    eval_model = LogisticRegression()
    eval_model.fit(X_train, y_train)
    model_metrics = compute_metrics(y_test, eval_model.predict(X_test))

    rule_pred = rule_based_predict(test_df, delta_turtle)
    rule_metrics = compute_metrics(y_test, rule_pred)

    report = format_report(model_metrics, rule_metrics, len(train_df), len(test_df))
    with open(_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    # 정확도 리포트는 위 eval_model(train 세션만)로 공정하게 측정하지만,
    # 실제 앱에 배포하는 모델은 가진 데이터를 전부(train+test) 써서 따로 학습한다
    # — 검증은 held-out으로 하되, 배포 모델은 데이터를 아끼지 않는다.
    final_model = LogisticRegression()
    final_model.fit(deviation_features(df), df["label"])
    with open(_MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)

    # scikit-learn 전체를 배포 앱에 넣지 않기 위해, 학습된 가중치 4개+절편만
    # 순수 숫자로 뽑아서 저장한다. 앱에서는 이 숫자로 직접 시그모이드를 계산한다
    # (src/ai_advisor.py 참고). 판단 기준 확률은 항상 50% — 규칙 기반과 맞도록 따로
    # 보정하지 않는다(독립적인 참고 의견으로 남겨두기 위함).
    weights = dict(zip(_FEATURE_COLS, final_model.coef_[0].tolist()))
    with open(_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"weights": weights, "intercept": float(final_model.intercept_[0])},
            f, ensure_ascii=False, indent=2,
        )

    print(report)
    print(f"리포트 저장: {_REPORT_PATH}")
    print(f"가중치 저장(전체 {len(df)}행으로 학습): {_WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
