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


def test_split_by_session_stratifies_across_classes():
    df = pd.DataFrame({
        "session_id": ["a", "b", "c", "d", "e"],
        "label": [0, 0, 0, 1, 1],
    })
    _, test_df = split_by_session(df, test_frac=1 / 3)

    test_sessions = set(test_df["session_id"])
    # 정상(0) 세션 3개 중 1개, 거북목(1) 세션 2개 중 1개가 각각 test로 빠져야 한다
    assert test_sessions == {"c", "e"}


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
