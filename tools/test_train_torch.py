import numpy as np
import pandas as pd
import torch

# train_torch 가 import 시 프로젝트 루트를 sys.path 에 넣으므로 먼저 import 한다
from train_torch import (  # noqa: I001
    FEATURE_DIM,
    LANDMARK_COUNT,
    SEQ_LEN,
    GruModel,
    MlpModel,
    apply_standardizer,
    export_onnx,
    filter_valid_rows,
    fit_standardizer,
    leave_one_group_out,
    make_sequences,
    normalize_landmarks,
    split_train_val,
    verify_onnx,
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
