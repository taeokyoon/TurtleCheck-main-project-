from collections import namedtuple

# collect_labels 가 import 시 프로젝트 루트를 sys.path 에 넣으므로 먼저 import 해야 src 를 찾는다
from collect_labels import (  # noqa: I001
    _UPPER_BODY_LANDMARK_COUNT,
    build_row,
    csv_columns,
)
from src.detector import PostureFeatures  # noqa: E402

# MediaPipe 랜드마크 객체를 흉내내는 최소 구조 (x, y, z, visibility 속성만 필요)
_FakeLandmark = namedtuple("_FakeLandmark", ["x", "y", "z", "visibility"])


def _fake_landmarks(count: int = 33) -> list:
    """i번째 랜드마크의 x,y,z,v가 i, i+0.1, i+0.2, i+0.3 이 되도록 만들어 순서 검증을 쉽게 한다."""
    return [_FakeLandmark(i, i + 0.1, i + 0.2, i + 0.3) for i in range(count)]


def test_csv_columns_has_metadata_features_baseline_then_landmarks():
    cols = csv_columns()
    # 메타 4 + 피처 4 + baseline 4 + 상체 13개 × 4 = 64
    assert len(cols) == 4 + 4 + 4 + _UPPER_BODY_LANDMARK_COUNT * 4
    assert cols[:4] == ["session_id", "person_id", "timestamp", "label"]
    assert cols[4:8] == ["y_score", "z_forward", "head_tilt", "ear_z_offset"]
    assert cols[8:12] == ["baseline_y_score", "baseline_z_forward", "baseline_head_tilt", "baseline_ear_z_offset"]
    assert cols[12:16] == ["lm00_x", "lm00_y", "lm00_z", "lm00_v"]
    assert cols[-4:] == ["lm12_x", "lm12_y", "lm12_z", "lm12_v"]


def test_build_row_matches_column_order():
    features = PostureFeatures(1.0, 2.0, 3.0, 4.0)
    baseline = PostureFeatures(0.1, 0.2, 0.3, 0.4)
    row = build_row("s1", "taeok", 1700000000.5, 1, features, baseline, _fake_landmarks())

    assert len(row) == len(csv_columns())
    assert row[:4] == ["s1", "taeok", 1700000000.5, 1]
    assert row[4:8] == [1.0, 2.0, 3.0, 4.0]
    assert row[8:12] == [0.1, 0.2, 0.3, 0.4]
    # 첫 랜드마크(0번)와 마지막 상체 랜드마크(12번)의 x,y,z,v 순서
    assert row[12:16] == [0, 0.1, 0.2, 0.3]
    assert row[-4:] == [12, 12.1, 12.2, 12.3]


def test_build_row_ignores_lower_body_landmarks():
    """33개 전부 넘겨도 상체 13개(0~12번)만 저장된다."""
    features = PostureFeatures(0, 0, 0, 0)
    row = build_row("s1", "p", 0.0, 0, features, features, _fake_landmarks(33))
    landmark_values = row[12:]
    assert len(landmark_values) == _UPPER_BODY_LANDMARK_COUNT * 4
    assert 13 not in landmark_values  # 13번(왼쪽 손목)의 x값이 들어가면 안 된다
