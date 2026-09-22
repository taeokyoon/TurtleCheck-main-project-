import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detector import PostureDetector  # noqa: E402


@pytest.fixture(scope="module")
def detector():
    d = PostureDetector(delta_turtle=0.13, delta_ok=0.07)
    yield d
    d.close()


def _blank_frame():
    """사람이 없는 검은 BGR 프레임 — MediaPipe 가 랜드마크를 찾지 못해야 한다."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_process_frame_with_landmarks_returns_none_pair_when_no_person(detector):
    features, landmarks = detector.process_frame_with_landmarks(_blank_frame())
    assert features is None
    assert landmarks is None


def test_process_frame_still_returns_none_when_no_person(detector):
    """기존 process_frame() 계약이 그대로 유지되는지 (회귀 방지)."""
    assert detector.process_frame(_blank_frame()) is None
