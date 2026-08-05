"""
ai_advisor.py — 학습된 로지스틱 회귀 가중치로 규칙 기반 판정을 보조하는
실시간 참고 지표를 계산한다.

이 모듈의 판단은 최종 판정에 관여하지 않는다 — 최종 판정 권한은 여전히
src/detector.py의 규칙 기반 판정(PostureDetector)에 있다. 가중치는
tools/train_model.py가 미리 학습해 dataset/model_weights.json으로 내보낸 것을
그대로 읽어 쓴다(앱에는 scikit-learn을 설치하지 않는다).
"""
import json
import math

from src.detector import PostureFeatures

_weights: "dict | None" = None
_intercept: float = 0.0


def load_weights(weights_path: str) -> bool:
    """가중치 파일을 로드한다. 파일이 없으면 False — 이후 predict_turtle()은 None만 반환한다."""
    global _weights, _intercept
    try:
        with open(weights_path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return False
    _weights = data["weights"]
    _intercept = data["intercept"]
    return True


def predict_turtle(features: PostureFeatures, baseline: PostureFeatures) -> "bool | None":
    """AI 모델이 지금 자세를 거북목으로 판단하는지 반환. 가중치 미로드 시 None.

    판단 기준 확률은 로지스틱 회귀의 표준 기본값인 50%를 그대로 쓴다 — 규칙 기반
    판정과 더 잘 맞도록 별도로 낮추지 않는다. 그래야 이 값이 규칙 기반에 대한
    독립적인 참고 의견으로 남는다."""
    if _weights is None:
        return None
    z = _intercept
    for name in PostureFeatures._fields:
        deviation = getattr(features, name) - getattr(baseline, name)
        z += _weights[name] * deviation
    probability = 1 / (1 + math.exp(-z))
    return probability >= 0.5
