"""
detector.py — MediaPipe Pose 기반 자세 지표 계산 및 거북목 판정
"""
import threading
import time
from collections import deque, namedtuple

import cv2
import mediapipe as mp

_WINDOW_MAXLEN    = 200   # 슬라이딩 윈도우 최대 샘플 수
_MIN_VISIBILITY   = 0.5   # 랜드마크 신뢰도 하한
_MIN_SHOULDER_W   = 0.05  # 어깨 너비 최솟값 (측면 촬영 필터)
_EVAL_INTERVAL    = 1.0   # 판정 주기 (초)
_MIN_SCORES       = 5     # 판정에 필요한 최소 샘플 수
_Z_WEIGHT         = 0.3   # z축 보조 가중치
_Z_GATE_Y         = 0.15  # 이 이상 y가 변하면 z 기여를 점진적으로 억제
_EMA_ALPHA        = 0.3   # 프레임간 지수이동평균 계수 (작을수록 더 부드럽게, 반응은 느리게)

# 프레임별 개별 자세 지표. 하나의 점수로 미리 합치지 않고 각각 보존해야
# 추후 이상탐지 모델(OCSVM 등)이 지표 간 상관관계를 학습할 수 있다.
PostureFeatures = namedtuple("PostureFeatures", ["y_score", "z_forward", "head_tilt", "ear_z_offset"])


def _ema(prev: "PostureFeatures | None", cur: PostureFeatures) -> PostureFeatures:
    if prev is None:
        return cur
    return PostureFeatures(*(_EMA_ALPHA * c + (1 - _EMA_ALPHA) * p for c, p in zip(cur, prev)))


def _average(window) -> PostureFeatures:
    n = len(window)
    sums = [0.0, 0.0, 0.0, 0.0]
    for _, f in window:
        for i, v in enumerate(f):
            sums[i] += v
    return PostureFeatures(*(s / n for s in sums))


def _combine(f: PostureFeatures) -> float:
    """규칙 기반 판정(v1)이 쓰는 단일 스칼라. 기존 로직과 동일하게 y_score/z_forward만 반영."""
    z_gate = max(0.0, 1.0 - abs(f.y_score) / _Z_GATE_Y)
    return f.y_score + _Z_WEIGHT * f.z_forward * z_gate


class PostureDetector:
    """
    자세 피처 벡터 계산 + 슬라이딩 윈도우 평균 + 히스테리시스 판정.
    Lock으로 보호되어 복수 스레드에서 안전하게 사용 가능.
    """

    def __init__(self, delta_turtle: float, delta_ok: float):
        self.delta_turtle   = delta_turtle
        self.delta_ok       = delta_ok
        self.feature_window: deque = deque(maxlen=_WINDOW_MAXLEN)  # (timestamp, PostureFeatures)
        self.baseline_score: float | None = None
        self.baseline_features: "PostureFeatures | None" = None
        self.is_turtle: bool = False
        self._last_eval: float = time.time()
        self._ema_prev: "PostureFeatures | None" = None
        self._lock              = threading.Lock()

        self._mp_pose = mp.solutions.pose
        self._pose    = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=_MIN_VISIBILITY,
            min_tracking_confidence=_MIN_VISIBILITY,
        )

    # ── 프레임 처리 ───────────────────────────────────────────────────────────

    def _calc_features(self, lms) -> "PostureFeatures | None":
        NOSE = lms[self._mp_pose.PoseLandmark.NOSE.value]
        LS   = lms[self._mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        RS   = lms[self._mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        LE   = lms[self._mp_pose.PoseLandmark.LEFT_EAR.value]
        RE   = lms[self._mp_pose.PoseLandmark.RIGHT_EAR.value]
        if min(LS.visibility, RS.visibility, NOSE.visibility, LE.visibility, RE.visibility) <= _MIN_VISIBILITY:
            return None
        sw = abs(LS.x - RS.x)
        if sw <= _MIN_SHOULDER_W:
            return None
        # 손(손목·손가락)이 얼굴 근처에 있으면 NOSE 추적이 불안정해지므로 스킵
        _HAND_LMS = [
            self._mp_pose.PoseLandmark.LEFT_WRIST,
            self._mp_pose.PoseLandmark.RIGHT_WRIST,
            self._mp_pose.PoseLandmark.LEFT_INDEX,
            self._mp_pose.PoseLandmark.RIGHT_INDEX,
            self._mp_pose.PoseLandmark.LEFT_PINKY,
            self._mp_pose.PoseLandmark.RIGHT_PINKY,
            self._mp_pose.PoseLandmark.LEFT_THUMB,
            self._mp_pose.PoseLandmark.RIGHT_THUMB,
        ]
        for lm_id in _HAND_LMS:
            pt = lms[lm_id.value]
            if pt.visibility > _MIN_VISIBILITY:
                dist = ((NOSE.x - pt.x) ** 2 + (NOSE.y - pt.y) ** 2) ** 0.5
                if dist < 0.20:
                    return None

        # 어깨너비(sw)로 정규화해서 카메라와의 거리에 무관하게 만든다.
        y_score   = ((LS.y + RS.y) / 2 - NOSE.y) / sw
        z_forward = (NOSE.z - (LS.z + RS.z) / 2) / sw

        # head_tilt: 왼쪽귀와 오른쪽귀의 y값 차이. 고개가 옆으로 기울면
        # 한쪽 귀가 다른 쪽보다 아래로 내려가므로 이 차이가 커진다.
        head_tilt = (LE.y - RE.y) / sw

        # ear_z_offset: z_forward와 같은 구조 — "귀 평균 z" - "어깨 평균 z".
        # 절대 깊이값이 아니라 어깨 대비 상대적으로 얼마나 앞에 있는지를 봐야
        # 카메라 거리·사람마다의 차이에 흔들리지 않는다.
        ear_z_offset = ((LE.z + RE.z) / 2 - (LS.z + RS.z) / 2) / sw

        return PostureFeatures(y_score, z_forward, head_tilt, ear_z_offset)

    def process_frame(self, frame) -> "PostureFeatures | None":
        """BGR 프레임을 받아 자세 피처 벡터 반환. 감지 실패 시 None."""
        with self._lock:
            result = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                return None
            return self._calc_features(result.pose_landmarks.landmark)

    def process_frame_visual(self, frame) -> "tuple[PostureFeatures | None, object]":
        """BGR 프레임 처리 후 (features, rgb_annotated) 반환. 시작 창 시각화 전용."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self._lock:
            result = self._pose.process(rgb)
            if not result.pose_landmarks:
                return None, rgb
            mp.solutions.drawing_utils.draw_landmarks(
                rgb,
                result.pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
            )
            return self._calc_features(result.pose_landmarks.landmark), rgb

    # ── 상태 갱신 (1초마다 판정) ──────────────────────────────────────────────

    def update(self, features: "PostureFeatures | None") -> tuple[bool, bool]:
        """
        features 를 EMA로 스무딩한 뒤 슬라이딩 윈도우에 추가하고 _EVAL_INTERVAL 마다 판정.

        Returns:
            did_evaluate  (bool): 이번 호출에서 판정이 수행됐는지
            state_changed (bool): is_turtle 상태가 바뀌었는지
        """
        with self._lock:
            now = time.time()
            if features is not None:
                self._ema_prev = _ema(self._ema_prev, features)
                self.feature_window.append((now, self._ema_prev))

            while self.feature_window and now - self.feature_window[0][0] > _EVAL_INTERVAL:
                self.feature_window.popleft()

            if now - self._last_eval < _EVAL_INTERVAL or len(self.feature_window) < _MIN_SCORES:
                return False, False

            self._last_eval = now
            avg   = _average(self.feature_window)
            score = _combine(avg)

            if self.baseline_score is None:
                return True, False

            deviation = score - self.baseline_score
            prev      = self.is_turtle

            if not self.is_turtle and deviation < -self.delta_turtle:
                self.is_turtle = True
            elif self.is_turtle and deviation > -self.delta_ok:
                self.is_turtle = False

            return True, (self.is_turtle != prev)

    # ── 캘리브레이션 ──────────────────────────────────────────────────────────

    def calibrate(self) -> float | None:
        """현재 슬라이딩 윈도우 평균을 baseline 으로 설정. 데이터 부족 시 None."""
        with self._lock:
            if len(self.feature_window) < _MIN_SCORES:
                return None
            self.baseline_features = _average(self.feature_window)
            self.baseline_score    = _combine(self.baseline_features)
            return self.baseline_score

    def close(self) -> None:
        self._pose.close()
