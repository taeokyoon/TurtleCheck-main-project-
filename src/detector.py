"""
detector.py — MediaPipe Pose 기반 자세 점수 계산 및 거북목 판정
"""
import threading
import time
from collections import deque

import cv2
import mediapipe as mp

_WINDOW_MAXLEN    = 200   # 슬라이딩 윈도우 최대 샘플 수
_MIN_VISIBILITY   = 0.5   # 랜드마크 신뢰도 하한
_MIN_SHOULDER_W   = 0.05  # 어깨 너비 최솟값 (측면 촬영 필터)
_EVAL_INTERVAL    = 1.0   # 판정 주기 (초)
_MIN_SCORES       = 5     # 판정에 필요한 최소 샘플 수
_Z_WEIGHT         = 0.3   # z축 보조 가중치
_Z_GATE_Y         = 0.15  # 이 이상 y가 변하면 z 기여를 점진적으로 억제


class PostureDetector:
    """
    head_forward_score 계산 + 슬라이딩 윈도우 평균 + 히스테리시스 판정.
    Lock으로 보호되어 복수 스레드에서 안전하게 사용 가능.
    """

    def __init__(self, delta_turtle: float, delta_ok: float):
        self.delta_turtle   = delta_turtle
        self.delta_ok       = delta_ok
        self.scores: deque  = deque(maxlen=_WINDOW_MAXLEN)
        self.baseline_score: float | None = None
        self.is_turtle: bool = False
        self._last_eval: float = time.time()
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

    def _calc_score(self, lms) -> float | None:
        NOSE = lms[self._mp_pose.PoseLandmark.NOSE.value]
        LS   = lms[self._mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        RS   = lms[self._mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        if min(LS.visibility, RS.visibility, NOSE.visibility) <= _MIN_VISIBILITY:
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

        y_score   = (LS.y + RS.y) / 2 - NOSE.y
        # y 변화가 클수록(고개 숙임) z 기여를 줄여 오탐 방지
        # y 변화가 작을 때(진짜 앞으로만 내밀기)만 z 가 의미 있음
        z_forward = NOSE.z - (LS.z + RS.z) / 2
        y_magnitude = abs(y_score) / sw           # 정규화된 y 변화량
        z_gate = max(0.0, 1.0 - y_magnitude / _Z_GATE_Y)
        return (y_score + _Z_WEIGHT * z_forward * z_gate) / sw

    def process_frame(self, frame) -> float | None:
        """BGR 프레임을 받아 head_forward_score 반환. 감지 실패 시 None."""
        with self._lock:
            result = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                return None
            return self._calc_score(result.pose_landmarks.landmark)

    def process_frame_visual(self, frame) -> tuple[float | None, object]:
        """BGR 프레임 처리 후 (score, rgb_annotated) 반환. 시작 창 시각화 전용."""
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
            return self._calc_score(result.pose_landmarks.landmark), rgb

    # ── 상태 갱신 (1초마다 판정) ──────────────────────────────────────────────

    def update(self, score: float | None) -> tuple[bool, bool]:
        """
        score 를 슬라이딩 윈도우에 추가하고 _EVAL_INTERVAL 마다 판정.

        Returns:
            did_evaluate  (bool): 이번 호출에서 판정이 수행됐는지
            state_changed (bool): is_turtle 상태가 바뀌었는지
        """
        with self._lock:
            now = time.time()
            if score is not None:
                self.scores.append((now, score))

            while self.scores and now - self.scores[0][0] > _EVAL_INTERVAL:
                self.scores.popleft()

            if now - self._last_eval < _EVAL_INTERVAL or len(self.scores) < _MIN_SCORES:
                return False, False

            self._last_eval = now
            avg = sum(s for _, s in self.scores) / len(self.scores)

            if self.baseline_score is None:
                return True, False

            deviation = avg - self.baseline_score
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
            if len(self.scores) < _MIN_SCORES:
                return None
            self.baseline_score = sum(s for _, s in self.scores) / len(self.scores)
            return self.baseline_score

    def close(self) -> None:
        self._pose.close()
