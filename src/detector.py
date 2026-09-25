"""
detector.py — MediaPipe Pose 기반 자세 지표 계산 및 거북목 판정
"""
import logging
import threading
import time
from collections import deque, namedtuple

import cv2
import mediapipe as mp

log = logging.getLogger(__name__)

_WINDOW_MAXLEN    = 200   # 슬라이딩 윈도우 최대 샘플 수
_MIN_VISIBILITY   = 0.5   # 랜드마크 신뢰도 하한
_MIN_SHOULDER_W   = 0.05  # 어깨 너비 최솟값 (측면 촬영 필터)
_EVAL_INTERVAL    = 1.0   # 판정 주기 (초)
_MIN_SCORES       = 5     # 판정에 필요한 최소 샘플 수
_Z_WEIGHT         = 0.3   # z축 보조 가중치
_Z_GATE_Y         = 0.15  # 이 이상 y가 변하면 z 기여를 점진적으로 억제
_EMA_ALPHA        = 0.3   # 프레임간 지수이동평균 계수 (작을수록 더 부드럽게, 반응은 느리게)

# 프레임별 개별 자세 지표. head_tilt/ear_z_offset은 판정에는 아직 안 쓰이지만
# 계속 계산·로깅해두면 추후 임계값 튜닝이나 디버깅에 참고할 수 있다.
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
    """자세 지표를 판정용 단일 스칼라로 합친다. y_score가 기본이고, z_forward는
    y가 baseline 근처(정상 범위)일 때만 보조적으로 더해진다."""
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
        self.last_avg_features: "PostureFeatures | None" = None  # AI 보조 지표(src/ai_advisor.py)가 읽어감
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

    def process_frame_with_landmarks(self, frame) -> "tuple[PostureFeatures | None, object]":
        """BGR 프레임 → (자세 피처 벡터, MediaPipe 랜드마크 33개 리스트).
        사람 미감지 시 (None, None). 랜드마크는 있지만 피처 계산이 거부된 경우
        (손 가림·측면 촬영 등)엔 (None, 랜드마크).
        판정에는 쓰이지 않고, 학습 데이터 수집 도구(tools/collect_labels.py)가 원시 랜드마크를 읽어간다."""
        with self._lock:
            result = self._pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                return None, None
            landmarks = result.pose_landmarks.landmark
            return self._calc_features(landmarks), landmarks

    def process_frame(self, frame) -> "PostureFeatures | None":
        """BGR 프레임을 받아 자세 피처 벡터 반환. 감지 실패 시 None."""
        features, _ = self.process_frame_with_landmarks(frame)
        return features

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
            avg = _average(self.feature_window)
            self.last_avg_features = avg

            if self.baseline_score is None:
                return True, False

            # baseline_score 와의 거리로 판정 (부호로 방향까지 판단).
            # delta_turtle(진입) > delta_ok(복귀) 여야 경계가 겹치지 않아 플래핑이 없다.
            deviation    = _combine(avg) - self.baseline_score
            enter_turtle = deviation < -self.delta_turtle
            exit_turtle  = deviation > -self.delta_ok

            prev = self.is_turtle
            if not self.is_turtle and enter_turtle:
                self.is_turtle = True
            elif self.is_turtle and exit_turtle:
                self.is_turtle = False

            log.debug(
                "avg=%s deviation=%.4f enter=%s exit=%s is_turtle=%s",
                tuple(round(v, 3) for v in avg), deviation, enter_turtle, exit_turtle, self.is_turtle,
            )

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
