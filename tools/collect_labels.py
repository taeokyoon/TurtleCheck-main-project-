"""
collect_labels.py — 거북목 판정 검증용 라벨링 데이터 수집 도구

실행: 프로젝트 루트에서 `python tools/collect_labels.py`

사용법:
  1. 카메라 앞에 정상 자세로 앉아 기다리면 자동으로 캘리브레이션된다.
  2. n 키: "정상" 라벨 기록 시작/종료 토글
  3. t 키: "거북목" 라벨 기록 시작/종료 토글
  4. q 키: 종료

배포 앱(turtleCheck.py)과 무관한 개발 전용 스크립트다. src/detector.py의
PostureDetector를 그대로 재사용해, 판정 로직과 100% 동일한 방식으로 피처를 계산한다.
"""
import csv
import json
import os
import sys
import time

import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.detector import PostureDetector  # noqa: E402

_CSV_PATH = os.path.join(_ROOT, "dataset", "posture_labels.csv")
_RECORD_INTERVAL = 1.0        # 초당 1행 기록
_CALIBRATION_SECONDS = 3.0

_CSV_COLUMNS = [
    "session_id", "y_score", "z_forward", "head_tilt", "ear_z_offset",
    "baseline_y_score", "baseline_z_forward", "baseline_head_tilt", "baseline_ear_z_offset",
    "label",
]


def _load_config() -> dict:
    with open(os.path.join(_ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _ensure_csv() -> None:
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)
    if not os.path.exists(_CSV_PATH):
        with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(_CSV_COLUMNS)


def _calibrate(detector: PostureDetector, cap: cv2.VideoCapture) -> bool:
    print(f"캘리브레이션 중... {_CALIBRATION_SECONDS:.0f}초간 정상 자세로 가만히 있어주세요.")
    start = time.time()
    while time.time() - start < _CALIBRATION_SECONDS:
        ok, frame = cap.read()
        if not ok:
            continue
        detector.update(detector.process_frame(frame))
        cv2.imshow("collect_labels", frame)
        cv2.waitKey(1)
    baseline = detector.calibrate()
    if baseline is None:
        print("캘리브레이션 실패 — 카메라에 얼굴/어깨가 잘 보이는지 확인 후 다시 실행하세요.")
        return False
    print(f"캘리브레이션 완료 (baseline_score={baseline:.4f})")
    return True


def main() -> None:
    _ensure_csv()
    cfg = _load_config()
    detector = PostureDetector(cfg["delta_turtle"], cfg["delta_ok"])
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    try:
        if not _calibrate(detector, cap):
            return
        baseline = detector.baseline_features
        assert baseline is not None  # calibrate() 성공 시 항상 설정됨

        print("n=정상 기록 토글, t=거북목 기록 토글, q=종료")
        recording_label: "int | None" = None
        session_id: "str | None" = None
        last_write = 0.0

        with open(_CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                features = detector.process_frame(frame)

                status = "대기 중"
                if recording_label is not None:
                    status = f"기록 중 (label={recording_label}, session={session_id})"
                display = frame.copy()
                cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("collect_labels", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("n"):
                    if recording_label == 0:
                        recording_label = None
                    elif recording_label is None:
                        recording_label = 0
                        session_id = time.strftime("%Y%m%d%H%M%S")
                elif key == ord("t"):
                    if recording_label == 1:
                        recording_label = None
                    elif recording_label is None:
                        recording_label = 1
                        session_id = time.strftime("%Y%m%d%H%M%S")

                now = time.time()
                if (recording_label is not None and features is not None
                        and now - last_write >= _RECORD_INTERVAL):
                    last_write = now
                    writer.writerow([
                        session_id,
                        features.y_score, features.z_forward, features.head_tilt, features.ear_z_offset,
                        baseline.y_score, baseline.z_forward, baseline.head_tilt, baseline.ear_z_offset,
                        recording_label,
                    ])
                    csv_file.flush()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
