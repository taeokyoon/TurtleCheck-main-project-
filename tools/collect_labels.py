"""
collect_labels.py — 거북목 판정 검증·학습용 라벨링 데이터 수집 도구 (v2)

실행: 프로젝트 루트에서 `python tools/collect_labels.py --person <이름> [--camera 0]`

사용법:
  1. 카메라 앞에 정상 자세로 앉아 기다리면 자동으로 캘리브레이션된다.
  2. 스크립트 시작~종료가 하나의 세션(session_id)이다. 세션 안에서 라벨만 바꾼다.
     n 키: 현재 라벨 = 정상(0)
     t 키: 현재 라벨 = 거북목(1)
     p 키: 일시정지 (라벨 없음 → 기록 안 함). 자세를 고쳐 앉을 때 눌러 라벨 노이즈를 막는다.
     q 키: 종료
  3. 라벨이 설정된 동안 0.2초마다 한 행씩 dataset/posture_v2.csv 에 저장된다.

저장 내용 (행당 64컬럼):
  - 메타: session_id, person_id, timestamp(유닉스 초), label
  - 규칙 기반 비교용 피처 4개 + 캘리브레이션 baseline 4개 (v1과 동일)
  - 상체 랜드마크 13개(0~12번: 코·눈·귀·입·어깨) × (x, y, z, visibility) = 52개

배포 앱(turtleCheck.py)과 무관한 개발 전용 스크립트다. src/detector.py의
PostureDetector를 그대로 재사용해, 판정 로직과 100% 동일한 방식으로 피처를 계산한다.
"""
import argparse
import csv
import json
import os
import sys
import time

import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from src.detector import PostureDetector, PostureFeatures  # noqa: E402

_CSV_PATH = os.path.join(_ROOT, "dataset", "posture_v2.csv")
_RECORD_INTERVAL = 0.2        # 초당 5행 기록 (시계열 학습용 해상도)
_CALIBRATION_SECONDS = 3.0
_UPPER_BODY_LANDMARK_COUNT = 13   # MediaPipe Pose 0~12번: 코, 눈(6), 귀(2), 입(2), 어깨(2)

_META_COLUMNS = ["session_id", "person_id", "timestamp", "label"]
_FEATURE_COLUMNS = list(PostureFeatures._fields)
_BASELINE_COLUMNS = [f"baseline_{name}" for name in _FEATURE_COLUMNS]
_LABEL_NAMES = {None: "일시정지", 0: "정상(0)", 1: "거북목(1)"}


# ── 순수 로직 (pytest 대상) ───────────────────────────────────────────────────

def csv_columns() -> list[str]:
    """CSV 헤더. 메타 → 피처 → baseline → 랜드마크(lm00_x, lm00_y, lm00_z, lm00_v, ...) 순."""
    landmark_columns = [
        f"lm{i:02d}_{axis}"
        for i in range(_UPPER_BODY_LANDMARK_COUNT)
        for axis in ("x", "y", "z", "v")
    ]
    return _META_COLUMNS + _FEATURE_COLUMNS + _BASELINE_COLUMNS + landmark_columns


def build_row(session_id: str, person_id: str, timestamp: float, label: int,
              features: PostureFeatures, baseline: PostureFeatures, landmarks) -> list:
    """csv_columns() 와 같은 순서로 한 행을 만든다. landmarks 는 MediaPipe 랜드마크 리스트(33개)."""
    landmark_values = []
    for lm in landmarks[:_UPPER_BODY_LANDMARK_COUNT]:
        landmark_values += [lm.x, lm.y, lm.z, lm.visibility]
    return [session_id, person_id, timestamp, label] + list(features) + list(baseline) + landmark_values


# ── 카메라 I/O ───────────────────────────────────────────────────────────────

def _load_config() -> dict:
    with open(os.path.join(_ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _ensure_csv() -> None:
    os.makedirs(os.path.dirname(_CSV_PATH), exist_ok=True)
    if not os.path.exists(_CSV_PATH):
        with open(_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(csv_columns())


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


def _draw_hud(frame, person_id: str, elapsed: float, label: "int | None", row_count: int):
    """화면 왼쪽 위에 현재 상태를 표시한다. (OpenCV 는 한글 폰트가 없어 영문으로 표기)"""
    label_text = {None: "PAUSED", 0: "NORMAL(0)", 1: "TURTLE(1)"}[label]
    color = {None: (200, 200, 200), 0: (0, 255, 0), 1: (0, 0, 255)}[label]
    text = f"[{person_id}] {elapsed:5.0f}s | {label_text} | rows={row_count}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="거북목 라벨링 데이터 수집 (v2)")
    parser.add_argument("--person", required=True, help="촬영 대상 식별자 (예: taeok)")
    parser.add_argument("--camera", type=int, default=0, help="카메라 인덱스 (기본 0)")
    args = parser.parse_args()

    _ensure_csv()
    cfg = _load_config()
    detector = PostureDetector(cfg["delta_turtle"], cfg["delta_ok"])
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    try:
        if not _calibrate(detector, cap):
            return
        baseline = detector.baseline_features
        assert baseline is not None  # calibrate() 성공 시 항상 설정됨

        # 이 실행 전체가 하나의 세션 — 라벨은 세션 안에서 자유롭게 바뀐다
        session_id = time.strftime("%Y%m%d%H%M%S")
        session_start = time.time()
        current_label: "int | None" = None   # 시작은 일시정지 상태
        row_count = 0
        last_write = 0.0
        print(f"세션 {session_id} 시작. n=정상, t=거북목, p=일시정지, q=종료")

        with open(_CSV_PATH, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            while True:
                ok, frame = cap.read()
                if not ok:
                    continue
                features, landmarks = detector.process_frame_with_landmarks(frame)

                display = frame.copy()
                _draw_hud(display, args.person, time.time() - session_start, current_label, row_count)
                cv2.imshow("collect_labels", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("n"):
                    current_label = 0
                elif key == ord("t"):
                    current_label = 1
                elif key == ord("p"):
                    current_label = None
                if key in (ord("n"), ord("t"), ord("p")):
                    print(f"라벨 → {_LABEL_NAMES[current_label]}")

                # 라벨 설정됨 + 피처 유효(손 가림 등 아님) + 기록 주기 도달 → 1행 저장
                now = time.time()
                if (current_label is not None and features is not None
                        and now - last_write >= _RECORD_INTERVAL):
                    last_write = now
                    writer.writerow(build_row(
                        session_id, args.person, now, current_label, features, baseline, landmarks,
                    ))
                    csv_file.flush()
                    row_count += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print(f"세션 종료. 저장 위치: {_CSV_PATH}")


if __name__ == "__main__":
    main()
