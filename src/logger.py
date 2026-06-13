"""
logger.py — JSON Lines 형식으로 분 단위 자세 기록 저장
"""
import json
import logging
import os
from datetime import datetime

log = logging.getLogger(__name__)


class PostureLogger:
    """
    초 단위 판정 결과를 누적하다가 flush() 호출 시
    한 줄의 JSON Lines 레코드로 posture_log.jsonl 에 append.

    user_dir : 로그를 저장할 디렉터리
               비로그인 → logs/anonymous/
               로그인   → logs/{uid}/
    """

    def __init__(self, user_dir: str):
        os.makedirs(user_dir, exist_ok=True)
        self.log_path    = os.path.join(user_dir, "posture_log.jsonl")
        self.turtle_secs = 0
        self.total_secs  = 0

    def tick(self, is_turtle: bool) -> None:
        self.total_secs += 1
        if is_turtle:
            self.turtle_secs += 1

    def flush(self) -> bool:
        return self.flush_with_record() is not None

    def flush_with_record(self) -> dict | None:
        """
        누적 데이터를 파일에 기록하고 카운터 초기화.
        저장된 레코드 dict 반환, 데이터 없거나 실패 시 None.
        """
        if self.total_secs == 0:
            return None

        record = {
            "timestamp":      datetime.now().isoformat(timespec="seconds"),
            "status":         1 if self.turtle_secs > self.total_secs / 2 else 0,
            "turtle_seconds": self.turtle_secs,
            "total_seconds":  self.total_secs,
        }
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.turtle_secs = 0
            self.total_secs  = 0
            return record
        except Exception as e:
            log.error("로그 기록 실패: %s", e)
            return None
