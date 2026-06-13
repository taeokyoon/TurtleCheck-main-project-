"""
upload_queue.py — 업로드 대기/재시도 큐 (JSONL 기반 영속 저장)

각 항목 구조:
    {"id": "<uuid>", "status": "pending|done|failed",
     "queued_at": "<ISO8601>", "record": {<posture record>}}

• pending  : 업로드 미완료 (초기값 + 재시도 대상)
• done     : 업로드 성공
• failed   : 업로드 실패 (retry_failed() 로 pending 으로 되돌릴 수 있음)
"""
import json
import logging
import os
import uuid
from datetime import datetime

log = logging.getLogger(__name__)


class UploadQueue:
    def __init__(self, queue_path: str):
        os.makedirs(os.path.dirname(queue_path), exist_ok=True)
        self.queue_path = queue_path

    # ── 쓰기 ──────────────────────────────────────────────────────────────────

    def enqueue(self, record: dict) -> None:
        entry = {
            "id":        str(uuid.uuid4()),
            "status":    "pending",
            "queued_at": datetime.now().isoformat(),
            "record":    record,
        }
        try:
            with open(self.queue_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            log.error("enqueue 실패: %s", e)

    # ── 읽기 ──────────────────────────────────────────────────────────────────

    def get_pending(self) -> list[dict]:
        return [e for e in self._read_all() if e.get("status") == "pending"]

    def get_all_records(self, hour_prefix: str | None = None) -> list[dict]:
        """done + pending 전체 레코드 반환. hour_prefix 지정 시 해당 시간대만."""
        entries = [
            e["record"] for e in self._read_all()
            if e.get("status") in ("done", "pending")
        ]
        if hour_prefix:
            ts_prefix = hour_prefix.replace("_", "T")
            entries = [r for r in entries if r.get("timestamp", "").startswith(ts_prefix)]
        return entries

    # ── 상태 업데이트 ─────────────────────────────────────────────────────────

    def mark_done(self, entry_ids: list[str]) -> None:
        self._update_status(set(entry_ids), "done")

    def mark_failed(self, entry_ids: list[str]) -> None:
        self._update_status(set(entry_ids), "failed")

    def retry_failed(self) -> None:
        if not os.path.exists(self.queue_path):
            return
        entries = self._read_all()
        changed = any(e.get("status") == "failed" for e in entries)
        if not changed:
            return
        for e in entries:
            if e.get("status") == "failed":
                e["status"] = "pending"
        self._write_all(entries)

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _read_all(self) -> list[dict]:
        if not os.path.exists(self.queue_path):
            return []
        entries = []
        with open(self.queue_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _write_all(self, entries: list[dict]) -> None:
        try:
            lines = [json.dumps(e, ensure_ascii=False) for e in entries]
            with open(self.queue_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        except Exception as e:
            log.error("파일 쓰기 실패: %s", e)

    def _update_status(self, id_set: set[str], new_status: str) -> None:
        entries = self._read_all()
        changed = False
        for e in entries:
            if e.get("id") in id_set:
                e["status"] = new_status
                changed = True
        if changed:
            self._write_all(entries)
