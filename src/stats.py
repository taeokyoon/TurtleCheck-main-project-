"""
stats.py — 로컬 JSONL + Firestore 기반 자세 통계 집계
"""
import json
import logging
import os
from datetime import datetime, timedelta

log = logging.getLogger(__name__)


def _parse_jsonl(log_path: str, date_prefix: str) -> dict:
    total = turtle = count = 0
    if not os.path.exists(log_path):
        return {"total_seconds": 0, "turtle_seconds": 0, "count": 0}
    try:
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("timestamp", "").startswith(date_prefix):
                        total  += rec.get("total_seconds", 0)
                        turtle += rec.get("turtle_seconds", 0)
                        count  += 1
                except json.JSONDecodeError:
                    continue
    except OSError as e:
        log.warning("JSONL 읽기 실패: %s", e)
    return {"total_seconds": total, "turtle_seconds": turtle, "count": count}


def get_today_local(log_path: str) -> dict:
    return _parse_jsonl(log_path, datetime.now().strftime("%Y-%m-%d"))


def get_week_local(log_path: str) -> dict:
    today = datetime.now()
    total = turtle = count = 0
    for i in range(7):
        r = _parse_jsonl(log_path, (today - timedelta(days=i)).strftime("%Y-%m-%d"))
        total  += r["total_seconds"]
        turtle += r["turtle_seconds"]
        count  += r["count"]
    return {"total_seconds": total, "turtle_seconds": turtle, "count": count}


def format_stats(email: str, today: dict, week: dict, cloud: dict | None = None) -> str:
    """로컬 JSONL 기반 통계 포맷 (Firebase 조회 실패 시 폴백)."""
    def _ratio(t, total):
        return f"{t / total * 100:.1f}%" if total > 0 else "—"

    lines = [
        f"계정: {email}",
        "",
        "[ 오늘 (로컬) ]",
        f"  측정: {today['total_seconds'] // 60}분 ({today['count']}건)",
        f"  거북목 비율: {_ratio(today['turtle_seconds'], today['total_seconds'])}",
        f"  거북목 시간: {today['turtle_seconds'] // 60}분",
        "",
        "[ 최근 7일 (로컬) ]",
        f"  측정: {week['total_seconds'] // 60}분",
        f"  거북목 비율: {_ratio(week['turtle_seconds'], week['total_seconds'])}",
        f"  거북목 시간: {week['turtle_seconds'] // 60}분",
    ]

    if cloud:
        c_total  = cloud.get("total_tracked_seconds", 0)
        c_turtle = cloud.get("total_turtle_seconds", 0)
        lines += [
            "",
            "[ 오늘 (클라우드) ]",
            f"  측정: {c_total // 60}분",
            f"  거북목 비율: {_ratio(c_turtle, c_total)}",
            f"  거북목 시간: {c_turtle // 60}분",
        ]

    return "\n".join(lines)


def format_firebase_stats(email: str, fb: dict) -> str:
    """Firebase Firestore 누적 통계 포맷."""
    def _ratio(t, total):
        return f"{t / total * 100:.1f}%" if total > 0 else "—"

    def _section(label, d):
        t = d.get("total_seconds", 0)
        u = d.get("turtle_seconds", 0)
        return [
            f"[ {label} ]",
            f"  측정: {t // 60}분",
            f"  거북목 비율: {_ratio(u, t)}",
            f"  거북목 시간: {u // 60}분",
        ]

    days = fb.get("days", 30)
    lines = [f"계정: {email}  (Firebase 누적)", ""]
    lines += _section("오늘", fb.get("today", {}))
    lines += [""]
    lines += _section("최근 7일", fb.get("week", {}))
    lines += [""]
    lines += _section(f"최근 {days}일 누적", fb.get("total", {}))
    return "\n".join(lines)
