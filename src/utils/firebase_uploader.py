"""
firebase_uploader.py — Firestore 업로드 (로그인 사용자 전용)

Firestore 경로:
    hour/{uid}/{YYYY-MM-DD}/{H~H+1}

uid 가 없으면 업로드를 건너뛴다 (비로그인 보호).
firebase_key.json 이 없거나 초기화 실패 시 graceful degradation.
"""
import json
import logging
import os
import sys
from datetime import datetime, timedelta

import requests

log = logging.getLogger(__name__)

class FirebaseUploader:
    def __init__(self, auth_manager, project_id: str):
        self.auth_manager = auth_manager
        # 클라우드 함수(서버) 접속 주소
        self.base_url = f"https://asia-northeast3-{project_id}.cloudfunctions.net"
        self._available = bool(project_id)

    def upload_log_file(self, file_path: str, uid: str | None) -> bool:
        # uid 대신 안전한 토큰을 가져옵니다.
        id_token = self.auth_manager.get_valid_token()
        if not self._available or not id_token or not os.path.exists(file_path):
            return False

        try:
            data = []
            total_tracked_seconds = 0
            total_turtle_seconds  = 0
            bad_posture_count     = 0

            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    entry = json.loads(line)
                    data.append(entry)
                    total_tracked_seconds += entry.get("total_seconds", 0)
                    total_turtle_seconds  += entry.get("turtle_seconds", 0)
                    bad_posture_count     += entry.get("status", 0)

            if not data or total_tracked_seconds == 0:
                return False

            base_name = os.path.basename(file_path)
            raw_name  = os.path.splitext(base_name)[0] 
            
            try:
                date_part, hour_part = raw_name.split('_')
                h_int = int(hour_part)
                hour_range = f"{h_int}~{h_int+1}"
            except ValueError:
                date_part = "unknown_date"
                hour_range = raw_name

            payload = {
                "date": date_part,
                "hour_range": hour_range,
                "total_tracked_seconds": total_tracked_seconds,
                "total_turtle_seconds": total_turtle_seconds,
                "bad_posture_count": bad_posture_count,
                "log_data": data
            }

            url = f"{self.base_url}/uploadLog"
            headers = {"Authorization": f"Bearer {id_token}"}
            
            response = requests.post(url, json=payload, headers=headers, timeout=15)

            if response.status_code == 200:
                log.info("업로드 완료: hour/%s/%s", date_part, hour_range)
                return True
            else:
                log.error("업로드 거부됨: %s", response.text)
                return False

        except Exception as e:
            log.error("업로드 실패: %s", e)
            return False

    def _init_admin(self) -> bool:
        """firebase-admin 초기화 (싱글톤). firebase_key.json 없으면 False."""
        try:
            import firebase_admin
            from firebase_admin import credentials

            if firebase_admin._apps:
                return True

            # 개발 환경 또는 exe 번들 내부에서 키 파일 탐색
            candidates = ["firebase_key.json"]
            if getattr(sys, "frozen", False):
                candidates.append(os.path.join(sys._MEIPASS, "firebase_key.json"))

            key_path = next((p for p in candidates if os.path.exists(p)), None)
            if not key_path:
                log.warning("firebase_key.json 없음 — Firestore 직접 조회 불가")
                return False

            firebase_admin.initialize_app(credentials.Certificate(key_path))
            return True
        except Exception as e:
            log.warning("firebase-admin 초기화 실패: %s", e)
            return False

    def get_firestore_cumulative_stats(self, uid: str, days: int = 30) -> dict | None:
        """Firestore에서 최근 N일 누적 통계 직접 조회."""
        if not self._init_admin():
            return None
        try:
            from firebase_admin import firestore
            db = firestore.client()
            user_ref = db.collection("hour").document(uid)

            today = datetime.now()
            result = {"today": {}, "week": {}, "total": {}}

            def _empty():
                return {"total_seconds": 0, "turtle_seconds": 0, "bad_count": 0}

            buckets = {"today": _empty(), "week": _empty(), "total": _empty()}

            for i in range(days):
                date = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                try:
                    docs = list(user_ref.collection(date).stream())
                except Exception:
                    continue

                for doc in docs:
                    d = doc.to_dict() or {}
                    t = d.get("total_tracked_seconds", 0)
                    u = d.get("total_turtle_seconds", 0)
                    b = d.get("bad_posture_count", 0)

                    buckets["total"]["total_seconds"]  += t
                    buckets["total"]["turtle_seconds"] += u
                    buckets["total"]["bad_count"]      += b

                    if i == 0:
                        buckets["today"]["total_seconds"]  += t
                        buckets["today"]["turtle_seconds"] += u
                        buckets["today"]["bad_count"]      += b

                    if i < 7:
                        buckets["week"]["total_seconds"]  += t
                        buckets["week"]["turtle_seconds"] += u
                        buckets["week"]["bad_count"]      += b

            buckets["days"] = days
            return buckets
        except Exception as e:
            log.error("Firestore 누적 통계 조회 실패: %s", e)
            return None

    def get_stats(self, uid: str) -> dict | None:
        id_token = self.auth_manager.get_valid_token()
        if not self._available or not id_token:
            return None
            
        try:
            today_str = datetime.now().strftime("%Y-%m-%d")
            url = f"{self.base_url}/getStats?date={today_str}"
            headers = {"Authorization": f"Bearer {id_token}"}
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            log.error("통계 데이터 조회 실패: %s", e)
            return None
