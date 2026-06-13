"""
auth.py — Firebase Auth REST API 기반 사용자 인증 + 세션 관리

비로그인 모드가 기본: api_key 미설정 또는 네트워크 오류 시 로그인 없이 계속 동작.
"""
import json
import logging
import os
import time
from datetime import datetime

import requests

log = logging.getLogger(__name__)

_SIGN_IN_IDP_URL   = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp"
_SIGN_IN_EMAIL_URL = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword"
_REFRESH_URL       = "https://securetoken.googleapis.com/v1/token"
class AuthManager:
    """
    이메일/비밀번호 로그인 + 로컬 세션 파일 기반 재시작 후 세션 복원.
    모든 메서드는 예외를 외부로 던지지 않는다 (호출자 부담 최소화).
    """

    def __init__(self, session_path: str, api_key: str):
        self.session_path = session_path
        self.api_key      = api_key
        self._uid:   str | None = None
        self._email: str | None = None
        self.last_error: str | None = None
        self._id_token: str | None = None
        self._refresh_token: str | None = None
        self._token_expires_at: float = 0

    # ── 세션 유지 ──────────────────────────────────────────────────────────────

    def load_session(self) -> bool:
        if not os.path.exists(self.session_path):
            return False
        try:
            with open(self.session_path, encoding="utf-8") as f:
                data = json.load(f)
            uid = data.get("uid")
            if not uid:
                return False
            self._uid   = uid
            self._email = data.get("email")
            self._id_token = data.get("id_token")
            self._refresh_token = data.get("refresh_token")
            self._token_expires_at = data.get("token_expires_at", 0)
            log.info("세션 복원: %s (%s)", self._email, self._uid)
            return True
        except Exception as e:
            log.warning("세션 파일 읽기 실패: %s", e)
            return False

    def save_session(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.session_path), exist_ok=True)
            data = {
                "uid":          self._uid,
                "email":        self._email,
                "id_token":     self._id_token,
                "refresh_token": self._refresh_token,
                "token_expires_at": self._token_expires_at,
                "logged_in_at": datetime.now().isoformat(),
            }
            with open(self.session_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning("세션 저장 실패: %s", e)

    def _clear_session(self) -> None:
        self._uid   = None
        self._email = None
        self._id_token = None
        self._refresh_token = None
        self._token_expires_at = 0
        if os.path.exists(self.session_path):
            try:
                os.remove(self.session_path)
            except Exception:
                pass

    # ── 인증 ──────────────────────────────────────────────────────────────────
        
    def login_with_google(self, client_secret_path: str = "client_secret.json") -> str | None:
        if not self.api_key:
            self.last_error = ".env 파일에 FIREBASE_API_KEY 가 설정되지 않았습니다."
            log.warning(self.last_error)
            return None

        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError:
            self.last_error = "구글 로그인 라이브러리가 설치되지 않았습니다. pip install google-auth-oauthlib 를 실행해주세요."
            return None

        # exe 실행 시 client_secret.json 을 번들 내부(_MEIPASS)에서 찾음
        import sys as _sys
        if not os.path.isabs(client_secret_path) and not os.path.exists(client_secret_path):
            if getattr(_sys, "frozen", False):
                client_secret_path = os.path.join(_sys._MEIPASS, client_secret_path)

        if not os.path.exists(client_secret_path):
            self.last_error = f"{client_secret_path} 파일이 앱 폴더에 없습니다!"
            return None

        try:
            flow = InstalledAppFlow.from_client_secrets_file(
                client_secret_path,
                scopes=['openid', 'https://www.googleapis.com/auth/userinfo.email']
            )
            credentials = flow.run_local_server(port=0, prompt='consent')
            google_id_token = credentials.id_token

            if not google_id_token:
                self.last_error = "구글 ID 토큰을 받지 못했습니다."
                return None

            payload = {
                "postBody": f"id_token={google_id_token}&providerId=google.com",
                "requestUri": "http://localhost",
                "returnIdpCredential": True,
                "returnSecureToken": True
            }

            resp = requests.post(
                f"{_SIGN_IN_IDP_URL}?key={self.api_key}",
                json=payload,
                timeout=15
            )
            resp.raise_for_status()
            body = resp.json()

            self._uid = body["localId"]
            self._email = body["email"]
            self._id_token = body.get("idToken")
            self._refresh_token = body.get("refreshToken")
            self._token_expires_at = time.time() + int(body.get("expiresIn", 3600)) - 300

            self.save_session()
            log.info("구글 로그인 성공: %s (%s)", self._email, self._uid)
            return self._uid

        except requests.exceptions.HTTPError as e:
            self.last_error = self._extract_firebase_error(e)
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def login_with_email(self, email: str, password: str) -> str | None:
        if not self.api_key:
            self.last_error = ".env 파일에 FIREBASE_API_KEY 가 설정되지 않았습니다."
            return None
        try:
            resp = requests.post(
                f"{_SIGN_IN_EMAIL_URL}?key={self.api_key}",
                json={"email": email, "password": password, "returnSecureToken": True},
                timeout=15,
            )
            resp.raise_for_status()
            body = resp.json()
            self._uid              = body["localId"]
            self._email            = body["email"]
            self._id_token         = body.get("idToken")
            self._refresh_token    = body.get("refreshToken")
            self._token_expires_at = time.time() + int(body.get("expiresIn", 3600)) - 300
            self.save_session()
            log.info("이메일 로그인 성공: %s (%s)", self._email, self._uid)
            return self._uid
        except requests.exceptions.HTTPError as e:
            self.last_error = self._extract_firebase_error(e)
            return None
        except Exception as e:
            self.last_error = str(e)
            return None

    def logout(self) -> None:
        email = self._email
        self._clear_session()
        log.info("로그아웃: %s", email)

    # ── 상태 조회 ─────────────────────────────────────────────────────────────

    def get_uid(self) -> str | None:
        return self._uid

    def get_email(self) -> str | None:
        return self._email

    def is_logged_in(self) -> bool:
        return self._uid is not None
    
    
    def get_valid_token(self) -> str | None:
        """현재 유효한 ID 토큰을 반환. 만료되었으면 자동으로 갱신합니다."""
        if not self.is_logged_in() or not self._refresh_token:
            return None
            
        if time.time() > self._token_expires_at:
            try:
                resp = requests.post(
                    f"{_REFRESH_URL}?key={self.api_key}",
                    data={"grant_type": "refresh_token", "refresh_token": self._refresh_token},
                    timeout=10
                )
                resp.raise_for_status()
                data = resp.json()
                self._id_token = data.get("id_token")
                self._refresh_token = data.get("refresh_token")
                self._token_expires_at = time.time() + int(data.get("expires_in", 3600)) - 300
                self.save_session()
                log.info("보안 토큰 자동 갱신 완료")
            except Exception as e:
                log.error("보안 토큰 갱신 실패: %s", e)
                return None
                
        return self._id_token

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_firebase_error(e: requests.exceptions.HTTPError) -> str:
        try:
            return e.response.json()["error"]["message"]
        except Exception:
            return str(e)
