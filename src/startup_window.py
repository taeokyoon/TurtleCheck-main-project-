"""
startup_window.py — 앱 시작 창 + 트레이 로그인 창 + 설정 창

크로스플랫폼: customtkinter + CTkImage (Windows / macOS 공통)

StartupWindow  : 앱 시작 시 (카메라 피드 + 로그인 + 캘리브레이션)
SettingsWindow : 트레이 "설정 화면 열기" 클릭 시
AuthWindow     : 트레이 "로그인" 클릭 시 (컴팩트 폼)
"""
import platform
import queue
import threading
import tkinter as tk
import customtkinter as ctk

import cv2
from PIL import Image, ImageDraw

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── 디자인 토큰 ───────────────────────────────────────────────────────────────
_BG       = "#0a0a0a"   # 윈도우 배경
_BG_CAM   = "#050505"   # 카메라 패널 배경
_SURF     = "#111111"   # 카드·버튼 표면
_BORDER   = "#1e1e1e"   # 테두리
_ACCENT   = "#22c55e"   # 초록 액센트
_ACCENT_H = "#16a34a"   # 초록 호버
_TEXT_HI  = "#e5e7eb"   # 주 텍스트
_TEXT_MID = "#374151"   # 보조 텍스트
_TEXT_DIM = "#252525"   # 희미한 텍스트
_YELLOW   = "#facc15"   # 대기 상태

# StringVar.__del__ 스레드 안전 패치 ──────────────────────────────────────────
_orig_variable_del = tk.Variable.__del__
def _safe_variable_del(self):
    try:
        _orig_variable_del(self)
    except RuntimeError:
        pass
tk.Variable.__del__ = _safe_variable_del


# ── 공유 헬퍼 ─────────────────────────────────────────────────────────────────

def _cancel_all_after(root):
    try:
        for after_id in root.tk.call('after', 'info'):
            try:
                root.after_cancel(after_id)
            except Exception:
                pass
    except Exception:
        pass


def _make_google_icon(size: int = 18) -> Image.Image:
    """Google G 로고를 PIL Image로 생성 (4× 슈퍼샘플링 안티에일리어스)."""
    scale = 4
    s     = size * scale
    img   = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d     = ImageDraw.Draw(img)
    pad   = scale
    outer = [pad, pad, s - pad, s - pad]
    ir    = int(s * 0.30)
    c     = s // 2
    inner = [c - ir, c - ir, c + ir, c + ir]
    # 4색 파이 슬라이스 (PIL: 0°=동/오른쪽, 시계방향)
    d.pieslice(outer,  -30,  90, fill="#4285F4")   # 파랑: 오른쪽
    d.pieslice(outer,   90, 165, fill="#34A853")   # 초록: 아래오른쪽
    d.pieslice(outer,  165, 207, fill="#FBBC05")   # 노랑: 아래왼쪽
    d.pieslice(outer,  207, 330, fill="#EA4335")   # 빨강: 위
    d.ellipse(inner, fill=(0, 0, 0, 0))
    bh = int(s * 0.10)
    d.rectangle([c - pad, c - bh, s - pad, c + bh], fill="#4285F4")
    return img.resize((size, size), Image.LANCZOS)


_GOOGLE_ICON_IMG = _make_google_icon(18)

# 설정 창 싱글톤 — 동시에 두 개가 열리지 않도록 추적
_active_settings_window: "SettingsWindow | None" = None


def _load_mascot(parent_frame, mascot_path: str | None, size: int = 130) -> None:
    if not mascot_path:
        return
    try:
        img     = Image.open(mascot_path).resize((size, size), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(size, size))
        lbl     = ctk.CTkLabel(parent_frame, image=ctk_img, text="")
        lbl.image = ctk_img
        lbl.pack(pady=(6, 2))
    except Exception:
        pass


def _hsep(parent) -> None:
    """1px 수평 구분선."""
    tk.Frame(parent, height=1, bg="#141414").pack(fill="x")


# ── 공유: 인증 UI 빌더 ───────────────────────────────────────────────────────

def _build_auth_section(parent, google_cmd, email_cmd, logout_cmd) -> tuple:
    """login_frame, logged_frame, logged_lbl, email_entry, pw_entry 반환."""
    google_icon = ctk.CTkImage(
        light_image=_GOOGLE_ICON_IMG,
        dark_image=_GOOGLE_ICON_IMG,
        size=(16, 16),
    )
    wrapper = ctk.CTkFrame(parent, fg_color="transparent")
    wrapper.pack(fill="x", pady=2)

    login_frame = ctk.CTkFrame(wrapper, fg_color="transparent")

    _entry_kw = dict(
        width=190, height=28,
        fg_color=_SURF, border_color=_BORDER, border_width=1,
        corner_radius=8, text_color=_TEXT_HI,
        placeholder_text_color="#4b5563",
        font=ctk.CTkFont(size=10),
    )
    email_entry = ctk.CTkEntry(login_frame, placeholder_text="이메일", **_entry_kw)
    email_entry.pack(pady=(0, 3))

    pw_entry = ctk.CTkEntry(login_frame, placeholder_text="비밀번호", show="*", **_entry_kw)
    pw_entry.pack(pady=(0, 3))

    email_entry.bind("<Return>", lambda e: pw_entry.focus())
    pw_entry.bind("<Return>", lambda e: email_cmd())

    ctk.CTkButton(
        login_frame, text="로그인",
        width=190, height=34,
        fg_color=_SURF, border_color=_BORDER, border_width=1,
        corner_radius=10, text_color="#6b7280", hover_color="#161616",
        font=ctk.CTkFont(size=10),
        command=email_cmd,
    ).pack(pady=(0, 3))

    ctk.CTkButton(
        login_frame,
        text="구글 계정으로 로그인",
        image=google_icon,
        compound="left",
        anchor="w",
        width=190, height=34,
        fg_color=_SURF, border_color=_BORDER, border_width=1,
        corner_radius=10, text_color="#6b7280", hover_color="#161616",
        font=ctk.CTkFont(size=10),
        command=google_cmd,
    ).pack()

    logged_frame = ctk.CTkFrame(wrapper, fg_color="transparent")
    logged_lbl   = tk.StringVar()
    ctk.CTkLabel(
        logged_frame, textvariable=logged_lbl,
        text_color=_ACCENT, font=ctk.CTkFont(size=9),
    ).pack(pady=(2, 4))
    ctk.CTkButton(
        logged_frame, text="로그아웃", width=80,
        fg_color="transparent", border_color=_BORDER, border_width=1,
        text_color="#6b7280", hover_color=_SURF, corner_radius=8,
        font=ctk.CTkFont(size=9),
        command=logout_cmd,
    ).pack()

    return login_frame, logged_frame, logged_lbl, email_entry, pw_entry


def _refresh_auth_ui(auth_manager, login_frame, logged_frame, logged_lbl) -> None:
    """auth_manager 상태에 따라 비로그인·로그인 프레임을 전환."""
    if auth_manager.is_logged_in():
        login_frame.pack_forget()
        logged_frame.pack(fill="x")
        logged_lbl.set(f"✓  {auth_manager.get_email()}")
    else:
        logged_frame.pack_forget()
        login_frame.pack(fill="x")


# ── 공유: 캘리브레이션 베이스라인 카드 ──────────────────────────────────────────

def _build_baseline_card(parent) -> tuple:
    """(card_frame, val_label, progress_bar) 반환."""
    card = ctk.CTkFrame(
        parent, fg_color="#0d0d0d",
        border_color="#181818", border_width=1, corner_radius=8,
    )
    row = ctk.CTkFrame(card, fg_color="transparent")
    row.pack(fill="x", padx=10, pady=(7, 3))
    ctk.CTkLabel(row, text="BASELINE",
                 font=ctk.CTkFont(size=7), text_color="#6b7280").pack(side="left")
    val_lbl = ctk.CTkLabel(row, text="—",
                            font=ctk.CTkFont(size=7), text_color="#6b7280")
    val_lbl.pack(side="right")
    bar = ctk.CTkProgressBar(
        card, height=3, corner_radius=2,
        fg_color="#151515", progress_color=_ACCENT,
    )
    bar.set(0)
    bar.pack(fill="x", padx=10, pady=(0, 7))
    return card, val_lbl, bar


# ── StartupWindow ─────────────────────────────────────────────────────────────

class StartupWindow:
    """앱 시작 시 표시되는 창. 좌: 카메라 피드, 우: 로그인 + 캘리브레이션."""

    _FRAME_W = 520
    _FRAME_H = 440
    _PANEL_W = 224
    _POLL_MS = 33

    def __init__(self, detector, auth_manager, on_done, switch_logger,
                 mascot_path: str | None = None):
        self.detector      = detector
        self.auth_manager  = auth_manager
        self.on_done       = on_done
        self.switch_logger = switch_logger
        self.mascot_path   = mascot_path

        self._frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._stop_cam = threading.Event()
        self._photo    = None

    # ── 카메라 스레드 ──────────────────────────────────────────────────────────

    def _cam_thread(self):
        cap = cv2.VideoCapture(0)
        while not self._stop_cam.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            score, rgb = self.detector.process_frame_visual(frame)
            if score is not None:
                self.detector.update(score)
            img = Image.fromarray(rgb).resize(
                (self._FRAME_W, self._FRAME_H), Image.BILINEAR
            )
            try:
                self._frame_queue.put_nowait(img)
            except queue.Full:
                pass
        cap.release()

    # ── tkinter 프레임 폴링 ───────────────────────────────────────────────────

    def _poll_frame(self):
        try:
            img = self._frame_queue.get_nowait()
            self._photo = ctk.CTkImage(
                light_image=img, dark_image=img,
                size=(self._FRAME_W, self._FRAME_H),
            )
            self._cam_label.configure(image=self._photo)
        except queue.Empty:
            pass
        if not self._stop_cam.is_set() and self._root.winfo_exists():
            self._poll_id = self._root.after(self._POLL_MS, self._poll_frame)

    # ── 캘리브레이션 ──────────────────────────────────────────────────────────

    def _on_calibrate(self):
        baseline = self.detector.calibrate()
        if baseline is None:
            self._auth_msg.set("자세가 감지되지 않았습니다. 잠시 후 다시 시도하세요.")
            return
        self._baseline_val.configure(text=f"{baseline:.3f}", text_color=_ACCENT)
        self._baseline_bar.set(min(abs(baseline) / 0.8, 1.0))
        self._badge_dot.configure(fg_color=_ACCENT)
        self._badge_lbl.configure(text="ACTIVE", text_color=_ACCENT)
        self._auth_msg.set(f"완료 — 기준값 {baseline:.3f}  만족하면 아래 버튼을 눌러주세요.")

    def _on_continue(self):
        if self.detector.baseline_score is None:
            self._auth_msg.set("캘리브레이션이 설정되지 않았습니다. 캘리브레이션을 먼저 설정해주세요.")
            return
        self._finish()

    def _finish(self):
        self._auth_msg = None
        self._stop_cam.set()
        _cancel_all_after(self._root)
        self.on_done()
        self._root.withdraw()
        self._root.quit()

    # ── 로그인 / 로그아웃 ─────────────────────────────────────────────────────

    def _on_email_login(self):
        email = self._email_entry.get().strip()
        pw    = self._pw_entry.get()
        if not email or not pw:
            self._auth_msg.set("이메일과 비밀번호를 입력해주세요.")
            return
        self._auth_msg.set("로그인 중...")
        self._root.update()

        def _do():
            uid = self.auth_manager.login_with_email(email, pw)
            if self._stop_cam.is_set() or not self._root.winfo_exists():
                return
            if uid:
                self.switch_logger(uid)
                try:
                    self._root.after(0, lambda: self._auth_msg.set("로그인 완료"))
                    self._root.after(0, self._update_auth_ui)
                except Exception:
                    pass
            else:
                err = self.auth_manager.last_error or "알 수 없는 오류"
                try:
                    self._root.after(0, lambda: self._auth_msg.set(f"로그인 실패: {err}"))
                except Exception:
                    pass
        threading.Thread(target=_do, daemon=True).start()

    def _on_google_login(self):
        self._auth_msg.set("브라우저에서 구글 로그인을 진행해주세요...")
        self._root.update()

        def _do():
            uid = self.auth_manager.login_with_google()
            if self._stop_cam.is_set() or not self._root.winfo_exists():
                return
            if uid:
                self.switch_logger(uid)
                try:
                    self._root.after(0, lambda: self._auth_msg.set("로그인 완료"))
                    self._root.after(0, self._update_auth_ui)
                except Exception:
                    pass
            else:
                err = self.auth_manager.last_error or "알 수 없는 오류"
                try:
                    self._root.after(0, lambda: self._auth_msg.set(f"로그인 실패: {err}"))
                except Exception:
                    pass
        threading.Thread(target=_do, daemon=True).start()

    def _on_logout(self):
        self.auth_manager.logout()
        self.switch_logger(None)
        self._auth_msg.set("로그아웃되었습니다.")
        self._update_auth_ui()

    # ── 인증 상태 UI 전환 ─────────────────────────────────────────────────────

    def _update_auth_ui(self):
        _refresh_auth_ui(self.auth_manager, self._login_frame, self._logged_frame, self._logged_lbl)
        if hasattr(self, "_continue_btn"):
            if self.auth_manager.is_logged_in():
                self._continue_btn.configure(text="시작하기")
            else:
                self._continue_btn.configure(text="비로그인으로 계속")

    # ── UI 빌드 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._root = ctk.CTk()
        self._root.title("Turtle Check")
        self._root.resizable(False, False)
        self._root.configure(fg_color=_BG)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        if platform.system() == "Darwin":
            try:
                self._root.createcommand("tk::mac::Quit", self._on_close)
            except Exception:
                pass

        # ── 좌측: 카메라 패널 ─────────────────────────────────────────────
        left = ctk.CTkFrame(self._root, fg_color=_BG_CAM, corner_radius=0)
        left.pack(side="left", fill="y")

        cam_wrap = ctk.CTkFrame(
            left, fg_color="#0d0d0d",
            border_color="#1c1c1c", border_width=1, corner_radius=14,
            width=self._FRAME_W, height=self._FRAME_H,
        )
        cam_wrap.pack(padx=14, pady=(14, 10))
        cam_wrap.pack_propagate(False)

        self._cam_label = ctk.CTkLabel(
            cam_wrap, text="", width=self._FRAME_W, height=self._FRAME_H,
        )
        self._cam_label.place(relx=0.5, rely=0.5, anchor="center")

        # MEDIAPIPE ACTIVE 상태 필
        pill = ctk.CTkFrame(
            left, fg_color=_SURF, corner_radius=20,
            border_color=_BORDER, border_width=1,
        )
        pill.pack(pady=(0, 14))
        self._cam_dot = ctk.CTkFrame(
            pill, width=7, height=7, corner_radius=4, fg_color=_ACCENT,
        )
        self._cam_dot.pack_propagate(False)
        self._cam_dot.pack(side="left", padx=(10, 5), pady=7)
        ctk.CTkLabel(
            pill, text="MEDIAPIPE ACTIVE",
            font=ctk.CTkFont(size=9), text_color="#3d4d3d",
        ).pack(side="left", padx=(0, 10), pady=7)

        # 세로 구분선
        tk.Frame(self._root, width=1, bg="#141414").pack(side="left", fill="y")

        # ── 우측: 컨트롤 패널 ─────────────────────────────────────────────
        right = ctk.CTkFrame(self._root, fg_color=_BG, corner_radius=0, width=self._PANEL_W)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        inner = ctk.CTkFrame(right, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=18)

        # 앱 제목
        ctk.CTkLabel(
            inner, text="TURTLE CHECK",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_TEXT_HI,
        ).pack(anchor="w")
        ctk.CTkLabel(
            inner, text="POSTURE MONITOR",
            font=ctk.CTkFont(size=7), text_color="#6b7280",
        ).pack(anchor="w", pady=(1, 10))
        _hsep(inner)

        # 상태 뱃지 (STANDBY)
        badge_f = ctk.CTkFrame(
            inner, fg_color=_SURF, corner_radius=20,
            border_color=_BORDER, border_width=1,
        )
        badge_f.pack(anchor="w", pady=(8, 8))
        self._badge_dot = ctk.CTkFrame(badge_f, width=6, height=6, corner_radius=3, fg_color=_YELLOW)
        self._badge_dot.pack_propagate(False)
        self._badge_dot.pack(side="left", padx=(8, 4), pady=5)
        self._badge_lbl = ctk.CTkLabel(
            badge_f, text="STANDBY",
            font=ctk.CTkFont(size=8), text_color=_YELLOW,
        )
        self._badge_lbl.pack(side="left", padx=(0, 8), pady=5)

        # ACCOUNT 섹션
        ctk.CTkLabel(
            inner, text="ACCOUNT",
            font=ctk.CTkFont(size=7), text_color="#6b7280",
        ).pack(anchor="w", pady=(0, 3))

        self._auth_msg = tk.StringVar(value="")
        ctk.CTkLabel(
            inner, textvariable=self._auth_msg,
            font=ctk.CTkFont(size=8), text_color="#4b5563", wraplength=190,
        ).pack(anchor="w", pady=(0, 2))

        (self._login_frame, self._logged_frame, self._logged_lbl,
         self._email_entry, self._pw_entry) = _build_auth_section(
            inner, self._on_google_login, self._on_email_login, self._on_logout,
        )
        self._update_auth_ui()

        self._continue_btn = ctk.CTkButton(
            inner, text="비로그인으로 계속",
            fg_color=_SURF, border_color=_BORDER, border_width=1,
            text_color="#6b7280", hover_color="#161616",
            corner_radius=10, font=ctk.CTkFont(size=10),
            height=36, width=190,
            command=self._on_continue,
        )
        self._continue_btn.pack(anchor="w", pady=(2, 6))
        self._update_auth_ui()

        # CALIBRATION 섹션 구분선
        div = ctk.CTkFrame(inner, fg_color="transparent")
        div.pack(fill="x", pady=(0, 6))
        tk.Frame(div, height=1, bg="#141414").pack(side="left", fill="x", expand=True, pady=6)
        ctk.CTkLabel(
            div, text="  CALIBRATION  ",
            font=ctk.CTkFont(size=7), text_color="#6b7280",
        ).pack(side="left")
        tk.Frame(div, height=1, bg="#141414").pack(side="left", fill="x", expand=True, pady=6)

        ctk.CTkLabel(
            inner, text="바른 자세로 앉은 후\n버튼을 눌러주세요.",
            font=ctk.CTkFont(size=8), text_color="#9ca3af", justify="left",
        ).pack(anchor="w", pady=(0, 4))

        # 베이스라인 카드
        baseline_card, self._baseline_val, self._baseline_bar = _build_baseline_card(inner)
        baseline_card.pack(fill="x", pady=(0, 8))

        # 캘리브레이션 버튼
        ctk.CTkButton(
            inner, text="캘리브레이션 시작  [P]",
            font=ctk.CTkFont(size=13, weight="bold"),
            width=190, height=60,
            fg_color=_ACCENT, hover_color=_ACCENT_H,
            text_color="#0a0a0a", corner_radius=12,
            command=self._on_calibrate,
        ).pack(fill="x")

        self._root.bind("<p>", lambda e: self._on_calibrate())
        self._root.bind("<P>", lambda e: self._on_calibrate())

    def _on_close(self):
        self._stop_cam.set()
        _cancel_all_after(self._root)
        self._root.destroy()

    def run(self):
        self._build_ui()
        self._cam_ref = threading.Thread(target=self._cam_thread, daemon=True)
        self._cam_ref.start()
        self._poll_id = self._root.after(self._POLL_MS, self._poll_frame)
        self._root.mainloop()
        self._stop_cam.set()
        self._cam_ref.join(timeout=2.0)
        return self._root


# ── SettingsWindow ────────────────────────────────────────────────────────────

class SettingsWindow:
    """트레이 "설정 화면 열기" 클릭 시 표시되는 창."""

    _FRAME_W = 520
    _FRAME_H = 440
    _PANEL_W = 224
    _POLL_MS = 33

    def __init__(self, detector, auth_manager, live_frame_queue,
                 start_visual, stop_visual, switch_logger,
                 mascot_path: str | None = None,
                 on_auth_change=None,
                 parent=None):
        self.detector          = detector
        self.auth_manager      = auth_manager
        self._live_frame_queue = live_frame_queue
        self._start_visual     = start_visual
        self._stop_visual      = stop_visual
        self.switch_logger     = switch_logger
        self.mascot_path       = mascot_path
        self._on_auth_change   = on_auth_change
        self._parent           = parent
        self._root             = None
        self._photo            = None

    def show_in_main_thread(self):
        global _active_settings_window
        # 이미 열려 있으면 해당 창을 앞으로 가져오고 종료
        if (_active_settings_window is not None
                and _active_settings_window._root is not None):
            try:
                if _active_settings_window._root.winfo_exists():
                    _active_settings_window._root.lift()
                    _active_settings_window._root.focus_force()
                    return
            except Exception:
                pass
        _active_settings_window = self
        self._start_visual()
        try:
            self._build_ui()
        except Exception:
            self._stop_visual()
            _active_settings_window = None
            raise

    def _close(self):
        global _active_settings_window
        _active_settings_window = None
        self._stop_visual()
        self._auth_msg = None
        if self._root and self._root.winfo_exists():
            # 설정 창 자신의 _poll_id 만 취소 — 메인 _poll 을 건드리지 않는다
            if hasattr(self, "_poll_id"):
                try:
                    self._root.after_cancel(self._poll_id)
                except Exception:
                    pass
            try:
                self._root.destroy()
            except Exception:
                pass

    # ── 캘리브레이션 ──────────────────────────────────────────────────────────

    def _on_calibrate(self):
        baseline = self.detector.calibrate()
        if baseline is None:
            self._auth_msg.set("자세가 감지되지 않았습니다. 잠시 후 다시 시도하세요.")
            return
        self._baseline_val.configure(text=f"{baseline:.3f}", text_color=_ACCENT)
        self._baseline_bar.set(min(abs(baseline) / 0.8, 1.0))
        self._badge_dot.configure(fg_color=_ACCENT)
        self._badge_lbl.configure(text="ACTIVE", text_color=_ACCENT)
        self._auth_msg.set(f"완료 — 기준값 {baseline:.3f}")

    # ── 인증 ──────────────────────────────────────────────────────────────────

    def _on_email_login(self):
        email = self._email_entry.get().strip()
        pw    = self._pw_entry.get()
        if not email or not pw:
            self._auth_msg.set("이메일과 비밀번호를 입력해주세요.")
            return
        self._auth_msg.set("로그인 중...")
        if self._root:
            self._root.update()

        def _do():
            uid = self.auth_manager.login_with_email(email, pw)
            if uid:
                self.switch_logger(uid)
                if self._root:
                    self._root.after(0, lambda: self._auth_msg.set("로그인 완료"))
                    self._root.after(0, self._update_auth_ui)
                if self._on_auth_change:
                    self._on_auth_change()
            else:
                err = self.auth_manager.last_error or "알 수 없는 오류"
                if self._root:
                    self._root.after(0, lambda: self._auth_msg.set(f"로그인 실패: {err}"))
        threading.Thread(target=_do, daemon=True).start()

    def _on_google_login(self):
        self._auth_msg.set("브라우저에서 구글 로그인을 진행해주세요...")
        if self._root:
            self._root.update()

        def _do():
            uid = self.auth_manager.login_with_google()
            if uid:
                self.switch_logger(uid)
                if self._root:
                    self._root.after(0, lambda: self._auth_msg.set("로그인 완료"))
                    self._root.after(0, self._update_auth_ui)
                if self._on_auth_change:
                    self._on_auth_change()
            else:
                err = self.auth_manager.last_error or "알 수 없는 오류"
                if self._root:
                    self._root.after(0, lambda: self._auth_msg.set(f"로그인 실패: {err}"))
        threading.Thread(target=_do, daemon=True).start()

    def _on_logout(self):
        self.auth_manager.logout()
        self.switch_logger(None)
        self._auth_msg.set("로그아웃되었습니다.")
        self._update_auth_ui()
        if self._on_auth_change:
            self._on_auth_change()

    def _update_auth_ui(self):
        _refresh_auth_ui(self.auth_manager, self._login_frame, self._logged_frame, self._logged_lbl)

    # ── 프레임 폴링 ───────────────────────────────────────────────────────────

    def _poll_frame(self):
        try:
            img = self._live_frame_queue.get_nowait()
            self._photo = ctk.CTkImage(
                light_image=img, dark_image=img,
                size=(self._FRAME_W, self._FRAME_H),
            )
            self._cam_label.configure(image=self._photo)
        except queue.Empty:
            pass
        if self._root and self._root.winfo_exists():
            self._poll_id = self._root.after(self._POLL_MS, self._poll_frame)

    # ── UI 빌드 ───────────────────────────────────────────────────────────────

    def _build_ui(self):
        if self._parent is not None:
            self._root = ctk.CTkToplevel(self._parent)
        else:
            self._root = ctk.CTk()

        self._root.title("Turtle Check — 설정")
        self._root.resizable(False, False)
        self._root.configure(fg_color=_BG)
        self._root.attributes("-topmost", True)
        self._root.protocol("WM_DELETE_WINDOW", self._close)

        if platform.system() == "Darwin":
            try:
                self._root.createcommand("tk::mac::Quit", self._close)
            except Exception:
                pass

        # ── 좌측: 카메라 패널 ─────────────────────────────────────────────
        left = ctk.CTkFrame(self._root, fg_color=_BG_CAM, corner_radius=0)
        left.pack(side="left", fill="y")

        cam_wrap = ctk.CTkFrame(
            left, fg_color="#0d0d0d",
            border_color="#1c1c1c", border_width=1, corner_radius=14,
            width=self._FRAME_W, height=self._FRAME_H,
        )
        cam_wrap.pack(padx=14, pady=(14, 10))
        cam_wrap.pack_propagate(False)

        self._cam_label = ctk.CTkLabel(
            cam_wrap, text="", width=self._FRAME_W, height=self._FRAME_H,
        )
        self._cam_label.place(relx=0.5, rely=0.5, anchor="center")

        # MEDIAPIPE ACTIVE 상태 필
        pill = ctk.CTkFrame(
            left, fg_color=_SURF, corner_radius=20,
            border_color=_BORDER, border_width=1,
        )
        pill.pack(pady=(0, 14))
        dot = ctk.CTkFrame(pill, width=7, height=7, corner_radius=4, fg_color=_ACCENT)
        dot.pack_propagate(False)
        dot.pack(side="left", padx=(10, 5), pady=7)
        ctk.CTkLabel(
            pill, text="MEDIAPIPE ACTIVE",
            font=ctk.CTkFont(size=9), text_color="#3d4d3d",
        ).pack(side="left", padx=(0, 10), pady=7)

        # 세로 구분선
        tk.Frame(self._root, width=1, bg="#141414").pack(side="left", fill="y")

        # ── 우측: 컨트롤 패널 ─────────────────────────────────────────────
        right = ctk.CTkFrame(self._root, fg_color=_BG, corner_radius=0, width=self._PANEL_W)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ctk.CTkButton(
            right, text="창 닫기",
            fg_color="transparent", text_color="#4b5563",
            hover_color=_SURF, corner_radius=8,
            font=ctk.CTkFont(size=8), height=26,
            command=self._close,
        ).pack(side="bottom", fill="x", padx=16, pady=(0, 14))

        inner = ctk.CTkFrame(right, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=16, pady=(18, 4))

        # 앱 제목
        ctk.CTkLabel(
            inner, text="TURTLE CHECK",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_TEXT_HI,
        ).pack(anchor="w")
        ctk.CTkLabel(
            inner, text="POSTURE MONITOR",
            font=ctk.CTkFont(size=7), text_color="#6b7280",
        ).pack(anchor="w", pady=(1, 10))
        _hsep(inner)

        # 상태 뱃지
        badge_f = ctk.CTkFrame(
            inner, fg_color=_SURF, corner_radius=20,
            border_color=_BORDER, border_width=1,
        )
        badge_f.pack(anchor="w", pady=(8, 8))
        self._badge_dot = ctk.CTkFrame(badge_f, width=6, height=6, corner_radius=3, fg_color=_YELLOW)
        self._badge_dot.pack_propagate(False)
        self._badge_dot.pack(side="left", padx=(8, 4), pady=5)
        self._badge_lbl = ctk.CTkLabel(
            badge_f, text="STANDBY",
            font=ctk.CTkFont(size=8), text_color=_YELLOW,
        )
        self._badge_lbl.pack(side="left", padx=(0, 8), pady=5)

        # ACCOUNT 섹션
        ctk.CTkLabel(
            inner, text="ACCOUNT",
            font=ctk.CTkFont(size=7), text_color="#6b7280",
        ).pack(anchor="w", pady=(0, 3))

        self._auth_msg = tk.StringVar(value="")
        ctk.CTkLabel(
            inner, textvariable=self._auth_msg,
            font=ctk.CTkFont(size=8), text_color="#4b5563", wraplength=190,
        ).pack(anchor="w", pady=(0, 2))

        (self._login_frame, self._logged_frame, self._logged_lbl,
         self._email_entry, self._pw_entry) = _build_auth_section(
            inner, self._on_google_login, self._on_email_login, self._on_logout,
        )
        self._update_auth_ui()

        # CALIBRATION 섹션 구분선
        div = ctk.CTkFrame(inner, fg_color="transparent")
        div.pack(fill="x", pady=(6, 6))
        tk.Frame(div, height=1, bg="#141414").pack(side="left", fill="x", expand=True, pady=6)
        ctk.CTkLabel(
            div, text="  CALIBRATION  ",
            font=ctk.CTkFont(size=7), text_color="#6b7280",
        ).pack(side="left")
        tk.Frame(div, height=1, bg="#141414").pack(side="left", fill="x", expand=True, pady=6)

        ctk.CTkLabel(
            inner, text="바른 자세로 앉은 후\n버튼을 눌러주세요.",
            font=ctk.CTkFont(size=8), text_color="#9ca3af", justify="left",
        ).pack(anchor="w", pady=(0, 4))

        baseline_card, self._baseline_val, self._baseline_bar = _build_baseline_card(inner)
        baseline_card.pack(fill="x", pady=(0, 8))

        ctk.CTkButton(
            inner, text="캘리브레이션 시작  [P]",
            font=ctk.CTkFont(size=13, weight="bold"),
            width=190, height=60,
            fg_color=_ACCENT, hover_color=_ACCENT_H,
            text_color="#0a0a0a", corner_radius=12,
            command=self._on_calibrate,
        ).pack(fill="x")

        self._root.bind("<p>", lambda e: self._on_calibrate())
        self._root.bind("<P>", lambda e: self._on_calibrate())

        self._poll_id = self._root.after(self._POLL_MS, self._poll_frame)


# ── AuthWindow ────────────────────────────────────────────────────────────────

class AuthWindow:
    """트레이 메뉴 '로그인' 클릭 시 표시되는 컴팩트 창."""

    def __init__(self, auth_manager, parent=None):
        self.auth_manager = auth_manager
        self._parent      = parent
        self._root        = None
        self._on_complete = None

    def show_in_main_thread(self, on_complete):
        self._on_complete = on_complete
        self._build_ui()

    def _close(self, uid=None):
        self._msg = None
        self._root.destroy()
        if self._on_complete:
            self._on_complete(uid)

    def _on_google_login(self):
        self._msg.set("브라우저에서 구글 로그인을 진행해주세요...")
        self._root.update()

        def _do():
            uid = self.auth_manager.login_with_google()
            if uid:
                self._root.after(0, lambda: self._close(uid))
            else:
                err = self.auth_manager.last_error or "알 수 없는 오류"
                self._root.after(0, lambda: self._msg.set(f"로그인 실패: {err}"))
        threading.Thread(target=_do, daemon=True).start()

    def _build_ui(self):
        if self._parent is not None:
            self._root = ctk.CTkToplevel(self._parent)
        else:
            self._root = ctk.CTk()

        self._root.title("로그인")
        self._root.resizable(False, False)
        self._root.configure(fg_color=_BG)
        self._root.attributes("-topmost", True)
        self._root.protocol("WM_DELETE_WINDOW", lambda: self._close(None))

        if platform.system() == "Darwin":
            try:
                self._root.createcommand("tk::mac::Quit", lambda: self._close(None))
            except Exception:
                pass

        frame = ctk.CTkFrame(self._root, fg_color=_SURF, corner_radius=14,
                             border_color=_BORDER, border_width=1)
        frame.pack(padx=24, pady=24)

        inner = ctk.CTkFrame(frame, fg_color="transparent")
        inner.pack(padx=24, pady=20)

        ctk.CTkLabel(
            inner, text="TURTLE CHECK",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_TEXT_HI,
        ).pack(pady=(0, 2))
        ctk.CTkLabel(
            inner, text="POSTURE MONITOR",
            font=ctk.CTkFont(size=7), text_color=_TEXT_DIM,
        ).pack(pady=(0, 16))

        _hsep(inner)

        self._msg = tk.StringVar(value="")
        ctk.CTkLabel(
            inner, textvariable=self._msg,
            font=ctk.CTkFont(size=8), text_color="#4b5563",
            wraplength=220,
        ).pack(pady=(12, 4))

        google_icon = ctk.CTkImage(
            light_image=_GOOGLE_ICON_IMG,
            dark_image=_GOOGLE_ICON_IMG,
            size=(16, 16),
        )
        ctk.CTkButton(
            inner,
            text="구글 계정으로 시작",
            image=google_icon,
            compound="left",
            anchor="w",
            width=220,
            height=38,
            fg_color="#161616",
            border_color=_BORDER,
            border_width=1,
            corner_radius=10,
            text_color="#6b7280",
            hover_color="#1e1e1e",
            font=ctk.CTkFont(size=10),
            command=self._on_google_login,
        ).pack(pady=(4, 10))

        ctk.CTkButton(
            inner, text="취소",
            fg_color="transparent", text_color=_TEXT_DIM,
            hover_color=_SURF, corner_radius=8,
            font=ctk.CTkFont(size=9), height=28,
            command=lambda: self._close(None),
        ).pack()
