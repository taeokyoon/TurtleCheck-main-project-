"""
startup_window.py — 앱 시작 창 + 트레이 설정 창

크로스플랫폼: customtkinter + CTkImage (Windows / macOS 공통)

StartupWindow  : 앱 시작 시 (카메라 피드 + 캘리브레이션)
SettingsWindow : 트레이 "설정 화면 열기" 클릭 시
"""
import platform
import queue
import threading
import tkinter as tk
import customtkinter as ctk

import cv2
from PIL import Image

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


# ── 공유: 캘리브레이션 버튼 클릭 → 완료 처리 ──────────────────────────────────────

class _CalibrationMixin:
    """캘리브레이션 버튼 클릭 시 즉시 baseline 을 설정하고 완료 상태를 표시한다.
    사용하는 클래스는 self.detector, self._status_msg, self._baseline_bar,
    self._baseline_val, self._badge_dot, self._badge_lbl 를 가지고 있어야 한다."""

    def _on_calibrate(self):
        baseline = self.detector.calibrate()
        if baseline is None:
            self._status_msg.set("자세가 감지되지 않았습니다. 잠시 후 다시 시도하세요.")
            return
        self._baseline_val.configure(text=f"{baseline:.3f}", text_color=_ACCENT)
        self._baseline_bar.set(min(abs(baseline) / 0.8, 1.0))
        self._badge_dot.configure(fg_color=_ACCENT)
        self._badge_lbl.configure(text="ACTIVE", text_color=_ACCENT)
        self._on_calibration_done(baseline)

    def _on_calibration_done(self, baseline: float) -> None:
        """완료 메시지 표시. 창마다 다른 문구/버튼 처리가 필요하면 오버라이드."""
        self._status_msg.set(f"완료 — 기준값 {baseline:.3f}")


# ── StartupWindow ─────────────────────────────────────────────────────────────

class StartupWindow(_CalibrationMixin):
    """앱 시작 시 표시되는 창. 좌: 카메라 피드, 우: 캘리브레이션."""

    _FRAME_W = 520
    _FRAME_H = 440
    _PANEL_W = 224
    _POLL_MS = 33

    def __init__(self, detector, on_done, mascot_path: str | None = None):
        self.detector    = detector
        self.on_done     = on_done
        self.mascot_path = mascot_path

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
            features, rgb = self.detector.process_frame_visual(frame)
            if features is not None:
                self.detector.update(features)
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

    # ── 캘리브레이션 완료 처리 (공통 흐름은 _CalibrationMixin) ──────────────────────

    def _on_calibration_done(self, baseline: float) -> None:
        self._status_msg.set(f"완료 — 기준값 {baseline:.3f}  만족하면 아래 버튼을 눌러주세요.")
        self._continue_btn.configure(text="시작하기")

    def _on_continue(self):
        if self.detector.baseline_score is None:
            self._status_msg.set("캘리브레이션이 설정되지 않았습니다. 캘리브레이션을 먼저 설정해주세요.")
            return
        self._finish()

    def _finish(self):
        self._status_msg = None
        self._stop_cam.set()
        _cancel_all_after(self._root)
        self.on_done()
        self._root.withdraw()
        self._root.quit()

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

        self._status_msg = tk.StringVar(value="바른 자세로 앉은 후 캘리브레이션을 시작해주세요.")
        ctk.CTkLabel(
            inner, textvariable=self._status_msg,
            font=ctk.CTkFont(size=8), text_color="#4b5563", wraplength=190, justify="left",
        ).pack(anchor="w", pady=(0, 8))

        self._continue_btn = ctk.CTkButton(
            inner, text="계속하기",
            fg_color=_SURF, border_color=_BORDER, border_width=1,
            text_color="#6b7280", hover_color="#161616",
            corner_radius=10, font=ctk.CTkFont(size=10),
            height=36, width=190,
            command=self._on_continue,
        )
        self._continue_btn.pack(anchor="w", pady=(2, 6))

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

class SettingsWindow(_CalibrationMixin):
    """트레이 "설정 화면 열기" 클릭 시 표시되는 창."""

    _FRAME_W = 520
    _FRAME_H = 440
    _PANEL_W = 224
    _POLL_MS = 33

    def __init__(self, detector, live_frame_queue,
                 start_visual, stop_visual,
                 mascot_path: str | None = None,
                 parent=None):
        self.detector          = detector
        self._live_frame_queue = live_frame_queue
        self._start_visual     = start_visual
        self._stop_visual      = stop_visual
        self.mascot_path       = mascot_path
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
        self._status_msg = None
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

    # ── 프레임 폴링 (캘리브레이션 흐름은 _CalibrationMixin) ─────────────────────────

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

        self._status_msg = tk.StringVar(value="")
        ctk.CTkLabel(
            inner, textvariable=self._status_msg,
            font=ctk.CTkFont(size=8), text_color="#4b5563", wraplength=190,
        ).pack(anchor="w", pady=(0, 2))

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
