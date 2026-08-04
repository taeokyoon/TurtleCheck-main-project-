"""
turtleCheck.py — 진입점

AppState 단일 인스턴스로 공유 상태 관리. 메인 스레드(tkinter) + 백그라운드 스레드
(camera_loop, pystray) 오케스트레이션.

로그는 logs/local/posture_log.jsonl 에 저장 (로그인/클라우드 업로드 없음, 로컬 전용).
"""
import json
import logging
import os
import queue
import sys
import threading
import time
import tkinter as tk

import cv2
from dotenv import load_dotenv
from PIL import Image

from src.detector   import PostureDetector
from src.log_config import setup_logging
from src.logger      import PostureLogger
from src.tray_app    import build_tray, set_tray_state, notify

# ── 경로 설정 (개발/exe 공통) ─────────────────────────────────────────────────

if getattr(sys, "frozen", False):
    _BASE = sys._MEIPASS   # 번들 리소스 (.env, config, assets 모두 포함)
else:
    _BASE = os.path.dirname(os.path.abspath(__file__))

_CONFIG_PATH = os.path.join(_BASE, "config.json")
_MASCOT_PATH = os.path.join(_BASE, "assets", "mascot.png")

load_dotenv(os.path.join(_BASE, ".env"))

with open(_CONFIG_PATH, encoding="utf-8") as f:
    cfg = json.load(f)

SAVE_INTERVAL    = cfg["save_interval_seconds"]
APP_DATA_DIR     = os.path.join(_BASE, "logs")
NOTIFY_COOLDOWN  = 10.0   # 거북목 알림 재발송 최소 간격 (초)
POLL_INTERVAL_MS = 200    # tkinter 이벤트 큐 폴링 간격 (ms)

os.makedirs(APP_DATA_DIR, exist_ok=True)
setup_logging(APP_DATA_DIR)
log = logging.getLogger(__name__)

# ── AppState ──────────────────────────────────────────────────────────────────

class AppState:
    """앱 전체 공유 상태 — 전역 변수를 단일 인스턴스로 관리."""

    def __init__(self):
        self.detector    = PostureDetector(cfg["delta_turtle"], cfg["delta_ok"])
        self.logger      = PostureLogger(os.path.join(APP_DATA_DIR, "local"))
        self.stop_event  = threading.Event()
        self.tray_icon   = None
        self.last_save   = time.time()
        self.tk_root:    tk.Tk | None = None
        self.tk_queue:   queue.Queue  = queue.Queue()
        self.show_visual: bool        = False
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)

# ── 트레이 콜백 ───────────────────────────────────────────────────────────────

def _make_callbacks(app: AppState) -> dict:
    """트레이 메뉴 콜백 딕셔너리 생성. app 을 클로저로 캡처."""

    def on_open_gui(icon, item) -> None:
        """설정 화면 열기 — 마스코트 + 카메라 + 캘리브레이션 창."""
        def _show():
            from src.startup_window import SettingsWindow
            SettingsWindow(
                detector=app.detector,
                live_frame_queue=app.frame_queue,
                start_visual=lambda: setattr(app, "show_visual", True),
                stop_visual=lambda: setattr(app, "show_visual", False),
                mascot_path=_MASCOT_PATH,
                parent=app.tk_root,
            ).show_in_main_thread()
        app.tk_queue.put(_show)

    def on_quit(icon, item) -> None:
        app.stop_event.set()
        icon.stop()
        app.tk_queue.put(lambda: app.tk_root.quit() if app.tk_root else None)

    return dict(
        on_open_gui=on_open_gui,
        on_quit=on_quit,
    )

# ── 카메라 루프 (백그라운드 스레드) ────────────────────────────────────────────

def camera_loop(app: AppState) -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        notify("오류", "카메라를 열 수 없습니다.")
        return

    last_notify_time = 0.0
    try:
        while not app.stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            if app.show_visual:
                features, rgb = app.detector.process_frame_visual(frame)
                try:
                    app.frame_queue.put_nowait(
                        Image.fromarray(rgb).resize((520, 390), Image.BILINEAR)
                    )
                except queue.Full:
                    pass
            else:
                features = app.detector.process_frame(frame)

            did_evaluate, changed = app.detector.update(features)

            if did_evaluate and app.detector.baseline_score is not None:
                app.logger.tick(app.detector.is_turtle)
                if changed:
                    set_tray_state(
                        app.tray_icon,
                        app.detector.baseline_score,
                        app.detector.is_turtle,
                    )
                    if not app.detector.is_turtle:
                        last_notify_time = 0.0

            if (app.detector.baseline_score is not None
                    and app.detector.is_turtle
                    and time.time() - last_notify_time >= NOTIFY_COOLDOWN):
                notify("거북목 감지!", "자세를 바로잡아 주세요.")
                last_notify_time = time.time()

            now = time.time()
            if app.detector.baseline_score is not None and now - app.last_save >= SAVE_INTERVAL:
                app.last_save = now
                app.logger.flush()
    finally:
        app.detector.close()
        cap.release()

# ── 진입점 ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.startup_window import StartupWindow

    app = AppState()

    # Phase 1: 시작 창 (마스코트 + 카메라 피드 + 캘리브레이션)
    startup_done = threading.Event()
    _startup_root = StartupWindow(
        detector=app.detector,
        on_done=startup_done.set,
        mascot_path=_MASCOT_PATH,
    ).run()  # CTk 루트 반환 (withdraw 상태)

    if not startup_done.is_set():
        raise SystemExit(0)

    # Phase 2: 트레이 모드
    callbacks     = _make_callbacks(app)
    app.tray_icon = build_tray(**callbacks)

    if app.detector.baseline_score is not None:
        set_tray_state(app.tray_icon, app.detector.baseline_score, app.detector.is_turtle)

    threading.Thread(target=camera_loop, args=(app,), daemon=True).start()
    threading.Thread(target=app.tray_icon.run, daemon=True).start()

    notify("백그라운드 모드", "트레이 아이콘 → '설정 화면 열기'에서 언제든 재설정할 수 있습니다.")

    # 메인 스레드: 팝업 전용 tkinter 이벤트 루프
    # StartupWindow의 CTk 루트를 재사용 — 새 Tk() 생성 시 stale Tcl 콜백 문제 방지
    app.tk_root = _startup_root

    def _poll() -> None:
        try:
            while True:
                cb = app.tk_queue.get_nowait()
                try:
                    cb()
                except Exception:
                    log.exception("tk_queue 콜백 오류")
        except queue.Empty:
            pass
        if not app.stop_event.is_set():
            app.tk_root.after(POLL_INTERVAL_MS, _poll)
        else:
            app.tk_root.quit()

    app.tk_root.after(POLL_INTERVAL_MS, _poll)
    app.tk_root.mainloop()
