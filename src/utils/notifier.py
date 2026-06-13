import logging
import platform

log = logging.getLogger(__name__)

_APP_ID = "거북목 감지기"


def send_notify(title: str, msg: str) -> None:
    os_type = platform.system()
    if os_type == "Windows":
        try:
            from winotify import Notification, audio
            toast = Notification(app_id=_APP_ID, title=title, msg=msg)
            toast.set_audio(audio.Default, loop=False)
            toast.show()
        except Exception as e:
            log.warning("Windows 알림 실패: %s", e)
    elif os_type == "Darwin":
        try:
            from plyer import notification
            notification.notify(title=title, message=msg, app_name=_APP_ID)
        except Exception as e:
            log.warning("macOS 알림 실패: %s", e)
    else:
        log.warning("알림 미지원 플랫폼 (%s) — %s: %s", os_type, title, msg)
