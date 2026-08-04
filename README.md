# TurtleNeckDetector

> **CNN 전이학습 기반 실시간 거북목 감지 및 자세 교정 코칭 시스템**
> 디지털 헬스케어 / AI 헬스테크 — 대학교 캡스톤 가상 창업 프로젝트

웹캠 하나로 거북목 자세를 실시간 감지하는 크로스플랫폼 백그라운드 앱입니다.
현재 **v1.0** (MediaPipe 규칙 기반 판정)을 운영 중이며, **v2.0**에서 EfficientNet-B0 전이학습 모델로 판정 엔진을 전면 교체할 예정입니다.

## 빠른 시작

1. 첫번째 방법
```bash
release 최신 배포판 .zip 압축파일을 다운로드 후 압축을 푼다.
.exe 파일을 실행한다
```

3. 두번째 방법
```bash
# 가상환경 venv 설치
py -3.11 -m venv .venv
.venv/scripts/activate

# 필요 라이브러리 설치
pip install -r requirements.txt

# 실행
python turtleCheck.py
```

트레이 아이콘 우클릭 → **캘리브레이션** → 모니터링 시작

# 실제 워크플로우
코드 수정
    ↓
py turtleCheck.py  ← 빠른 테스트 (이걸로 주로 개발)
    ↓
잘 되면 pyinstaller로 빌드  ← .exe 갱신
    ↓
.exe로 최종 확인

## 상세 문서

[DEVELOP.md](DEVELOP.md) 참고
