# TurtleNeckDetector

> **데이터 기반으로 검증한 실시간 거북목 감지 및 자세 교정 코칭 시스템**
> 디지털 헬스케어 / AI 헬스테크 — 대학교 캡스톤 가상 창업 프로젝트

웹캠 하나로 거북목 자세를 실시간 감지하는 크로스플랫폼 백그라운드 앱입니다.
**v1.0** (MediaPipe 랜드마크 + 규칙 기반 판정)을 운영 중이며, 직접 라벨링한 데이터셋으로
규칙 기반 판정을 정량 검증했습니다. 이어서 원시 랜드마크로 PyTorch MLP를 학습시켜 규칙 기반보다
높은 정확도(95.3% vs 91.6%)를 확인했고, 앱 통합(Phase 3)을 앞두고 있습니다 — 진행률과 다음 단계는
[DEVELOP.md 로드맵](DEVELOP.md#로드맵-및-진행-현황) 참고.

## 빠른 시작

1. 첫번째 방법
```bash
release 최신 배포판 .zip 압축파일을 다운로드 후 압축을 푼다.
.exe 파일을 실행한다
```

2. 두번째 방법
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
