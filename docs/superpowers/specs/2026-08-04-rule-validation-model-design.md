# 규칙 기반 거북목 판정의 데이터 기반 검증 — 설계 문서

## 배경 및 목표

**문제**: "자유주제 AI 경진대회" 규정상 AI 활용과 데이터셋 사용이 필수 심사 항목이다. 그런데 현재
앱(`turtleCheck.py`)의 거북목 판정은 MediaPipe Pose(구글의 사전학습 모델, 파인튜닝 없이 그대로 사용)로
랜드마크만 뽑고, 실제 판정은 사람이 정한 임계값(`delta_turtle=0.10`)과 비교하는 규칙 기반 로직
(`src/detector.py`의 `_combine()`)이다. 학습된 모델이 파이프라인에 전혀 없어 "AI가 어디 있냐"는
질문에 답할 근거가 없다.

**검토했다가 기각한 대안**:
- CNN(EfficientNet-B0) 전이학습으로 판정 엔진 교체 (`AGENTS.md`의 기존 v2.0 로드맵) — 이미지 데이터셋이
  필요한데 팀 인력상 본인 1인 데이터만 가능해 다양성 확보가 불가능. 과적합 위험이 크고, 실시간 CPU
  추론 최적화 등 부수 리스크도 큼. 기각.
- 학습된 분류기로 판정 로직 자체를 교체 — 이미 튜닝되어 정상 동작 중인 배포 로직을 건드리는 리스크가
  있고, "의학적으로 검증된 판정"이라고 주장할 근거가 애초에 없다는 지적에 따라, 판정 로직 자체는
  건드리지 않기로 결정.
- 합성(랜덤) 데이터로 "여러 사람 데이터"를 흉내내는 방안 — 판정 공식으로 라벨을 만들면 그 공식을
  다시 배우는 순환 논리이며, 데이터 조작으로 비칠 위험이 있어 기각.

**최종 방향**: 배포되는 판정 로직(`src/detector.py`)은 **그대로 유지**한다. 대신 본인이 직접 라벨링한
데이터셋을 수집하고, 그 데이터로 로지스틱 회귀 모델을 학습시켜 "우리 규칙 기반 판정이 실제 라벨과
얼마나 일치하는가"를 정량적으로 검증하는 별도 도구를 추가한다. 학습된 모델은 배포되지 않고, 검증
근거(리포트)로만 쓰인다.

## 아키텍처 개요

```
tools/collect_labels.py     라벨링 데이터 수집 (신규, 개발용 스크립트)
tools/train_model.py        학습 + 검증 리포트 생성 (신규, 개발용 스크립트)
tools/requirements.txt      scikit-learn, pandas (신규, 개발 전용 의존성)
dataset/posture_labels.csv  수집된 데이터셋 (신규, 제출 증거)
dataset/validation_report.md 정확도/혼동행렬 비교 리포트 (신규, 발표 자료용)
```

`turtleCheck.py`, `src/detector.py`, `src/tray_app.py`, `src/startup_window.py` 등 배포 코드는
**일체 수정하지 않는다.** `src/detector.py`의 `PostureDetector`, `_combine()`은 두 신규 도구에서
import하여 재사용한다.

## 데이터 수집 (`tools/collect_labels.py`)

**흐름**:
1. 스크립트 시작 시 웹캠을 열고 `PostureDetector`로 짧게 캘리브레이션(`calibrate()`)을 먼저 수행해
   `baseline_score`/`baseline_features`를 확보한다.
2. 이후 매 프레임 피처를 계산하고, **baseline 대비 편차**(`feature - baseline_feature`, 4개 값)를
   화면에 표시한다.
3. 키보드 토글 방식으로 기록한다: `n` 키를 누르면 "정상 기록 시작", 다시 누르면 "정상 기록 종료".
   `t` 키는 거북목용으로 동일하게 동작. 두 라벨은 동시에 기록되지 않는다 — 한쪽이 기록 중일 때
   다른 키를 누르면 무시한다(먼저 현재 기록을 종료해야 다른 라벨 기록을 시작할 수 있음).
   기록 중에는 약 1초 간격으로 한 행씩 자동 저장된다. `q`로 스크립트를 종료한다.
4. 각 기록 세션은 고유한 `session_id`를 가진다 (기록 시작 시각의 타임스탬프 문자열, 예:
   `20260804153012`). 세션 단위 구분은 이후 train/test 분리에 사용된다.
5. 랜드마크 인식 실패(`features is None`, 손 가림 등)로 유효 피처를 못 얻은 프레임은 기록하지 않는다.

**저장 포맷** — `dataset/posture_labels.csv`, 컬럼:

| 컬럼 | 설명 |
|---|---|
| `session_id` | 기록 세션 식별자 |
| `y_score`, `z_forward`, `head_tilt`, `ear_z_offset` | 원시 피처 값 (baseline 차감 전) |
| `baseline_y_score`, `baseline_z_forward`, `baseline_head_tilt`, `baseline_ear_z_offset` | 캘리브레이션 시점의 baseline 피처 (세션 내 모든 행에 동일하게 기록) |
| `label` | 0=정상, 1=거북목 (사람이 직접 지정) |

> **왜 편차를 미리 빼지 않고 원시값+baseline을 그대로 저장하는가**: `src/detector.py`의
> `_combine()`은 `z_forward` 항에 `y_score`의 절댓값 기반 게이트(`z_gate`)가 곱해지는 비선형
> 함수다. 편차를 미리 계산해서 저장하면 이 게이트가 "절대 y_score" 대신 "y_score 편차"로
> 계산되어 배포 로직과 다른 값이 나온다. 원시값과 baseline을 그대로 저장해두면, 검증
> 스크립트에서 `_combine(raw) - _combine(baseline)`으로 프로덕션과 완전히 동일한 공식을
> 재현할 수 있다.

**데이터 수집 가이드라인** (스크립트 실행 시 안내 문구로도 표시):
- 클래스(정상/거북목)당 2~3분짜리 세션을 5개 정도로 나눠 진행 (클래스당 총 10~15분).
- 세션마다 하위 자세를 다르게 한다.
  - 정상: 똑바로 앉기 / 등받이에 기대기 / 고개 좌우로 살짝 돌리기 / 어깨 위치 바꾸기
  - 거북목: 살짝 숙이기 / 폰 보듯 아래로 보기 / 심하게 숙이기 / 한쪽으로 기울며 숙이기
- 한 자세로 오래 가만히 있지 않는다 — 값이 거의 중복되어 모델이 특정 자세만 암기하게 된다.
- 정상/거북목 데이터 개수를 비슷하게 맞춘다(클래스 불균형 방지).

## 학습 및 검증 (`tools/train_model.py`)

**흐름**:
1. `dataset/posture_labels.csv`를 로드한다.
2. `session_id` 기준으로 세션 단위 train(약 80%)/test(약 20%) 분리를 한다. **행 단위 무작위 분리는
   금지** — 같은 세션의 행들은 서로 거의 동일해 데이터 누수(leakage)가 발생하고 정확도가 부풀려진다.
3. 각 행의 원시 피처에서 baseline을 뺀 편차(4개)를 계산하고, `sklearn.linear_model.LogisticRegression`을
   train 세션에 대해 학습한다.
4. **학습 모델 평가**: test 세션에서 accuracy/precision/recall/confusion matrix를 계산한다.
5. **규칙 기반 평가**: `src/detector.py`에서 `_combine()`과 `delta_turtle`(`config.json` 값)을
   import하여, 동일한 test 세션 데이터에 그대로 적용한 예측을 만들고 동일한 지표를 계산한다.
6. 4번과 5번 결과를 나란히 비교하는 리포트를 `dataset/validation_report.md`에 마크다운으로 저장한다
   (표 형태: 모델명 / accuracy / precision / recall / confusion matrix).
7. 재현성을 위해 학습된 모델을 `dataset/model.pkl`로 저장한다 (배포에는 쓰이지 않음, 증거 자료용).

**에러 처리**: CSV가 없거나 특정 라벨이 하나도 없으면 명확한 에러 메시지를 출력하고 중단한다.

## 테스트 방침

개발용 1회성 스크립트이므로 자동화 테스트는 작성하지 않는다. 각 스크립트를 직접 실행해
(1) CSV에 행이 정상적으로 쌓이는지, (2) 리포트의 지표가 상식적인 범위인지를 육안으로 확인하는
것으로 검증을 대신한다.

## 문서 갱신

- `AGENTS.md`의 "목표(v2.0): EfficientNet-B0 전이학습 모델로 판정 엔진 교체" 및 "v2.0 개발 제약"
  섹션(CNN/Colab/3-class 관련 내용)을 이번 결정 내용으로 교체한다.

## 범위 밖 (Out of Scope)

- 판정 로직(`src/detector.py`) 자체의 변경 — 하지 않는다.
- LLM 기반 코칭 피드백 — 별도 논의 대상, 이 spec에는 포함하지 않는다.
- 다인 데이터 수집, 이미지 기반 CNN — 팀 인력 제약상 채택하지 않는다.
