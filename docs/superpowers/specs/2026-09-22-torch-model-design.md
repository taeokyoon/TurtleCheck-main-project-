# 랜드마크 기반 PyTorch 거북목 판정 모델 — 설계 문서 (Phase 2)

## 배경 및 목표

**현재 상태**: 판정 결정권은 규칙 기반(`src/detector.py`의 `_combine()` + 임계값)에 있고, AI는 로지스틱
회귀(`src/ai_advisor.py`)로 "일치/불일치"를 참고 표시하는 수준이다. 8월 검증에서 로지스틱 회귀(87.9%)가
규칙 기반(98.8%)보다 못했는데, 이는 사람이 설계한 피처 4개를 선형 결합만 할 수 있어 규칙의 비선형
게이트(`z_gate`)를 재현하지 못한 결과였다.

**Phase 1에서 확보한 것**: `dataset/posture_v2.csv` — 상체 랜드마크 13개 원시값(x,y,z,visibility)이
5Hz·타임스탬프 포함으로 저장되고, 한 세션 안에 정상↔거북목 전환이 섞여 있다 (2세션, 1,156행, 1명).

**목표**: 사람이 설계한 피처 대신 **원시 랜드마크에서 모델이 스스로 특징을 학습**하게 하고,
(1) 단일 프레임 MLP, (2) 시계열 GRU, (3) 기존 규칙 기반 — 세 판정을 같은 데이터에서 공정하게
비교한다. 최고 모델은 ONNX로 내보내 Phase 3(앱 통합)의 입력으로 삼는다.

**검토했다가 기각한 대안**:
- Colab GPU 학습 — 1,156행 × 파라미터 수천 개 모델은 CPU로 수 초면 끝난다. GPU 이점 없음,
  데이터 업로드·결과 동기화 번거로움만 추가. 로직을 `tools/train_torch.py`에 두므로 필요해지면
  노트북에서 import만 하면 된다.
- 웹캠 이미지 직접 CNN — 이미지 데이터 미수집, 1인 데이터로는 얼굴·배경 암기, 개인정보, 실시간 CPU
  추론 부담. MediaPipe BlazePose가 이미 이미지→좌표를 담당하는 CNN이므로 그 위에 작은 모델을 얹는
  현재 구조가 데이터가 적을 때 유리하다 (8월 spec의 EfficientNet 기각과 같은 이유).
- 합성(보간) 데이터 증강 — 증강 없는 기준선이 먼저 있어야 효과를 비교할 수 있다. Phase 2 완료 후
  별도 실험으로 미룬다.

## 아키텍처 개요

```
dataset/posture_v2.csv (실제 데이터, 수정 없음)
        │
        ▼
tools/train_torch.py
        │  ① normalize_landmarks()   어깨 기준 정규화 → 프레임당 52차원
        │  ② make_sequences()        GRU용 10스텝 윈도우 (같은 세션·같은 라벨 구간)
        │  ③ leave_one_session_out() 세션 단위 교차검증 폴드 생성
        │  ④ fit/apply standardizer  학습 폴드 기준 mean/std 표준화
        │  ⑤ MLP / GRU 학습 (early stopping) + 규칙 기반 예측 → 폴드별 지표
        │  ⑥ 전체 데이터로 최종 재학습 → ONNX 내보내기
        ▼
dataset/torch_report.md            3자 비교 리포트 (폴드별 + 평균)
dataset/torch_report/*.png         학습 곡선, 3자 비교 막대, 혼동행렬
dataset/mlp_model.onnx             Phase 3 앱 통합용
dataset/gru_model.onnx
dataset/feature_norm.json          표준화 mean/std + 윈도우 길이 (앱에서 동일 적용 필수)

tools/test_train_torch.py          ①~④ 순수 로직 + 모델 forward shape pytest
tools/requirements.txt             torch(CPU), onnx, onnxruntime, matplotlib 추가
```

`turtleCheck.py`, `src/` 배포 코드는 **수정하지 않는다.** 배포 `requirements.txt`도 손대지 않는다
(Phase 3에서 `onnxruntime`만 추가 예정). 규칙 기반 비교는 `tools/train_model.py`의
`rule_based_predict()`를 import해 재사용한다.

## 입력 피처 — 어깨 기준 정규화 랜드마크

프레임마다 13개 랜드마크(MediaPipe Pose 0~12번)를 다음처럼 변환한다:

```
center = (LEFT_SHOULDER(11) + RIGHT_SHOULDER(12)) / 2        # x, y, z 각각
width  = |LEFT_SHOULDER.x - RIGHT_SHOULDER.x|
for each landmark i in 0..12:
    x'_i = (x_i - center.x) / width
    y'_i = (y_i - center.y) / width
    z'_i = (z_i - center.z) / width
feature = [x'_0, y'_0, z'_0, ..., x'_12, y'_12, z'_12, v_0, ..., v_12]   # 39 + 13 = 52
```

- 원점을 어깨 중심에 두면 카메라 좌우·상하 위치에 무관해지고, 어깨 너비로 나누면 카메라 거리에
  무관해진다. `src/detector.py`의 `_calc_features()`가 어깨 너비로 나누는 것과 같은 원리다.
- 캘리브레이션(baseline)을 쓰지 않는다 — 모델이 절대 자세를 배운다. 이는 규칙 기반과의 차이점이며,
  "캘리브레이션 없이도 되는가"가 비교 포인트 중 하나다.
- `width`가 0에 가까우면(측면 촬영) 그 행은 제외한다. 수집 도구가 이미 `_MIN_SHOULDER_W`로
  걸러서 실제로는 드물다.

## 시퀀스 생성 (GRU용)

- 윈도우 길이 `SEQ_LEN = 10` (5Hz × 2초), 스트라이드 1.
- 같은 `session_id` 안에서, `label`이 같은 **연속 구간** 안에서만 윈도우를 자른다. 라벨이 바뀌는
  지점을 걸치는 윈도우는 만들지 않는다 (라벨을 하나로 정할 수 없으므로).
- 윈도우의 라벨 = 구간의 라벨. 시퀀스의 세션 id는 원본 세션 id를 그대로 가져 폴드 분할에 쓴다.
- 연속 구간이 10행 미만이면 그 구간은 버린다.

## 검증 방법 — 세션 단위 Leave-One-Out

- 폴드 수 = 세션 수. 각 폴드에서 세션 하나가 test, 나머지가 train.
- 지금은 2세션이라 2폴드. 사람이 늘어나면 그룹 키를 `person_id`로 바꾸는 것만으로 사람 단위
  검증이 된다 (함수 인자 `group_col`로 열어둔다).
- 표준화(mean/std)는 **train 폴드에서만** 계산해 test에 적용한다. test 정보가 새면 지표가 부풀려진다.
- 지표: Accuracy, Precision, Recall, F1, 혼동행렬. 폴드별 + 평균을 리포트에 쓴다.
- MLP와 GRU는 행 수가 다르므로(GRU는 윈도우 단위), 규칙 기반은 각각의 단위에 맞춰 두 번 계산한다
  — MLP와는 프레임 단위, GRU와는 윈도우 마지막 프레임 기준.

## 모델

| | MLP (단일 프레임) | GRU (시계열) |
|---|---|---|
| 입력 | 52 | (10, 52) |
| 구조 | Linear 52→32, ReLU, Dropout 0.2, Linear 32→16, ReLU, Linear 16→1 | GRU(input 52, hidden 32, 1층) → 마지막 스텝 → Linear 32→16, ReLU, Linear 16→1 |
| 파라미터 | ≈2.2K | ≈9K |
| 출력 | 로짓 1개. 확률 = sigmoid, 판정 기준 50% (v1과 같은 원칙: 규칙에 맞추려 조정하지 않음) | 동일 |
| 손실/옵티마이저 | BCEWithLogitsLoss, Adam(lr 1e-3) | 동일 |
| 배치 / epoch | 64 / 최대 100 | 동일 |
| Early stopping | 검증 손실이 10 epoch 연속 개선 없으면 중단, 최저 시점 가중치 복원. 검증 세트 = train 폴드의 마지막 세션 (폴드 안에서 다시 세션 단위 분리; 세션이 1개뿐이면 train의 뒤 20%) | 동일 |
| 재현성 | `torch.manual_seed(42)`, `numpy.random.seed(42)` | 동일 |

클래스 불균형(651:505)은 크지 않아 가중치 조정 없이 간다. 모델을 일부러 작게 잡은 이유는
1,156행에 큰 모델은 암기만 하기 때문이다.

## 최종 모델과 내보내기

- 교차검증이 끝나면 **전체 데이터**로 MLP·GRU를 다시 학습한다 (v1 `train_model.py`와 같은 원칙:
  검증은 held-out으로 공정하게, 배포 모델은 데이터를 아끼지 않는다). 이때 early stopping용 검증
  세트는 마지막 세션.
- `torch.onnx.export`로 `dataset/mlp_model.onnx`, `dataset/gru_model.onnx` 저장. 입력 이름
  `"input"`, 출력 이름 `"logit"`, 배치 차원은 동적.
- `dataset/feature_norm.json`:
  ```json
  {"mean": [52개], "std": [52개], "seq_len": 10, "landmark_count": 13}
  ```
- 내보낸 뒤 `onnxruntime`으로 같은 입력을 넣어 PyTorch 출력과 확률 차이가 1e-5 이하인지
  스크립트 안에서 자동 확인하고 리포트에 기록한다.

## 리포트 — `dataset/torch_report.md`

- 데이터 요약 (행 수, 세션 수, 사람 수, 라벨 분포)
- 폴드별 표: MLP / GRU / 규칙 기반 × Accuracy, Precision, Recall, F1
- 평균 표 + 혼동행렬
- 그래프 (`dataset/torch_report/`): `mlp_loss.png`, `gru_loss.png`, `comparison.png`, `confusion.png`
- ONNX 동일성 확인 결과
- 해석 문단은 스크립트가 쓰지 않는다 — 결과를 보고 사람이 DEVELOP.md에 쓴다.

## 테스트 (`tools/test_train_torch.py`)

| 대상 | 검증 |
|---|---|
| `normalize_landmarks()` | 어깨 중심이 (0,0,0)이 되는지, 어깨 너비가 1이 되는지, 평행이동·확대한 입력이 같은 출력을 내는지 |
| `make_sequences()` | 라벨 경계·세션 경계를 넘는 윈도우가 없는지, shape가 (N, 10, 52)인지, 짧은 구간이 버려지는지 |
| `leave_one_session_out()` | 폴드 수 = 세션 수, 각 test에 세션 정확히 1개, train/test 겹침 없음 |
| `fit_standardizer()` / `apply_standardizer()` | train으로 fit 후 train 변환 평균 ≈ 0, std ≈ 1 |
| `MlpModel` / `GruModel` forward | 입력 shape → 출력 shape (배치, 1) |

학습 루프 자체는 느리고 확률적이라 단위 테스트에서 제외한다. 통합 검증은 실제 실행으로 한다.

## 통합 검증 (수동)

1. `cd tools && pytest` — 기존 11개 + 신규 통과
2. `python tools/train_torch.py` — 수십 초 안에 종료, `torch_report.md`에 3자 지표, PNG 4개,
   ONNX 2개, `feature_norm.json` 생성
3. 리포트의 ONNX 동일성 확인이 통과로 기록됨

## 문서 갱신

- `AGENTS.md` 검증 도구 표에 `train_torch.py`, `test_train_torch.py` 추가
- `DEVELOP.md`: "규칙 기반 판정 검증" 섹션 뒤에 "랜드마크 기반 PyTorch 모델" 섹션 추가 — 배경,
  피처, 3자 비교 결과 표(실행 후 채움), 해석. 개발 단계 현황 표에 행 추가.

## Save 기준

1. 피처·시퀀스·분할·표준화 함수 + 테스트 통과
2. 모델·학습 루프 + 로컬 실행으로 리포트·그래프 생성
3. ONNX 내보내기 + 동일성 확인 통과
4. 문서 갱신

## 이 단계에서 하지 않는 것

- 앱 통합 (Phase 3) — `onnxruntime` 추론 모듈, AI 판정 모드 토글, 규칙 기반 fallback
- 데이터 증강 실험 — Phase 2 기준선이 나온 뒤 별도로
- 다인 데이터 수집 — 도구는 준비돼 있음 (`--person`)
- MediaPipe Tasks API 이전 (Phase 4)
