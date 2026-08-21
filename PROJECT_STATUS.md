# PROJECT_STATUS.md

## 프로젝트
MCT 설비 데이터 기반 AI 예지보전 프로젝트

현재 목표:
- 기존 AI/백엔드 파이프라인은 유지
- Streamlit은 내부 검증/간단 시연용으로 보관
- 교수님/기업 시연용 최종 프론트는 React로 신규 구축
- 실제 동작하는 기능만 구현
- 향후 기업이 참고할 수 있는 서비스형 프로토타입 수준으로 완성

---

## 1. 현재 완료 상태

### 데이터 전처리 / EDA
- 2026년 5월 1일~31일 MCT 원본 로그 확보
- 총 원본 라인 수: 약 6,261,455
- 파싱 결과: 481,648행 × 21열
- 정제 결과: 481,648행 × 31열
- 정제 파일: `output/mct_clean.csv`

### STATUS
- STATUS_RUN: 243,319
- STATUS_READY: 238,313
- STATUS_ERROR: 16

### 실제 절삭 정의
```text
STATUS_RUN
AND spindle_rpm > 0
AND feed_rate > 0
```

실제 절삭 행:
- 219,195

RUN 내부:
- RPM>0 & FEED>0: 219,195
- RPM=0 & FEED=0: 16,754
- RPM=0 & FEED>0: 6,509
- RPM>0 & FEED=0: 861

### 샘플링 간격
- 전체 데이터 평균: 약 5.56초
- 중앙값: 약 5.42초

### 공구별 정상 기준
전체 평균이 아니라 공구별 정상 Baseline 사용.

Baseline은 Train 정상 Window 기준으로 생성.

지원 공구:
```text
[1,2,3,4,5,6,8,10,11,12,13,14,18,19]
```

Tool 15/16은 Train 정상 Baseline이 없어 분석 제외.

---

## 2. Feature Engineering

1분 Window 사용.

이유:
- 약 5~6초 샘플링
- 1분에 약 10~11개 센서값
- 짧은 순간값보다 구간의 변화 패턴을 보기 위함

최소 Window 표본 수:
- 9개

모델 입력 Feature 14개:
```text
rpm_mean
rpm_std
rpm_min
rpm_max
rpm_range
load_mean
load_std
load_max
rpm_error_mean
rpm_error_std
rpm_error_absmax
load_vs_normal
rpm_std_vs_normal
rpm_error_vs_normal
```

RPM_ERROR:
```text
spindle_rate - spindle_rpm
```

---

## 3. 가상 이상 데이터

실제 고장/정비 라벨이 부족하여 가상 이상 패턴 생성.

종류:
1. load_rise
2. rpm_instability
3. rpm_error_drift

중요:
- 실제 고장 데이터가 아님
- 물리적으로 설명 가능한 가상 이상 시나리오
- 정상=0 / 가상 이상=1로 지도학습

최종 학습 데이터:
- 4,560 Window
- 14 Feature
- 정상: 1,368
- 가상 이상: 3,192

Split:
- Train: 3,630
- Validation: 370
- Test: 560

---

## 4. 모델 비교

비교 모델:
- Logistic Regression
- Random Forest
- XGBoost

모두 동일:
- 14개 Feature
- 동일 Train/Validation/Test
- 동일한 학습 데이터 기준

Validation 결과:

| Model | Accuracy | F1 | PR-AUC | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.986486 | 0.990291 | 0.999838 | 0.053698 | 0.012469 |
| Random Forest | 1.000000 | 1.000000 | 1.000000 | 0.006400 | 0.000513 |
| XGBoost | 1.000000 | 1.000000 | 1.000000 | 0.002670 | 0.000066 |

최종 선택:
- XGBoost

선택 이유:
- Random Forest와 분류 성능 동률
- XGBoost가 Log Loss / Brier Score에서 더 낮음
- 현재 서비스에서 Anomaly Score를 활용하기 때문에 확률형 출력 품질도 중요
- 소규모 정형 데이터에 적합
- Feature 중요도/기여도 해석 가능

XGBoost:
- Best Iteration: 509

최종 Test 결과:
- Accuracy: 0.996429
- Precision: 0.997449
- Recall: 0.997449
- F1: 0.997449
- PR-AUC: 0.999961
- ROC-AUC: 0.999909
- Log Loss: 0.012655
- Brier: 0.002921

Confusion Matrix:
```text
[[167, 1],
 [  1,391]]
```

주의:
- 실제 고장 정확도 99.64%가 아님
- 가상 이상 시나리오 기반 Test 분류 정확도

---

## 5. Anomaly Score 의미

Anomaly Score는 실제 고장 확률이 아니다.

정확한 의미:
> 현재 1분 구간이 XGBoost가 학습한 가상 이상 패턴과 얼마나 유사한지를 나타내는 점수.

사용 금지 표현:
- 고장 확률
- 실제 고장 가능성 XX%
- N시간 후 고장
- 특정 부품 고장 확정

---

## 6. 실제 31일 로그 검증

지원 Window:
- 19,718

주요 분포:
- 평균: 0.008030
- 중앙값: 0.001130
- P95: 0.018857
- P99: 0.136186
- 최대: 0.999632

최고점 사례:
- 2026-05-29
- Tool 3
- 약 14:00:37 ~ 14:01:29
- Score 약 0.999632

이 사례 직후:
- Spindle 정지
- Tool 3 → Tool 4 교체
- 이후 절삭 재개

즉 높은 Score가 실제 고장뿐 아니라
공구 교체/공정 전환에서도 발생할 수 있음을 확인.

따라서 현재 UI에서 P95/P99는:
- 고장 임계값이 아님
- 실제 31일 로그 분포의 상대적 참고 기준

---

## 7. 저장된 모델

```text
models/
├─ xgboost_model.json
├─ feature_columns.json
└─ train_normal_baseline.csv
```

---

## 8. 추론 파이프라인

현재 핵심 파일:
```text
src/inference_pipeline.py
```

주요 함수:
```text
load_model_artifacts
parse_raw_log
prepare_clean_data
extract_cutting_segments
build_one_minute_windows
filter_valid_windows
calculate_base_features
apply_tool_baseline
predict_windows
run_inference
```

중요:
- React 개발 중에는 이 추론 로직을 가급적 변경하지 않는다.
- UI 기능 추가 때문에 모델 Feature/Window/Baseline 로직을 바꾸지 않는다.

---

## 9. FastAPI

현재 파일:
```text
api/main.py
```

실행:
```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --reload
```

주소:
```text
http://127.0.0.1:8000
```

Swagger:
```text
http://127.0.0.1:8000/docs
```

현재 `/predict`:
- `.log` 단일 파일 분석 지원
- Streamlit에서 정상 동작 확인

반환 데이터 주요 항목:
```text
filename
parsed_rows
clean_rows
cutting_rows
cutting_ratio
result_windows
supported_windows
unsupported_windows
insufficient_windows
score_mean
score_max
highest_window
top_windows
top_contributions
score_series
used_tool_count
segment_count
sampling_interval_mean
file_start
file_end
tool_summary
cutting_segments
```

향후 React 연결 시:
- CORS 설정 필요

향후 추가 예정:
- `.zip` 다중 로그 분석 API

---

## 10. Streamlit

현재 역할:
- 내부 검증용
- AI 추론 파이프라인 확인
- 비상 시 간단 시연용

실행:
```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

최종 서비스 화면은 React로 별도 구축 예정.

---

## 11. React 최종 서비스 목표

React 프론트는 실제로 동작하는 기능만 표시한다.

구현할 기능:
1. 새 분석
2. 단일 `.log` 업로드
3. AI 설비 진단 결과
4. 업로드 데이터 분석
5. 여러 `.log`가 들어 있는 `.zip` 월간 일괄 분석
6. 분석 결과 PDF 다운로드

구현하지 않을 기능:
- 로그인
- 회원가입
- 알림
- 설정
- 설비 관리
- 가짜 분석 이력
- 동작하지 않는 메뉴

---

## 12. React 서비스 화면

### 새 분석 화면
```text
MCT Insight

파일 업로드
- .log 단일 분석
- .zip 다중 로그 분석

[분석 시작]
```

### 단일 로그 결과
```text
분석 결과
├─ 참고 수준
├─ 최대 Anomaly Score
├─ 파일 내 최고 Score 구간
├─ 분석 요약
├─ 무엇이 평소와 달랐는가
├─ 현장 점검 가이드
├─ 시간대별 Score 추이
├─ 최고 Score 구간
├─ 업로드 데이터 요약
├─ 공구별 사용 현황
└─ 절삭 Timeline

[PDF 보고서 다운로드]
```

### ZIP 월간 결과
```text
분석 파일 수
분석 기간
전체 측정 데이터
실제 절삭 데이터
월간 최대 Score
일별 Score 추이
월간 최고 Score 구간
공구별 사용 현황
월간 공정 흐름
```

---

## 13. PDF 보고서 예정 구성

```text
MCT AI 설비 분석 보고서

1. 분석 대상
2. 데이터 요약
3. AI 분석 요약
4. 주요 변화
5. Anomaly Score 추이
6. 최고 Score 구간
7. 현장 점검 권장 사항
8. 분석 한계 / 주의사항
```

중요:
- PDF에서도 Anomaly Score를 고장 확률이라고 표현하지 않는다.

---

## 14. React 기술 스택

예정:
```text
React
Vite
TypeScript
Tailwind CSS
Recharts
```

Node:
- v24.12.0

npm:
- 11.6.2

---

## 15. 현재 프로젝트 구조 핵심

```text
MCT-Data-Preprocessing/
├─ api/
│  └─ main.py
├─ data/
│  └─ raw/
├─ models/
├─ notebooks/
├─ output/
├─ reports/
├─ scripts/
├─ src/
│  └─ inference_pipeline.py
├─ streamlit_app.py
└─ README.md
```

React 생성 후:
```text
MCT-Data-Preprocessing/
├─ frontend/
│  └─ React 프로젝트
├─ api/
├─ src/
├─ models/
└─ ...
```

---

## 16. 지금 바로 다음 작업

### STEP 1
프로젝트 루트에서 React + Vite + TypeScript 프로젝트 생성.

예정 폴더:
```text
frontend/
```

### STEP 2
React 기본 실행 확인.

### STEP 3
불필요한 Vite 기본 화면 제거 후 서비스 기본 레이아웃 제작.

### STEP 4
FastAPI CORS 추가.

### STEP 5
단일 `.log` 업로드 → 기존 `/predict` 연결.

### STEP 6
실제 분석 결과 UI 구현.

### STEP 7
`.zip` 일괄 분석 API 및 React UI 추가.

### STEP 8
PDF 다운로드 추가.

### STEP 9
최종 교수님/기업 시연 검증.
