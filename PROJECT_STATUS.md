# PROJECT_STATUS.md

## 프로젝트
MCT 설비 데이터 기반 AI 예지보전 프로젝트

현재 목표:
- MCT 원본 `.log` / `.zip` 데이터를 업로드하여 설비 상태를 분석
- 비지도학습 M2*와 지도학습 XGBoost를 함께 활용
- Streamlit을 최종 시연용 통합 대시보드로 사용
- FastAPI를 통해 두 모델의 분석 요청을 통합 처리
- 실제 동작하는 기능만 제공
- 기업/교수님 시연이 가능한 서비스형 PoC 수준으로 완성

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

---

## 2. XGBoost 지도학습

### Feature Engineering
1분 Window 사용.

이유:
- 약 5~6초 샘플링
- 1분에 약 10~11개 센서값
- 순간값보다 구간의 변화 패턴을 보기 위함

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

### 공구별 정상 Baseline
전체 평균이 아니라 공구별 정상 Baseline 사용.

지원 공구:
```text
[1,2,3,4,5,6,8,10,11,12,13,14,18,19]
```

일부 공구는 Train 정상 Baseline 부족으로 분석 제외.

---

## 3. XGBoost 가상 이상 데이터

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

## 4. 지도학습 모델 비교

비교 모델:
- Logistic Regression
- Random Forest
- XGBoost

모두 동일:
- 14개 Feature
- 동일 Train/Validation/Test
- 동일 학습 데이터

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
- Anomaly Score 활용 가능
- 정형 데이터에 적합
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

## 5. XGBoost Anomaly Score 의미

Anomaly Score는 실제 고장 확률이 아니다.

정확한 의미:
> 현재 1분 구간이 XGBoost가 학습한 가상 이상 패턴과 얼마나 유사한지를 나타내는 점수.

사용 금지 표현:
- 고장 확률
- 실제 고장 가능성 XX%
- N시간 후 고장
- 특정 부품 고장 확정

---

## 6. 실제 로그 XGBoost 검증

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

따라서 높은 Score는 실제 고장뿐 아니라
공구 교체/공정 전환에서도 발생할 수 있다.

P95/P99:
- 고장 임계값이 아님
- 실제 로그 Score 분포를 기준으로 한 상대적 참고값

---

## 7. M2* 비지도학습

M2*는 실제 고장 라벨이 부족한 현재 데이터 특성을 고려하여
정상 운전조건에서 얼마나 벗어났는지를 감시하는 방식이다.

핵심:
- Tool × RPM Band 기준 정상 Baseline
- Robust z-score
- EWMA
- Persistence
- 보조적으로 Isolation Forest 사용
- 실제 절삭 데이터만 모델 판단에 사용

저장된 정상 기준:
```text
mct_poc_dashboard_common/dashboard/data/model_artifacts.joblib
```

업로드된 새 데이터로 모델을 다시 학습하지 않고,
저장된 정상 기준으로 추론한다.

### 현재 M2* 재현 결과
2026년 5월 전체 로그 기준:
- 전체 데이터: 481,648행
- 실제 절삭: 219,195행
- M2* 이상 알람: 29건
- 신규 운전조건: 4건
- 설비 이벤트: 2건

### 모델 개발용 합성 벤치마크
- 지속 과부하: 60/60 탐지
- 공구 파손형: 53/60 탐지
- 완만한 저하: 60/60 탐지
- 정상 시험구간 알람 후보 빈도: 약 1.4건/일

주의:
- 실제 고장 정확도가 아님
- 모델 특성을 확인하기 위한 합성 이상 실험 결과

---

## 8. M2* / XGBoost 비교

두 모델은 같은 문제를 다른 방식으로 본다.

### M2*
질문:
> 정상 운전조건에서 얼마나 벗어났는가?

특징:
- 정상 데이터 중심
- 지속적인 변화 탐지
- 학습하지 않은 이상 형태도 정상 기준 이탈 관점으로 확인 가능

### XGBoost
질문:
> 현재 1분 Window가 학습한 가상 이상 패턴과 얼마나 유사한가?

특징:
- 정상 + 가상 이상 데이터로 학습
- 특정 학습 이상 패턴에 반응
- Feature Contribution으로 판단 근거 확인 가능

### 현재 활용 제안
- **M2***: 주 분석 방식
- **XGBoost**: 보조 분석 방식

실제 고장 라벨이 없으므로:
- 탐지 후보가 많다고 더 정확한 모델이 아님
- 실제 Precision / Recall / 현장 고장 Accuracy 계산 불가
- 실제 고장 라벨 확보 후 재검증 필요

---

## 9. 최종 서비스 구조

```text
사용자
  ↓
Streamlit
  ↓
FastAPI
  ├─ /predict/m2  → M2*
  └─ /predict     → XGBoost
        ↓
      분석 결과
        ↓
     Streamlit
```

### Streamlit
- 파일 업로드
- 모델 선택
- 분석 결과 시각화
- 운영 리포트
- 모델 비교
- PDF 리포트

### FastAPI
- M2* 분석 요청 처리
- XGBoost 분석 요청 처리
- 공통 백엔드 진입점

---

## 10. FastAPI

현재 파일:
```text
api/main.py
```

기본 주소:
```text
http://127.0.0.1:8000
```

주요 Endpoint:
```text
GET  /health
POST /predict
POST /predict/m2
```

### `/predict`
XGBoost 분석:
- `.log`
- `.zip`
- 공통 전처리
- 1분 Window
- Anomaly Score
- Window 상세 정보
- Feature Contribution
- 공구별 분석 가능 범위

### `/predict/m2`
M2* 분석:
- `.log`
- `.zip`
- 공통 전처리
- 저장된 정상 Baseline 기반 추론
- 알람 / 신규조건 / 이벤트 / 운영 데이터 반환

---

## 11. Streamlit 통합 대시보드

현재 메인 파일:
```text
mct_poc_dashboard_common/dashboard/streamlit_integrated.py
```

상단 분석 메뉴:
```text
비지도학습 M2*
지도학습 XGBoost
모델 비교
```

### M2* 화면
- `.log` / `.zip` 업로드
- 실시간 관제 형태의 로그 리플레이
- 이상 알람
- 운영 리포트
- 공구 사용 분석
- 데이터 품질 안내
- 모델 개발 벤치마크

### XGBoost 화면
- `.log` / `.zip` 업로드
- AI 분석 요약
- 최고 Anomaly Score
- 1분 Window 상세 조회
- Feature Contribution
- 시간대별 Score
- 공구별 분석
- PDF 보고서

### 모델 비교 화면
- 실제 로그 반응 비교
- 동일 합성 이상 시나리오 비교
- 모델별 탐지 특성 정리
- M2* 주 분석 / XGBoost 보조 분석 제안

---

## 12. 통합 실행

현재 실행 파일:
```text
run_app.py
```

실행:
```powershell
.\.venv\Scripts\Activate.ps1
python run_app.py
```

동작:
1. FastAPI 실행
2. `/health` 확인
3. Streamlit 실행
4. 브라우저에서 통합 대시보드 사용

종료:
```text
Ctrl + C
```

현재 확인 완료:
- M2* 화면 정상
- XGBoost 화면 정상
- 모델 비교 화면 정상
- M2* FastAPI 분석 정상
- XGBoost FastAPI 분석 정상
- XGBoost 1분 Window 상세 조회 정상

---

## 13. 저장 모델

### XGBoost
```text
models/
├─ xgboost_model.json
├─ feature_columns.json
└─ train_normal_baseline.csv
```

### M2*
```text
mct_poc_dashboard_common/dashboard/data/
└─ model_artifacts.joblib
```

---

## 14. 주요 추론 코드

### 공통 / XGBoost
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

### M2*
```text
mct_poc_dashboard_common/dashboard/
├─ m2_upload_pipeline.py
└─ pipeline_core.py
```

중요:
- UI 정리 때문에 모델 Feature / Window / Baseline 로직을 임의로 변경하지 않는다.
- 모델 재학습이 필요한 변경은 기존 추론 코드 수정과 분리한다.

---

## 15. 데이터 / EDA / 실험 자료 보존

기존 분석 및 팀원 실험 자료는 최종 서비스와 별도로 보존한다.

```text
notebooks/
reports/eda/
reports/model_eda/
reports/model/
reports/model_comparison/
reports/synthetic/
mct_poc_dashboard_common/dashboard/experiments/
mct_poc_dashboard_common/dashboard/reports/
```

이 자료들은:
- EDA
- 모델 개발 과정
- 모델 비교
- 합성 이상 실험
- M2* 후보 방식 검증
- 결과 재현

에 사용되므로 임의 삭제하지 않는다.

---

## 16. 현재 프로젝트 구조 핵심

```text
MCT-Data-Preprocessing/
│
├─ run_app.py
├─ README.md
├─ PROJECT_STATUS.md
│
├─ api/
│  └─ main.py
│
├─ src/
│  ├─ inference_pipeline.py
│  ├─ parse_mct_log.py
│  ├─ build_clean_dataset.py
│  ├─ validate_mct_data.py
│  └─ eda_mct.py
│
├─ models/
│  ├─ xgboost_model.json
│  ├─ feature_columns.json
│  └─ train_normal_baseline.csv
│
├─ mct_poc_dashboard_common/
│  └─ dashboard/
│     ├─ streamlit_integrated.py
│     ├─ xgb_view.py
│     ├─ comparison_view.py
│     ├─ m2_upload_pipeline.py
│     ├─ pipeline_core.py
│     ├─ regenerate.py
│     ├─ data/
│     ├─ experiments/
│     └─ reports/
│
├─ data/
│  └─ raw/
│
├─ output/
├─ notebooks/
├─ reports/
├─ scripts/
├─ web_report/
└─ archive/
```

`archive/`:
- 현재 서비스 실행에는 사용하지 않는 구버전/백업 파일 보관
- 삭제 전 보존 목적

---

## 17. 현재 남은 작업

### 완료
- 데이터 전처리 / EDA
- XGBoost 학습 및 검증
- M2* 구현 및 검증
- 공통 전처리 적용
- M2* `.log` / `.zip` 분석
- XGBoost `.log` / `.zip` 분석
- 통합 FastAPI
- 통합 Streamlit
- 모델 비교 화면
- 운영 리포트
- XGBoost Feature Contribution
- PDF 보고서
- `run_app.py` 통합 실행
- 기본 UI 통일
- 구버전 파일 archive 정리
- README 최신화

### 남은 작업
1. 최종 폴더 구조 및 불필요 파일 점검
2. README / PROJECT_STATUS 최종 확인
3. 전체 `.log` / `.zip` 시연 테스트
4. 발표용 시나리오 정리
5. 발표 자료 및 결과 설명 문구 정리
6. 최종 Git commit / push

---

## 18. 발표 시 핵심 설명

프로젝트 핵심:
> 실제 고장 라벨이 부족한 MCT 로그 환경에서 정상 데이터 기반 비지도 이상 탐지와 가상 이상 기반 지도학습을 함께 구현하고 비교하였다.

모델 역할:
> M2*는 정상 운전조건에서 벗어나는 지속적인 변화를 감시하고, XGBoost는 학습한 이상 패턴과 유사한 1분 구간을 분석한다.

서비스 구조:
> 사용자가 Streamlit에서 로그를 업로드하면 FastAPI가 분석 요청을 받아 M2* 또는 XGBoost 추론을 수행하고 결과를 다시 대시보드에 시각화한다.

주의:
> 현재 실제 고장 정답 라벨이 부족하므로 모델 출력은 실제 고장 확률이나 미래 고장 시점으로 해석하지 않는다.
