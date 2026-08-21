# MCT 예지보전 AI 프로젝트

MCT 설비 로그를 분석하여 이상 징후를 탐지하는 예지보전 PoC 프로젝트입니다.

비지도학습 **M2\***와 지도학습 **XGBoost**를 함께 구현했으며,  
사용자는 Streamlit 대시보드에서 `.log` 또는 `.zip` 파일을 업로드하여 분석할 수 있습니다.

---

## 1. 최종 서비스 구조

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

- **Streamlit**: 사용자 화면, 로그 업로드, 결과 시각화
- **FastAPI**: 분석 요청 처리
- **M2\***: 정상 운전조건에서 벗어나는 지속적인 변화 탐지
- **XGBoost**: 학습한 가상 이상 패턴과 유사한 1분 절삭 Window 분석
- **모델 비교**: 실제 로그 반응과 동일 합성 이상 실험 결과 비교

> 실제 설비 고장 정답 라벨이 부족하므로 XGBoost의 Anomaly Score를 실제 고장 확률로 해석하지 않습니다.

---

## 2. 실행 방법

프로젝트 최상위 폴더에서 가상환경을 활성화한 뒤 실행합니다.

```powershell
.\.venv\Scripts\Activate.ps1
python run_app.py
```

`run_app.py`가 다음을 순서대로 실행합니다.

1. FastAPI 실행
2. `/health` 연결 확인
3. Streamlit 통합 대시보드 실행

종료할 때는 실행한 터미널에서 `Ctrl + C`를 누릅니다.

---

## 3. 주요 기능

### 비지도학습 M2*

- `.log`, `.zip` 업로드
- 공통 전처리
- 저장된 정상 기준을 이용한 추론
- 이상 알람 확인
- 신규 운전조건 확인
- 실시간 관제 형태의 로그 리플레이
- 운영 리포트
- 공구별 운전 분석

### 지도학습 XGBoost

- `.log`, `.zip` 업로드
- 1분 절삭 Window 생성
- Anomaly Score 계산
- 점수가 높은 시간대 확인
- Feature Contribution 확인
- 공구별 분석 가능 범위 확인
- PDF 리포트

### 모델 비교

- 공통 전처리 기준 실제 로그 반응 비교
- 동일 합성 이상 시나리오 비교
- M2*와 XGBoost의 탐지 특성 비교

실제 로그에는 고장 정답 라벨이 없으므로  
**탐지 후보가 많다고 더 정확한 모델이라는 의미는 아닙니다.**

---

## 4. 주요 폴더 구조

```text
MCT-Data-Preprocessing/
│
├─ run_app.py
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

---

## 5. 데이터 분석 및 실험 자료

초기 EDA, 모델 개발 및 검증 과정은 최종 서비스 코드와 별도로 보존합니다.

- `notebooks/` : EDA, Feature Engineering, 합성 이상 생성, 모델 학습 및 비교
- `reports/eda/` : EDA 결과
- `reports/model/` : 지도학습 모델 평가 결과
- `reports/model_comparison/` : M2* / XGBoost 비교 결과
- `reports/synthetic/` : 합성 이상 예시
- `mct_poc_dashboard_common/dashboard/experiments/` : M2* 후보 모델 및 벤치마크 실험
- `mct_poc_dashboard_common/dashboard/reports/` : M2* 실험 보고서

`archive/`에는 현재 서비스 실행에는 사용하지 않는 구버전/백업 파일을 보관합니다.

---

## 6. API

FastAPI 기본 주소:

```text
http://127.0.0.1:8000
```

주요 Endpoint:

```text
GET  /health
POST /predict
POST /predict/m2
```

- `/predict` : XGBoost 분석
- `/predict/m2` : M2* 분석

---

## 7. 모델 해석 주의사항

현재 데이터에는 실제 고장 라벨이 충분하지 않습니다.

따라서 다음 표현은 사용하지 않습니다.

- 실제 고장 확률
- N시간 후 고장 예측
- 실제 제품 불량 확률
- 실제 현장 고장 Accuracy

XGBoost의 출력은 **Anomaly Score(이상 패턴 점수)**로 해석하며,  
M2*는 **정상 운전조건에서 벗어나는 정도와 지속성**을 중심으로 해석합니다.

실제 고장 라벨이 확보되면 두 모델의 현장 성능을 다시 검증해야 합니다.
