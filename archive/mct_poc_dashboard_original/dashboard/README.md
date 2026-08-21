# MCT 3호 예지보전 PoC 대시보드

CNC 머시닝센터(MCT 3호) 2026년 5월 로그(31일, 48만 사이클)를
조건부 잔차 이상탐지 모델(M2*)로 스코어링하고 Streamlit으로 시각화한 산학 PoC입니다.

> 처음 보는 분은 [프로젝트_설명.md](프로젝트_설명.md) 부터 — 비전공자용으로 전체 과정을 풀어 쓴 문서입니다.

## 실행 (3줄)
```bash
pip install -r requirements.txt
cd 이_폴더
streamlit run app.py        # 브라우저가 자동으로 열립니다 (localhost:8501)
```
모델 연산은 전부 사전 계산되어 `data/`에 들어 있으므로 노트북에서도 즉시 구동됩니다.

## 화면 구성
- **① 실시간 관제 (리플레이)** — 5월 로그를 실시간처럼 재생. 부하/RPM, 컨텍스트 기대범위 밴드,
  가감속 구간 표시, 알람 시작 지점·지속 구간 마킹. 그라파나식 관제 화면의 PoC 버전.
- **② 이상 알람** — 이상 점수 타임라인(순간 |z| / 드리프트 EWMA, 임계값), 알람 22건 목록과
  설명("T01·9k rpm: 기대 7% 대비 부하 높음 98초 지속"), 알람별 상세(요약 카드 + 전후 차트).
  신규 운전조건 알림(이상 아님, 별도 채널)과 설비 자체 이벤트 원본 포함.
- **③ 월간 운영 리포트** — 가동률(일별·시간대별), 공구 사용, 데이터 품질 진단과 개선 제안
  (로거 이슈 2건, 감속 제동 스파이크 69건의 정체, 일요일 03시 공백 등).
- **④ 분석 방법과 신뢰성** — 7개 후보 동일조건 벤치마크(오탐/일 + 합성 이상 주입 탐지율)와 채택 근거.
  수치는 하드코딩이 아니라 `experiments/benchmark.py`가 산출한 `data/bench.csv`를 읽는다.
  화면 문구는 데이터를 제공한 회사 담당자가 읽는 것을 전제로 작성 (모델 전문용어는 이 탭에만 최소한으로).

## 데이터 재생성 (선택)
원본 `.log` 폴더에서 파싱→학습→스코어링→`data/` 생성까지 처음부터 다시 돌리려면:
```bash
python regenerate.py <로그폴더>
python regenerate.py <로그폴더> --save-model   # + api.py 서빙용 data/model_artifacts.joblib 저장
```
이 스크립트가 곧 모델 코드입니다: 정상/가감속 상태 분리 → 공구×RPM밴드 컨텍스트별
robust 통계(앞 3주 학습) → 잔차 z + 30초 지속성 + 조건별 EWMA + IsolationForest 보조 채널.
수식의 단일 소스는 `pipeline_core.py`이며 regenerate.py(배치)와 api.py(실시간)가 함께 임포트합니다.

## 실시간 예측 API (선택)
```bash
uvicorn api:app --port 8000        # 대화형 문서: http://localhost:8000/docs
```
- `POST /predict` — 한 사이클 센서 값 → `state`·`zload`·`novel`·`instant_exceed`(순간 임계 초과 여부)·기준통계.
  **알람 확정은 반환하지 않습니다**: 순간 채널은 6사이클(≈30초) 연속 지속이 조건이고
  드리프트(EWMA) 채널은 이력이 필요해, 단일 사이클로는 판정이 원리적으로 불가하기 때문입니다.
- `POST /predict_batch` — 시간순 최근 N사이클 배열 → 지속성·EWMA까지 적용한 **알람 확정** 반환
  (EWMA는 배열 안에서만 누적되므로 60사이클 이상 이력 권장).
- novel(학습되지 않은 공구×RPM 조합)은 이상이 아니라 "판단 불가"로, 응답에 그 사실이 명시됩니다.
- 대시보드(app.py)는 API를 임포트하지 않으며, API 없이도 단독 구동됩니다.

## 실험 (계획서 산출물 — experiments/)
모두 `data/scored.csv.gz`에서 출발하므로 원본 로그 없이 재실행 가능, seed 고정.
```bash
python experiments/benchmark.py    # 탭④ 벤치마크 재실험 → data/bench.csv
python experiments/lstm_ae.py      # LSTM-AE 학습(CPU ~2분) → data/lstm_ae.pt · reports/lstm_ae.md
python experiments/classifier.py   # 상태 4분류 → reports/classifier.md
python experiments/fairness_check.py  # M2*·조건부IF 임계값 held-out 분리보정 검증 → reports/fairness_check.md
```
lstm_ae 실행 후 benchmark를 재실행하면 LSTM-AE가 7번째 후보로 표에 편입됩니다.
fairness_check는 data/bench.csv를 건드리지 않는 후속 검증용이며, 대시보드 표시값과는 무관하다.

## 구조
```
app.py            # 대시보드 (모델 연산 없음, data/만 읽음 — 실험·API 의존성 불필요)
regenerate.py     # 원본 로그 → data/ 전체 파이프라인 (--save-model로 API 산출물 저장)
pipeline_core.py  # M2* 수식 공용 모듈 (regenerate·api·experiments가 임포트)
api.py            # FastAPI 실시간 예측 API (data/model_artifacts.joblib 필요)
experiments/      # benchmark.py · lstm_ae.py · classifier.py · fairness_check.py (계획서 산출물 실험 + 공정성 후속 검증)
reports/          # lstm_ae.md · classifier.md · benchmark_conditions.md · benchmark_k_sweep.md · fairness_check.md
data/             # 사전 계산 결과 (scored.csv.gz, alarms.csv, novel.csv, events.csv,
                  #   daily.csv, meta.json, bench.csv, lstm_ae.pt, model_artifacts.joblib)
```

## 한계 (발표 시 명시 권장)
- 임계값은 5/1~21 학습 기준. 실제 운영이라면 주 단위 재베이스라이닝 필요(드리프트 채널 오탐 관리).
- 합성 이상 주입으로 검증한 탐지율이며 실제 고장 라벨 검증이 아님.
- 축별 부하 복제 버그·알람코드 미기록 때문에 축 계통 진단과 알람 원인 분석은 현 데이터로 불가.
- 탭 ④ 수치는 `experiments/benchmark.py` 재실험 산출값(`data/bench.csv`)으로 교체되었다.
  주입 위치가 확률적이고, 주입 창을 학습된 운전조건(known≥80%)으로 제한하는 프로토콜을
  명문화했기 때문에 이전 세션의 하드코딩 수치와 절대값은 다르다. 후보 간 순위·경향은 재현되었고
  (채택 방식만 3유형 전부 탐지, SPC는 오탐 최저지만 공구 파손 0%, 전역 ML류는 오탐 과다),
  학습 통계 재현 검증(스크립트 콘솔 출력)에서 알람 채널이 저장 데이터와 완전 일치함을 확인했다.
- `reports/benchmark_conditions.md`는 "임계값 분리(D·C)는 전역 후보 4종에만 적용됐고 M2*·조건부IF는
  검증 안 됨"이라는 한계를 남겼는데, `reports/fairness_check.md`에서 M2*·조건부IF에도 동일한
  held-out 분리보정을 적용해 봤다. 임계값(thr_z)이 사실상 그대로(차이 0.00%) 나오고 탐지율도
  동일해, 채택 방식의 우위가 in-sample 임계값 착시가 아님을 확인했다.
