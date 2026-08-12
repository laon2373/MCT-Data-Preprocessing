# MCT Data Preprocessing
MCT 설비에서 수집된 원본 로그를 분석 가능한 CSV 형태로 변환하고, 데이터 품질을 검증하여 예지보전 분석용 데이터를 생성하는 프로젝트

## 목적
- MCT 원본 로그 구조 분석
- 날짜·시간 및 수치형 데이터 변환
- 결측치, 중복값, 수집 간격 및 특이값 검증
- 알람·비상·RESET·과부하·수집 공백 플래그 생성
- 설비 좌표를 실제 mm 단위로 변환
- 최종 분석용 데이터 생성
- 탐색적 데이터 분석(EDA) 수행
- 예지보전 모델링에 사용할 데이터 제공
- 전처리 및 분석 결과 문서화

## 프로젝트 구조

```text
MCT-Data-Preprocessing/
├── data/                         # 원본 로그 저장 폴더
├── output/                       # 파싱 및 최종 정제 데이터
├── reports/                      # 데이터 품질 검증 및 EDA 결과
│   ├── abnormal_intervals.csv
│   ├── alarm_counts.csv
│   ├── daily_counts.csv
│   ├── interval_summary.csv
│   ├── machine_counts.csv
│   ├── missing_counts.csv
│   ├── numeric_summary.csv
│   ├── status_counts.csv
│   └── eda/                      # EDA 결과 저장 폴더
├── src/
│   ├── parse_mct_log.py          # 원본 로그 파싱
│   ├── validate_mct_data.py      # 데이터 품질 검증
│   ├── build_clean_dataset.py    # 최종 분석용 데이터 생성
│   └── eda_mct.py                # 탐색적 데이터 분석
├── .gitignore
└── README.md
```

## 데이터 보안
기업에서 제공받은 원본 데이터와 전처리 결과는 GitHub에 업로드하지 않음.

- `data/`: 기업에서 제공받은 원본 로그
- `output/`: 전처리 과정에서 생성된 대용량 결과 데이터

위 폴더의 데이터 파일은 `.gitignore`를 통해 Git 추적에서 제외함. 기업 데이터에서 생성된 분석 결과 파일도 필요한 경우 별도로 관리함.

## 실행 환경
- Python 3
- pandas
- matplotlib
- seaborn

필요한 라이브러리가 설치되지 않은 경우 다음 명령으로 설치함.

```powershell
pip install pandas matplotlib seaborn
```

## 실행 방법
프로젝트 루트 디렉터리에서 다음 순서로 실행함.

### 1. 원본 로그 파싱

```powershell
python .\src\parse_mct_log.py
```

생성 파일:

```text
output/mct_parsed.csv
```

### 2. 데이터 품질 검증

```powershell
python .\src\validate_mct_data.py
```

생성 결과:

```text
reports/
```

### 3. 최종 분석용 데이터 생성

```powershell
python .\src\build_clean_dataset.py
```

생성 파일:

```text
output/mct_clean.csv
```

### 4. 탐색적 데이터 분석(EDA)

```powershell
python .\src\eda_mct.py
```

`mct_clean.csv`를 기준으로 데이터 구조, 결측치, 주요 변수의 분포, 이벤트 발생 현황, 정상·이벤트 구간 비교 및 변수 간 상관관계를 분석함.

생성 결과:

```text
reports/eda/
```

주요 결과 파일:

- `basic_info.csv`: 열 자료형 및 결측치 정보
- `numeric_summary.csv`: 주요 수치형 변수의 기초 통계
- `event_summary.csv`: 이벤트별 발생 횟수
- `daily_event_summary.csv`: 날짜별 이벤트 현황
- `event_comparison.csv`: 정상 구간과 이벤트 구간 비교
- `correlation.csv`: 수치형 변수 간 상관관계
- `.png` 파일: 변수 분포, 이벤트 비교 및 상관관계 그래프

## 품질 검증 결과
총 481,648행을 검증한 결과는 다음과 같음.

- 파싱 후 행 수: 481,648행
- 최종 데이터 행 수: 481,648행
- 완전 중복 행: 0건
- 시간 변환 실패: 0건
- 알람 발생: 16행
- 실제 비상 상태 발생: 0행
- RESET 상태 발생: 4행
- 스핀들 부하 100% 초과: 60행
- 수집 간격 15초 초과 또는 0초 이하: 17행
- 최종 관심 관측치: 97행

`error_message`는 알람이 없는 481,632행에서 비어 있으며, 이는 오류 메시지가 존재하지 않는 정상적인 결측임. `interval_seconds`는 각 설비의 첫 번째 관측치에서 이전 측정값이 없어 결측으로 기록됨.

## 처리 원칙
특이값을 임의로 삭제하거나 수정하지 않고 원본 관측치를 보존함. 분석과 모델링에 필요한 변환 열 및 상태 플래그를 최종 데이터에 추가함.

| 열 이름 | 의미 |
|---|---|
| `x_axis_mm`, `y_axis_mm`, `z_axis_mm` | 스케일 팩터 10000을 적용한 mm 단위 좌표 |
| `interval_seconds` | 같은 설비의 이전 관측치와의 시간 차이 |
| `is_alarm` | 알람 코드 발생 여부 |
| `is_emergency` | 실제 비상 상태 코드 1 여부 |
| `is_reset` | RESET 상태 코드 2 여부 |
| `is_spindle_overload` | 스핀들 부하 100% 초과 여부 |
| `is_abnormal_interval` | 수집 간격 15초 초과 여부 |
| `has_event` | 하나 이상의 관심 조건 해당 여부 |

원본 X·Y·Z 좌표에는 스케일 팩터 10000이 적용되어 있음. 원본 좌표를 추적·검증용으로 보존하면서 실제 mm 단위로 변환한 `x_axis_mm`, `y_axis_mm`, `z_axis_mm` 열을 최종 데이터에 추가함.

`emergency` 상태 코드는 0(비상 아님), 1(실제 비상), 2(RESET)로 구분하여 처리함. 데이터에서 확인된 코드 2의 4개 행은 실제 비상이 아닌 RESET 상태임.

`aut_mode`, `emergency`, 스핀들 부하 100% 초과 값은 파싱 오류로 확정할 근거가 없어 원본값을 그대로 보존하고 별도의 분석용 플래그로 구분함.

X·Y·Z축 부하값은 전체 행에서 동일하지만, 원본 로그 대조 결과에서도 같은 값으로 기록되어 있어 파싱 오류가 아닌 원천 데이터의 특성으로 판단함.

## EDA 수행 범위
EDA에서는 다음 항목을 확인함.

- 데이터 행·열 구조 및 자료형
- 열별 결측치 현황
- 주요 수치형 변수의 최솟값·최댓값·평균·중앙값
- 알람·비상·RESET·과부하·비정상 수집 간격 발생 현황
- 날짜별 데이터 및 이벤트 발생 추이
- 정상 구간과 이벤트 구간의 주요 변수 비교
- 수치형 변수 간 상관관계
- 모델 학습에 사용할 수 있는 후보 변수

`has_event`는 실제 고장 여부를 나타내는 정답 라벨이 아니라, 알람·RESET·과부하·수집 공백 등 분석할 필요가 있는 행을 표시하는 보조 플래그임. 따라서 실제 고장 예측 모델을 학습하려면 별도의 고장·정비 이력이 필요함.

## EDA 주요 결과
- 전체 481,648행 중 `has_event=True`인 관심 관측치는 97행으로, 전체의 약 0.02%임
- `spindle_rate`와 `spindle_load`의 상관계수는 약 0.95로 높은 양의 상관관계를 보임
- `spindle_rate`와 `z_axis_mm`의 상관계수는 약 -0.92로 높은 음의 상관관계를 보임
- `x_axis_mm`와 `y_axis_mm`의 상관계수는 약 0.75로 비교적 높은 양의 상관관계를 보임
- `delta_spindle_rpm`은 대부분 0에 가깝지만 일부 큰 값이 존재하여 이벤트 구간과 정상 구간의 분포 차이가 확인됨
- 이벤트 구간에서는 `spindle_load`의 중앙값과 범위가 정상 구간보다 높게 나타남
- 이벤트 구간의 `feed_rate`와 `spindle_rate`는 정상 구간보다 낮은 경향을 보임
- `has_event`는 알람·RESET·과부하·수집 공백을 합친 보조 플래그이며 실제 고장 라벨은 아님

## 진행 상태
- [v] GitHub 저장소 및 프로젝트 구조 생성
- [v] 원본 및 결과 데이터 Git 추적 제외
- [v] 원본 로그 구조 분석
- [v] 원본 로그 파싱
- [v] 결측치·중복값·특이값 점검
- [v] 원본 로그와 파싱 결과 대조
- [v] 데이터 품질 검증 보고서 생성
- [v] 최종 분석용 데이터 생성
- [v] 전처리 결과 검증
- [v] EDA 코드 작성
- [v] 주요 변수 분포 및 이벤트 현황 분석
- [v] 정상 구간과 이벤트 구간 비교
- [v] 수치형 변수 상관관계 분석