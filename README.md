# MCT Data Preprocessing
MCT 설비에서 수집된 원본 로그를 분석 가능한 CSV 형태로 변환하고, 데이터 품질을 검증하여 예지보전 분석용 데이터를 생성하는 프로젝트

## 목적
- MCT 원본 로그 구조 분석
- 날짜·시간 및 수치형 데이터 변환
- 결측치, 중복값, 수집 간격 및 특이값 검증
- 알람·비상 상태·과부하·수집 공백 플래그 생성
- 예지보전 모델링에 사용할 최종 데이터 생성
- 전처리 및 품질 검증 결과 문서화

## 프로젝트 구조
```text
MCT-Data-Preprocessing/
├── data/                         # 원본 로그 저장 폴더
├── output/                       # 파싱 및 최종 정제 데이터
├── reports/                      # 데이터 품질 검증 결과
│   ├── abnormal_intervals.csv
│   ├── alarm_counts.csv
│   ├── daily_counts.csv
│   ├── interval_summary.csv
│   ├── machine_counts.csv
│   ├── missing_counts.csv
│   ├── numeric_summary.csv
│   └── status_counts.csv
├── src/
│   ├── parse_mct_log.py          # 원본 로그 파싱
│   ├── validate_mct_data.py      # 데이터 품질 검증
│   └── build_clean_dataset.py    # 최종 분석용 데이터 생성
├── .gitignore
└── README.md
```

## 데이터 보안
기업에서 제공받은 원본 데이터와 전처리 결과는 GitHub에 업로드하지 않음.

- `data/`: 기업에서 제공받은 원본 로그
- `output/`: 전처리 과정에서 생성된 대용량 결과 데이터

위 폴더의 데이터 파일은 `.gitignore`를 통해 Git 추적에서 제외. 품질 검증 결과인 `reports/`의 요약 CSV만 저장소에서 관리.

## 실행 환경
- Python 3
- pandas

pandas가 설치되지 않은 경우 다음 명령으로 설치.

```powershell
pip install pandas
```

## 실행 방법
프로젝트 루트 디렉터리에서 다음 순서로 실행.

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

## 품질 검증 결과
총 481,648행을 검증한 결과는 다음과 같음.

- 파싱 후 행 수: 481,648행
- 최종 데이터 행 수: 481,648행
- 완전 중복 행: 0건
- 시간 변환 실패: 0건
- 알람 발생: 16행
- 비상 상태 코드 발생: 4행
- 스핀들 부하 100% 초과: 60행
- 수집 간격 15초 초과 또는 0초 이하: 17행
- 최종 관심 관측치: 97행

`error_message`는 알람이 없는 481,632행에서 비어 있음, 이는 오류 메시지가 존재하지 않는 정상적인 결측. 
`interval_seconds`의 첫 번째 관측치도 이전 측정값이 없어 결측으로 기록됨.

## 처리 원칙
특이값을 임의로 삭제하거나 수정하지 않고 원본 관측치를 보존함. 다음 분석용 플래그를 최종 데이터에 추가함.

| 열 이름 | 의미 |
|---|---|
| `interval_seconds` | 같은 설비의 이전 관측치와의 시간 차이 |
| `is_alarm` | 알람 코드 발생 여부 |
| `is_emergency` | 비상 상태 코드 발생 여부 |
| `is_spindle_overload` | 스핀들 부하 100% 초과 여부 |
| `is_abnormal_interval` | 수집 간격 15초 초과 여부 |
| `has_event` | 하나 이상의 관심 조건 해당 여부 |

`aut_mode`, `emergency`, 스핀들 부하 100% 초과 값은 파싱 오류로 확정할 근거가 없어 원본 그대로 보존함.

X·Y·Z축 부하값은 전체 행에서 동일하지만, 원본 로그 대조 결과에서도 같은 값으로 기록되어 있어 파싱 오류가 아닌 원천 데이터의 특성으로 판단함.

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