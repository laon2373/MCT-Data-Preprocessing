# MCT Data Preprocessing

MCT 설비에서 수집된 원본 데이터를 분석 가능한 형태로 변환하기 위한 데이터 전처리 프로젝트.

## 목적

- MCT 원본 데이터 구조 파악
- 결측치, 중복값 및 이상값 점검
- 날짜·시간 및 수치형 데이터 변환
- 분석 및 예지보전 모델링에 사용할 데이터 생성
- 전처리 과정과 데이터 품질 검증 결과 기록

## 프로젝트 구조

```text
MCT-Data-Preprocessing/
├── data/                   # 원본 데이터 저장 폴더
├── output/                 # 전처리 결과 저장 폴더
├── src/
│   └── parse_mct_log.py    # 데이터 전처리 코드
├── .gitignore
└── README.md
```

## 데이터 보안

기업에서 제공받은 원본 데이터와 전처리 결과는 GitHub에 업로드하지 않음.

- `data/`: 기업에서 제공받은 원본 데이터
- `output/`: 전처리 과정에서 생성된 결과 데이터

위 폴더의 데이터 파일은 `.gitignore`를 통해 Git 추적에서 제외.

## 실행 환경

- Python 3
- pandas

## 실행 방법

```powershell
python .\src\parse_mct_log.py
```

## 진행 상태

- [x] GitHub 저장소 생성
- [x] 기본 프로젝트 구조 생성
- [x] 원본 및 결과 데이터 Git 추적 제외
- [ ] 원본 데이터 구조 분석
- [ ] 결측치·중복값·이상값 점검
- [ ] 전처리 코드 구현
- [ ] 전처리 결과 검증