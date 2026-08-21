# MCT 예지보전 Streamlit UI 개선 구현 가이드

> 목적: 채팅에서 긴 코드를 반복해서 주고받지 않고, 이 문서 하나를 기준으로 Streamlit 화면 개선을 단계적으로 진행한다.
>
> 현재 기준일: 2026-08-21

---

## 1. 현재 상태

현재 프로젝트의 기본 흐름은 아래와 같다.

```text
Streamlit
   ↓
FastAPI /predict
   ↓
inference_pipeline.py
   ↓
XGBoost
   ↓
JSON 결과
   ↓
Streamlit 결과 표시
```

현재 `streamlit_app.py`는 다음 기능까지 정상 동작한다.

- `.log` 파일 업로드
- FastAPI `/predict` 호출
- 분석 결과를 `st.session_state["analysis_result"]`에 저장
- 전체 측정 행 표시
- 실제 절삭 행 표시
- 분석 가능 Window 표시
- 최대 Anomaly Score 표시
- 최고점 Window 표시
- 정상 대비 지표 표시
- XGBoost 주요 기여 Feature 표시
- Anomaly Score 상위 10개 Window 표시

즉 **분석 기능은 이미 동작하고 있으며, 지금부터는 사용자 관점의 화면 구조를 개선하는 것이 핵심**이다.

---

# 2. 화면 개선의 핵심 원칙

현재 화면은 분석 결과를 개발자 관점에서 순서대로 보여준다.

```text
전체 행
→ 분석 Window
→ 최고점 Window
→ 정상 대비 지표
→ Feature Contribution
→ Top 10
```

최종 화면은 아래처럼 사용자 관점으로 바꾼다.

```text
무슨 상태인가?
        ↓
어느 구간을 확인해야 하나?
        ↓
왜 그 구간을 확인해야 하나?
        ↓
사용자가 무엇을 확인하면 되나?
        ↓
필요하면 상세 데이터를 확인
```

### 사용자에게 직접 보여주지 않을 표현

다음 내부 용어는 메인 화면에서는 가능한 한 숨긴다.

- `supported_windows`
- `unsupported_windows`
- `insufficient_windows`
- `load_vs_normal`
- `rpm_std_vs_normal`
- `rpm_error_vs_normal`
- `rpm_error_absmax`

대신 사람이 이해하기 쉬운 표현으로 바꾼다.

예:

```text
supported_windows
→ AI 분석 가능 구간

insufficient_windows
→ 데이터가 부족해 분석하지 못한 구간

load_vs_normal
→ 평소 대비 Spindle Load

rpm_std_vs_normal
→ 평소 대비 RPM 흔들림

rpm_error_vs_normal
→ 평소 대비 명령 RPM-실제 RPM 차이
```

---

# 3. 중요한 모델 표현 규칙

현재 XGBoost의 `Anomaly Score`는 실제 고장 확률이 아니다.

반드시 아래 의미로 사용한다.

> Anomaly Score는 현재 1분 구간이 모델이 학습한 가상 이상 패턴과 얼마나 유사한지를 나타내는 점수입니다.

사용 금지 표현:

- 고장 확률 90%
- 실제 고장 가능성 90%
- 99% 확률로 고장
- 베어링 고장 확정
- N시간 후 고장
- 제품 불량 확률

또한 실제 31일 로그 분석에서 공구 교체/공정 전환 직전에도 높은 Anomaly Score가 관찰되었기 때문에, 다음 안내를 포함한다.

> 높은 점수가 반드시 실제 고장을 의미하지 않으며, 공구 교체나 공정 전환에 의해서도 점수가 높아질 수 있습니다.

---

# 4. 최종 화면 구조

Streamlit은 유지한다.

React/Next.js로 변경하지 않는다.

이유:

- FastAPI 연결이 이미 완료되어 있음
- 파일 업로드와 분석 요청이 이미 동작함
- 현재 목표는 연구/프로젝트 시연용 프로토타입
- Streamlit + CSS + Plotly 정도면 필요한 화면 구현 가능
- React로 변경하면 파일 업로드, API 통신, CORS, 상태 관리, UI 컴포넌트 등을 다시 구현해야 함

최종 화면은 크게 2개 영역으로 나눈다.

```text
[ AI 설비 진단 ]    [ 업로드 데이터 분석 ]
```

Streamlit의 `st.tabs()`를 사용하는 것을 1차 권장안으로 한다.

---

# 5. 화면 1 — AI 설비 진단

## 5-1. 상단

표시:

```text
MCT 예지보전 시스템

MCT 설비 로그를 업로드하면 실제 절삭 구간을 추출하고,
AI를 이용해 평소와 다른 이상 패턴을 분석합니다.
```

파일 업로드 영역:

- `.log` 파일 선택
- 파일명
- 파일 크기
- `AI 분석 시작` 버튼

기존 FastAPI 요청 로직은 그대로 유지한다.

---

## 5-2. 종합 결과

분석 완료 후 가장 먼저 보여준다.

카드는 최대 3개만 둔다.

```text
[ 분석 결과 ]
높은 이상 패턴 없음 / 확인 필요

[ 최대 Anomaly Score ]
0.0107

[ 가장 주의할 구간 ]
Tool 13
05:19:14 ~ 05:20:12
```

### 상태 표현

현재 정상/주의/이상 임계값은 최종 확정되지 않았다.

따라서 초기 구현에서는 임의로 3단계 임계값을 만들지 않는다.

1차 UI에서는 다음처럼 표현한다.

```text
뚜렷한 이상 패턴 없음
높은 이상 패턴 확인
```

실제 임계값 정책은 추후 검증 후 추가한다.

---

## 5-3. 한눈에 보기

숫자를 다시 나열하지 않고 자동 설명문을 만든다.

예:

```text
이번 파일에서 842개의 절삭 구간을 AI가 분석했습니다.
가장 높은 Anomaly Score는 0.0107이며,
Tool 13의 05:19~05:20 구간에서 확인되었습니다.
전체적으로 학습된 가상 이상 패턴과의 유사도는 낮은 수준입니다.
```

이 영역의 목적:

- 사용자가 숫자의 의미를 바로 이해
- 교수/기업 시연 시 화면을 읽는 것만으로 분석 흐름 이해

---

# 6. 왜 이 구간을 확인해야 하나요?

이 영역은 반드시 추가한다.

현재 API가 제공하는 두 종류의 정보를 함께 활용한다.

### 1) 정상 대비 지표

현재 API 결과:

```text
load_vs_normal
rpm_std_vs_normal
rpm_error_vs_normal
```

사용자 표현:

```text
평소 대비 Spindle Load
평소 대비 RPM 흔들림
평소 대비 명령 RPM-실제 RPM 차이
```

예시 설명:

```text
평소 대비 Spindle Load가 약 3.3% 높았습니다.
RPM 변동은 평소와 비슷했습니다.
명령 RPM과 실제 RPM의 차이는 정상 기준보다 크게 증가하지 않았습니다.
```

주의:
`rpm_error_vs_normal = 0.091`을 "9.1% 증가"라고 단순 표현하지 않는다.
이 값은 공구별 기준과 비교한 상대지표이므로 계산 정의에 맞는 문장을 사용한다.

---

### 2) XGBoost 주요 기여 Feature

현재 API 결과:

```text
top_contributions
```

Feature 이름을 그대로 노출하지 않는다.

매핑 예시:

```text
rpm_error_absmax
→ 명령 RPM과 실제 RPM의 최대 차이

load_vs_normal
→ 평소 대비 Spindle Load

load_max
→ 구간 내 최대 Spindle Load

rpm_std
→ RPM 흔들림 정도

rpm_range
→ 구간 내 RPM 변화 폭
```

표현:

```text
AI가 이상 방향 점수를 높게 계산할 때
다음 특징의 영향이 상대적으로 컸습니다.

1. 구간 내 최대 Spindle Load
2. RPM 최댓값
3. 평균 Spindle Load
```

반드시 아래 문구 포함:

> 이 값들은 XGBoost 내부 판단에 영향을 준 특징이며 실제 고장 원인을 의미하지 않습니다.

---

# 7. 사용자 확인 권장 사항

"AI가 진단했으니 고장이다"가 아니라
현장 점검 보조 시스템처럼 보여야 한다.

예:

```text
확인 권장 사항

• 해당 시간대의 사용 공구 상태를 확인하세요.
• 공구 교체 또는 공정 전환이 있었는지 확인하세요.
• 명령 RPM과 실제 RPM의 차이가 일시적으로 증가했는지 확인하세요.
• Spindle Load가 평소보다 크게 증가했다면 공구 마모나 가공 조건을 함께 확인하세요.

※ AI 분석만으로 실제 고장을 확정하지 않습니다.
```

문구는 Feature에 따라 동적으로 구성할 수 있다.

---

# 8. 상위 주의 구간

메인 화면에서는 Top 10이 아니라 **Top 3만 표시**한다.

컬럼:

```text
시작 시간
종료 시간
공구
Anomaly Score
```

제거:

- sample_count
- load_vs_normal
- rpm_std_vs_normal
- rpm_error_vs_normal

상세 데이터는 두 번째 탭으로 이동한다.

---

# 9. 화면 2 — 업로드 데이터 분석

이 화면은 "AI가 어떤 결론을 내렸나?"가 아니라

> "내가 올린 로그가 어떤 데이터인가?"

를 설명한다.

초기 메인 카드:

```text
전체 측정 데이터
실제 절삭 데이터
연속 절삭 구간
사용 공구 수
평균 센서 수집 간격
```

예시:

```text
전체 측정 데이터     15,170건
실제 절삭 데이터      8,505건
연속 절삭 구간            3개
사용 공구                 3종
평균 수집 간격          약 5.5초
```

---

# 10. 업로드 데이터 분석 — 쉬운 설명

숫자 아래 자동 설명:

```text
이 파일에는 총 15,170건의 설비 측정 데이터가 있습니다.
이 중 실제 절삭 작업으로 판단된 데이터는 8,505건이며,
총 3개의 연속 절삭 작업 구간이 확인되었습니다.
절삭 과정에서는 3종의 공구가 사용되었습니다.
```

---

# 11. 운전/절삭 현황

표시 예:

```text
실제 절삭 데이터     56.1%
비절삭 데이터        43.9%
```

쉬운 설명:

> 실제 절삭 데이터는 설비가 RUN 상태이면서 Spindle RPM과 Feed Rate가 모두 0보다 큰 측정값입니다.

---

# 12. 사용 공구 분석

추가할 데이터:

- 공구 번호
- 공구별 절삭 행 수
- 공구별 사용 비율
- 대표 RPM
- 대표 Spindle Load

예:

```text
Tool 13
절삭 데이터 5,268건
대표 RPM 약 22,000

Tool 14
절삭 데이터 3,204건
대표 RPM 약 22,000

Tool 2
절삭 데이터 33건
대표 RPM 약 7,000
```

그래프:

- 공구별 절삭 데이터량 Bar Chart

Plotly 사용 권장.

---

# 13. 절삭 Timeline

사용자가 파일 전체 작업 흐름을 이해하도록 한다.

예:

```text
시간 ───────────────────────────────────>

Tool 14  █████████████
                   Tool 13  ████████████████████
                                           Tool 2 ██
```

각 Segment 정보:

- 시작 시간
- 종료 시간
- Tool
- 지속 시간

Plotly timeline chart를 우선 고려한다.

이 화면은 향후 AI 결과의 주의 구간과 연결한다.

예:

```text
주의 구간: 14:01
→ 직후 공구 교체 발생 여부 확인
```

---

# 14. 데이터 분석 가능 여부

기존 개발자 표현:

```text
supported_windows
unsupported_windows
insufficient_windows
```

사용자 표현:

```text
AI 분석 가능 구간
정상 기준이 없어 제외된 구간
측정값이 부족해 제외된 구간
```

예:

```text
AI 분석 가능 구간             842개
정상 기준이 없어 제외된 구간     0개
측정값이 부족해 제외된 구간       1개
```

---

# 15. FastAPI에 추가로 필요한 데이터

현재 Streamlit 화면에 이미 들어오는 값:

```text
clean_rows
cutting_rows
supported_windows
unsupported_windows
insufficient_windows
score_mean
score_max
highest_window
top_windows
top_contributions
```

상세 데이터 분석 화면을 위해 `/predict` 응답에 아래 값을 추가한다.

```text
segment_count
used_tool_count
sampling_interval_mean
cutting_ratio
tool_summary
cutting_segments
```

권장 JSON 구조:

```json
{
  "segment_count": 3,
  "used_tool_count": 3,
  "sampling_interval_mean": 5.56,
  "cutting_ratio": 0.5606,

  "tool_summary": [
    {
      "tool": 13,
      "cutting_rows": 5268,
      "cutting_ratio": 0.6194,
      "rpm_median": 22001,
      "load_median": 14
    }
  ],

  "cutting_segments": [
    {
      "tool": 14,
      "start": "2026-05-01 00:00:04",
      "end": "2026-05-01 05:17:52",
      "duration_minutes": 317.79
    }
  ]
}
```

---

# 16. 수정할 파일

## A. `streamlit_app.py`

역할:

- 화면 구조
- 파일 업로드
- FastAPI 호출
- 결과 카드
- 쉬운 설명 생성
- 두 개 탭
- Top 3 / Top 10 표시
- Plotly 그래프
- 사용자용 문구

수정 순서:

```text
1. 기존 FastAPI 호출 유지
2. 기존 결과 영역 2~7 제거
3. AI 설비 진단 탭 생성
4. 종합 결과 카드
5. 한눈에 보기
6. 왜 이 구간을 확인해야 하나요?
7. 확인 권장 사항
8. Top 3
9. 업로드 데이터 분석 탭 생성
10. 상세 데이터 카드
11. 공구 현황
12. Timeline
```

---

## B. `api/main.py`

초기 메인 화면 개선 때는 수정하지 않는다.

두 번째 탭 개발 시 수정한다.

추가 응답:

```text
segment_count
used_tool_count
sampling_interval_mean
cutting_ratio
tool_summary
cutting_segments
```

---

## C. `src/inference_pipeline.py`

가능하면 기존 추론 로직은 변경하지 않는다.

필요한 경우 이미 생성된 중간 DataFrame에서 통계만 계산한다.

중요:

- 모델 입력 Feature 계산 방식 변경 금지
- 기존 모델 예측 로직 변경 금지
- 공구별 Baseline 계산 방식 변경 금지
- 1분 Window 생성 기준 변경 금지

이번 UI 개선은 **모델 성능을 건드리는 작업이 아니다.**

---

# 17. 구현 순서

## Phase 1 — 메인 AI 진단 화면

수정 파일:

```text
streamlit_app.py
```

목표:

```text
종합 결과
→ 한눈에 보기
→ 가장 주의할 구간
→ 왜 주의해야 하나
→ 확인 권장 사항
→ Top 3
```

FastAPI 수정 없음.

---

## Phase 2 — 상세 데이터 API 확장

수정 파일:

```text
api/main.py
```

필요하면:

```text
src/inference_pipeline.py
```

목표:

- 연속 절삭 Segment 수 반환
- 사용 공구 정보 반환
- 평균 수집 간격 반환
- 공구별 요약 반환
- Segment Timeline 데이터 반환

---

## Phase 3 — 업로드 데이터 분석 탭

수정 파일:

```text
streamlit_app.py
```

목표:

```text
파일 데이터 개요
운전/절삭 비율
사용 공구 분석
공구별 데이터량 그래프
절삭 Timeline
AI 분석 가능 여부
```

---

## Phase 4 — 시각 디자인 정리

수정 파일:

```text
streamlit_app.py
```

추가:

- CSS 카드
- 여백/폰트 정리
- Plotly 그래프
- expander 사용
- 기술 정보는 상세 영역으로 이동

---

## Phase 5 — 최종 검증

테스트 파일:

```text
Data.log_20260501.log
```

기존 검증 결과와 비교:

```text
전체 측정 행       15,170
실제 절삭 행        8,505
분석 결과 Window      842
표본 부족 Window         1
최대 Score       0.01073095
최고점 Tool             13
최고점 시작       05:19:14
```

UI를 바꾼 뒤에도 위 값이 달라지면 안 된다.

---

# 18. 코드 작업 시 권장 방식

앞으로 채팅에서 긴 코드 조각을 계속 주고받지 않는다.

권장 흐름:

```text
1. 사용자: "Phase 1 진행하자"
2. ChatGPT: 수정된 전체 streamlit_app.py 파일 생성
3. 사용자: 파일 다운로드 후 프로젝트에 덮어쓰기
4. 사용자: 실행 화면 캡처
5. 필요한 부분만 수정
6. Phase 2로 이동
```

즉 **부분 코드 복붙보다 완성된 파일 단위로 교체**한다.

이 방식의 장점:

- 코드 누락 방지
- 들여쓰기 오류 방지
- 어디에 붙여야 하는지 혼란 감소
- 채팅 길이 감소
- 새로운 ChatGPT 대화에서도 이 문서 하나로 바로 이어갈 수 있음

---

# 19. 앞으로 유지할 프로젝트 문서

권장:

```text
docs/
├─ STREAMLIT_UI_REDESIGN_GUIDE.md
└─ PROJECT_HANDOFF.md
```

### STREAMLIT_UI_REDESIGN_GUIDE.md

이 문서.

UI 설계와 구현 순서를 기록한다.

### PROJECT_HANDOFF.md

프로젝트 전체 현재 상태를 요약한다.

포함 내용:

- 데이터 구조
- EDA 핵심 결과
- Feature Engineering
- 가상 이상 데이터
- 모델 비교
- XGBoost 최종 결과
- FastAPI
- Streamlit
- 남은 작업

새 대화를 열 때 두 파일만 제공하면 프로젝트를 빠르게 이어갈 수 있다.

---

# 20. 현재 바로 다음 작업

다음 작업은 **Phase 1 — 메인 AI 설비 진단 화면 개선**이다.

수정 대상:

```text
streamlit_app.py
```

변경할 부분:

```text
현재
2. 분석 결과 요약
3. 분석 데이터 현황
4. 가장 높은 이상 패턴 구간
5. 정상 대비 지표
6. 모델 주요 이상 기여 Feature
7. Top 10

↓

변경

종합 진단
한눈에 보기
가장 주의할 구간
왜 이 구간을 확인해야 하나요?
확인 권장 사항
상위 주의 구간 3개
상세 분석 탭
```

FastAPI 및 XGBoost 추론 코드는 Phase 1에서 변경하지 않는다.
