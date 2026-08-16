// ============================================================================
// notebook_data.js
// Jupyter Notebook에 저장된 실행 결과(output)에서 그대로 옮겨 적은 값입니다.
// 각 항목의 source 필드에 근거 Notebook / Cell 을 표기했습니다.
// 이 파일은 web_report 전용 신규 파일이며, 원본 Notebook은 수정하지 않았습니다.
// ============================================================================

const NB_DATA = {

  // --- Window 크기 비교 (정상 절삭 구간 기준, 변동성 Feature의 P95) ---
  windowCompare: {
    source: "model_eda.ipynb Cell 5",
    rows: [
      { feature: "rpm_std",       s30: 0.00,    m1: 3.02,    m3: 3.97,    m5: 1012.30 },
      { feature: "rpm_range",     s30: 0.00,    m1: 10.00,   m3: 10.00,   m5: 3000.00 },
      { feature: "load_std",      s30: 2.16,    m1: 2.21,    m3: 2.37,    m5: 2.45 },
      { feature: "load_range",    s30: 5.00,    m1: 6.00,    m3: 7.00,    m5: 8.00 },
      { feature: "feed_std",      s30: 1610.15, m1: 1436.83, m3: 1333.33, m5: 1296.91 },
      { feature: "rpmerr_std",    s30: 1.26,    m1: 2.91,    m3: 2.90,    m5: 3.03 },
      { feature: "rpmerr_absmax", s30: 6.00,    m1: 6.00,    m3: 7.00,    m5: 7.00 }
    ],
    winN: { s30: { median: 6, max: 6 }, m1: { median: 11, max: 12 },
            m3: { median: 32, max: 33 }, m5: { median: 52, max: 55 } }
  },

  // --- 실제 ALARM 연속 이벤트 2건의 사전 구간 분석 ---
  alarmEvents: {
    source: "model_eda.ipynb Cell 7",
    events: [
      {
        id: 1, start: "2026-05-19 11:02:05.048", end: "2026-05-19 11:02:31.702",
        tool: 2, rows: 6,
        segments: [
          { seg: "30분 전", n: 209, run_n: 209, rpm: 14989.00, load: 10.42, feed: 1671.33 },
          { seg: "10분 전", n: 55,  run_n: 20,  rpm: 5450.55,  load: 3.80,  feed: 956.60 },
          { seg: "5분 전",  n: 45,  run_n: 0,   rpm: 0,        load: 0,     feed: 0 },
          { seg: "1분 전",  n: 11,  run_n: 0,   rpm: 0,        load: 0,     feed: 0 },
          { seg: "직전(5초)", n: 0, run_n: 0,   rpm: null,     load: null,  feed: null },
          { seg: "발생 중", n: 6,   run_n: 0,   rpm: 0,        load: 0,     feed: 0 }
        ]
      },
      {
        id: 2, start: "2026-05-29 09:42:12.338", end: "2026-05-29 09:43:00.847",
        tool: 14, rows: 10,
        segments: [
          { seg: "30분 전", n: 225, run_n: 0, rpm: 0, load: 0, feed: 0 },
          { seg: "10분 전", n: 56,  run_n: 0, rpm: 0, load: 0, feed: 0 },
          { seg: "5분 전",  n: 45,  run_n: 0, rpm: 0, load: 0, feed: 0 },
          { seg: "1분 전",  n: 11,  run_n: 0, rpm: 0, load: 0, feed: 0 },
          { seg: "직전(5초)", n: 0, run_n: 0, rpm: null, load: null, feed: null },
          { seg: "발생 중", n: 10,  run_n: 0, rpm: 0, load: 0, feed: 0 }
        ]
      }
    ]
  },

  // --- 861행(RPM>0 & FEED=0) 맥락 분석 ---
  rows861: {
    source: "model_eda.ipynb Cell 8",
    totalRows: 861, segments: 559, singleRowSegments: 549, multiRowSegments: 10,
    longestRows: 274, longestSeconds: 1479.7,
    beforeFeedPosPct: 65.1, afterFeedPosPct: 95.9,
    beforeCuttingPct: 41.3, afterCuttingPct: 29.9,
    loadBins: { "0": 38, "1-10": 562, "11-50": 116, "51-100": 85, ">100": 60 }
  },

  // --- 연속 절삭 세그먼트 ---
  segments: {
    source: "synthetic_anomaly.ipynb Cell 2",
    total: 839, rowsSum: 219195,
    rowsMedian: 139, rowsMax: 5268,
    durMedianMin: 13.31, durMaxMin: 521.63,
    over3min: 620, over5min: 560, over10min: 463
  },

  // --- 시간순 split + seed 선택 ---
  seedSplit: {
    source: "synthetic_anomaly.ipynb Cell 3",
    total: 462,
    splits: [
      { name: "train",      from: "2026-05-01", to: "2026-05-21", days: 19, seeds: 363 },
      { name: "validation", from: "2026-05-22", to: "2026-05-26", days: 4,  seeds: 43 },
      { name: "test",       from: "2026-05-27", to: "2026-05-30", days: 4,  seeds: 56 }
    ],
    byType: { load_rise: 155, rpm_instability: 154, rpm_error_drift: 153 },
    cross: {
      train:      { load_rise: 121, rpm_instability: 121, rpm_error_drift: 121 },
      validation: { load_rise: 15,  rpm_instability: 14,  rpm_error_drift: 14 },
      test:       { load_rise: 19,  rpm_instability: 19,  rpm_error_drift: 18 }
    },
    byTool: { 1: 100, 2: 1, 3: 67, 4: 126, 5: 19, 6: 40, 8: 31, 10: 28,
              11: 4, 12: 1, 13: 16, 14: 2, 15: 3, 16: 3, 18: 10, 19: 11 }
  },

  // --- Window 분할 검증 ---
  windowBuild: {
    source: "feature_engineering.ipynb Cell 2",
    totalWindows: 4620, perSeed: 10,
    winNMin: 9, winNMedian: 10, winNMax: 11,
    mixedStageWindows: 0, outOfRangeWindows: 0,
    stageCount: { normal: 1386, warning: 1386, anomaly: 1848 },
    labelCount: { 0: 1386, 1: 3234 },
    splitCount: { train: 3630, validation: 430, test: 560 }
  },

  // --- XGBoost 하이퍼파라미터 (실제 실행값) ---
  xgbParams: {
    source: "train_xgboost.ipynb Cell 2",
    version: "3.4.0",
    params: [
      ["n_estimators", "1000", "최대 트리 개수. Early Stopping으로 실제 사용량 결정"],
      ["max_depth", "4", "트리 최대 깊이. 과적합 억제"],
      ["learning_rate", "0.05", "한 트리가 반영되는 비율. 낮게 설정"],
      ["subsample", "0.8", "트리마다 학습 행의 80%만 사용"],
      ["colsample_bytree", "0.8", "트리마다 Feature의 80%만 사용"],
      ["min_child_weight", "3", "리프 노드 최소 가중치. 과적합 억제"],
      ["gamma", "0.0", "분할 최소 이득"],
      ["reg_lambda", "1.0", "L2 정규화"],
      ["objective", "binary:logistic", "이진 분류"],
      ["eval_metric", "logloss", "Early Stopping 판정 지표 (실제 실행값)"],
      ["scale_pos_weight", "1", "클래스 가중치 미조정"],
      ["random_state", "42", "재현성 고정"],
      ["early_stopping_rounds", "50", "50라운드 개선 없으면 중단"]
    ],
    bestIteration: 509
  },

  // --- 성능 지표 ---
  metrics: {
    source: "train_xgboost.ipynb Cell 2 / Cell 3",
    train:      { n: 3630, acc: 0.9992, prec: 0.9992, rec: 0.9996, f1: 0.9994, prauc: 1.0000, cm: [[1087, 2], [1, 2540]] },
    validation: { n: 370,  acc: 1.0000, prec: 1.0000, rec: 1.0000, f1: 1.0000, prauc: 1.0000, cm: [[111, 0], [0, 259]] },
    test:       { n: 560,  acc: 0.9964, prec: 0.9974, rec: 0.9974, f1: 0.9974, prauc: 1.0000, cm: [[167, 1], [1, 391]] }
  },

  // --- TEST 세부 (이상 유형별 / stage별) ---
  testDetail: {
    source: "train_xgboost.ipynb Cell 3",
    byType: [
      { name: "normal (정상을 0으로)", n: 168, rate: 0.9940 },
      { name: "load_rise",             n: 133, rate: 1.0000 },
      { name: "rpm_instability",       n: 133, rate: 0.9925 },
      { name: "rpm_error_drift",       n: 126, rate: 1.0000 }
    ],
    byStage: [
      { name: "normal",  n: 168, rate: 0.9940, target: 0 },
      { name: "warning", n: 168, rate: 0.9940, target: 1 },
      { name: "anomaly", n: 224, rate: 1.0000, target: 1 }
    ],
    warningCaught: 167
  },

  // --- Feature Importance (gain %) ---
  importance: {
    source: "train_xgboost.ipynb Cell 3",
    rows: [
      { f: "rpm_error_absmax",    v: 39.39, rel: false },
      { f: "load_vs_normal",      v: 18.81, rel: true },
      { f: "load_max",            v: 16.82, rel: false },
      { f: "rpm_error_vs_normal", v: 9.48,  rel: true },
      { f: "load_mean",           v: 3.93,  rel: false },
      { f: "rpm_range",           v: 2.34,  rel: false },
      { f: "rpm_std",             v: 2.11,  rel: false },
      { f: "rpm_mean",            v: 1.53,  rel: false },
      { f: "rpm_max",             v: 1.49,  rel: false },
      { f: "load_std",            v: 1.34,  rel: false },
      { f: "rpm_min",             v: 1.15,  rel: false },
      { f: "rpm_std_vs_normal",   v: 1.07,  rel: true },
      { f: "rpm_error_mean",      v: 0.33,  rel: false },
      { f: "rpm_error_std",       v: 0.22,  rel: false }
    ]
  },

  // --- 원본 로그 예시 (data/raw/Data.log_20260501.log 첫 13줄) ---
  rawLogSample: [
    "2026-05-01 00:00:04,522 [MCT_MCT3호] FEED_RATE: 392",
    "2026-05-01 00:00:04,569 [MCT_MCT3호] SPINDLE_RATE: 21997",
    "2026-05-01 00:00:04,679 [MCT_MCT3호] X_AXIS: -227440",
    "2026-05-01 00:00:04,757 [MCT_MCT3호] Y_AXIS: -50048",
    "2026-05-01 00:00:04,819 [MCT_MCT3호] Z_AXIS: -11100",
    "2026-05-01 00:00:05,119 [MCT_MCT3호] CURRENT_TOOL: 14",
    "2026-05-01 00:00:05,151 [MCT_MCT3호] X LOAD: 4 %",
    "2026-05-01 00:00:05,151 [MCT_MCT3호] Y LOAD: 4 %",
    "2026-05-01 00:00:05,151 [MCT_MCT3호] Z LOAD: 4 %",
    "2026-05-01 00:00:05,279 [MCT_MCT3호] SPINDLE_LOAD: 15%",
    "2026-05-01 00:00:05,279 [MCT_MCT3호] SPINDLE_RPM: 22001",
    "2026-05-01 00:00:05,279 [MCT_MCT3호] DELTA_SPINDLE_RPM: 0",
    "2026-05-01 00:00:05,404 [MCT_MCT3호] STATUS: STATUS_RUN, AUT_MODE: 1, RUN_MODE: 3, EMERGENCY: 0, ALARM: 0, ERROR_MESSAGE: "
  ],

  // --- Feature 14종 설명 ---
  features: [
    { g: "RPM", f: "rpm_mean",  d: "1분 동안의 평균 회전수", why: "그 Window가 어떤 운전점(공구/조건)에 있었는지 알려주는 기준값", tgt: "-" },
    { g: "RPM", f: "rpm_std",   d: "1분 동안 회전수가 얼마나 흔들렸는지", why: "회전 안정성이 나빠지면 가장 먼저 커지는 값", tgt: "rpm_instability" },
    { g: "RPM", f: "rpm_min",   d: "1분 내 최소 회전수", why: "순간적으로 회전이 떨어진 적이 있는지", tgt: "rpm_instability" },
    { g: "RPM", f: "rpm_max",   d: "1분 내 최대 회전수", why: "순간적으로 회전이 튄 적이 있는지", tgt: "rpm_instability" },
    { g: "RPM", f: "rpm_range", d: "max − min. 1분 내 전체 흔들림 폭", why: "std보다 극단값에 민감해 짧은 튐을 잡음", tgt: "rpm_instability" },
    { g: "LOAD", f: "load_mean", d: "1분 평균 스핀들 부하", why: "절삭 저항이 전반적으로 올라갔는지", tgt: "load_rise" },
    { g: "LOAD", f: "load_std",  d: "1분 부하 변동성", why: "부하가 불규칙하게 출렁이는지", tgt: "load_rise" },
    { g: "LOAD", f: "load_max",  d: "1분 내 순간 최대 부하", why: "평균은 정상이어도 순간 피크가 있는지", tgt: "load_rise" },
    { g: "RPM_ERROR", f: "rpm_error_mean",   d: "지령 − 실제 회전수의 1분 평균", why: "지령 추종이 한쪽으로 치우쳤는지", tgt: "rpm_error_drift" },
    { g: "RPM_ERROR", f: "rpm_error_std",    d: "추종 오차 자체의 변동성", why: "오차가 흔들리는지(불안정 추종)", tgt: "rpm_error_drift / rpm_instability" },
    { g: "RPM_ERROR", f: "rpm_error_absmax", d: "1분 중 가장 크게 벌어진 지령−실제 차이", why: "짧은 순간의 큰 추종 실패를 놓치지 않기 위해", tgt: "rpm_error_drift" },
    { g: "상대", f: "load_vs_normal",      d: "현재 부하 ÷ 이 공구의 정상 부하 중심", why: "공구마다 정상 부하가 6~17로 달라 절대값 비교가 무의미하기 때문", tgt: "load_rise" },
    { g: "상대", f: "rpm_std_vs_normal",   d: "현재 RPM 변동성 ÷ 이 공구의 정상 변동성", why: "정상 변동 폭 자체가 공구마다 다르기 때문", tgt: "rpm_instability" },
    { g: "상대", f: "rpm_error_vs_normal", d: "정상 오차 중심에서 벗어난 크기 ÷ 허용 편차", why: "정상 중심이 −4 ~ +4로 공구마다 달라 0 기준 비교가 불가능하기 때문", tgt: "rpm_error_drift" }
  ]
};
