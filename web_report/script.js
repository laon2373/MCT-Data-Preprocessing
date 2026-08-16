/* ==========================================================================
   MCT AI 예지보전 프로토타입 — 프로젝트 보고서 렌더링 스크립트
   REPORT_DATA : 원본 CSV에서 재계산한 값 (assets/data.js)
   NB_DATA     : Notebook 저장 실행 결과에서 옮긴 값 (assets/notebook_data.js)
   ========================================================================== */
(function () {
  "use strict";

  const R = window.REPORT_DATA;
  const N = window.NB_DATA;

  /* ------------------------------------------------------------ helpers */
  const $ = (id) => document.getElementById(id);

  function fmt(v, digits) {
    if (v === null || v === undefined || v === "" || Number.isNaN(v)) return "—";
    if (typeof v !== "number") return String(v);
    if (digits !== undefined) return v.toLocaleString("ko-KR", { minimumFractionDigits: digits, maximumFractionDigits: digits });
    if (Number.isInteger(v)) return v.toLocaleString("ko-KR");
    return v.toLocaleString("ko-KR", { maximumFractionDigits: 2 });
  }
  const pct = (v, d) => (v * 100).toFixed(d === undefined ? 2 : d) + "%";

  /**
   * 표 렌더링
   * @param {string} id      table 요소 id
   * @param {Array}  headers [{t: 표시명, n: 숫자열 여부}]
   * @param {Array}  rows    [[셀, ...], ...]  셀은 값 또는 {v, cls, html}
   * @param {object} opt     {caption, rowClass: fn(rowIdx) => class}
   */
  function table(id, headers, rows, opt) {
    const el = $(id);
    if (!el) return;
    opt = opt || {};
    const th = headers
      .map((h) => `<th class="${h.n ? "num" : ""}">${h.t}</th>`)
      .join("");
    const tb = rows
      .map((r, ri) => {
        const cls = opt.rowClass ? opt.rowClass(ri, r) : "";
        const tds = r
          .map((c, ci) => {
            const isNum = headers[ci] && headers[ci].n;
            if (c && typeof c === "object" && !Array.isArray(c)) {
              return `<td class="${isNum ? "num " : ""}${c.cls || ""}">${c.html !== undefined ? c.html : fmt(c.v)}</td>`;
            }
            return `<td class="${isNum ? "num" : ""}">${typeof c === "number" ? fmt(c) : c === null || c === undefined ? "—" : c}</td>`;
          })
          .join("");
        return `<tr class="${cls}">${tds}</tr>`;
      })
      .join("");
    el.innerHTML =
      `<thead><tr>${th}</tr></thead><tbody>${tb}</tbody>` +
      (opt.caption ? `<caption>${opt.caption}</caption>` : "");
  }

  /* ------------------------------------------------------------ Chart.js */
  const PALETTE = {
    blue: "#1d4ed8", blueSoft: "#93b4f5",
    green: "#0f7a52", greenSoft: "#7fc9ab",
    amber: "#d68a10", amberSoft: "#f0c47c",
    red: "#b42318", redSoft: "#e79c95",
    violet: "#7c3aed", violetSoft: "#bda0f7",
    grey: "#8a94a6", greySoft: "#cdd4de",
    ink: "#17202f"
  };

  if (window.Chart) {
    Chart.defaults.font.family =
      "Pretendard, 'Malgun Gothic', '맑은 고딕', -apple-system, 'Segoe UI', sans-serif";
    Chart.defaults.font.size = 11.5;
    Chart.defaults.color = "#46536b";
    Chart.defaults.plugins.legend.labels.boxWidth = 12;
    Chart.defaults.plugins.legend.labels.boxHeight = 12;
    Chart.defaults.maintainAspectRatio = false;
  }

  function chart(id, cfg) {
    const el = $(id);
    if (!el || !window.Chart) return null;
    return new Chart(el.getContext("2d"), cfg);
  }

  /* ==================================================================== */
  /* 01. KPI 카드                                                          */
  /* ==================================================================== */
  function renderKpi() {
    const items = [
      ["분석 기간", "2026년 5월", "31일"],
      ["원본 로그 파일", "31", "개"],
      ["원본 로그 라인", "6,261,455", "줄"],
      ["Clean Data", fmt(R.clean_shape[0]), "행"],
      ["실제 절삭 데이터", fmt(R.cut_rows), "행"],
      ["Synthetic Seed", fmt(R.syn_seeds), "개"],
      ["Synthetic Data", fmt(R.syn_shape[0]), "행"],
      ["최종 1분 Window", fmt(R.td_shape[0]), "개"],
      ["Feature", "14", "종"],
      ["모델 적용 공구", String(R.td_tools.length), "종"],
      ["Model", "XGBoost", "3.4.0"]
    ];
    $("kpiGrid").innerHTML = items
      .map(
        (i) =>
          `<div class="kpi"><div class="k-label">${i[0]}</div>` +
          `<div class="k-value">${i[1]}<span class="k-unit">${i[2]}</span></div></div>`
      )
      .join("");
  }

  /* ==================================================================== */
  /* 01. 전체 파이프라인 Flow                                              */
  /* ==================================================================== */
  const FLOW = [
    ["raw", "기업 원본 MCT 로그", "31개 파일 · 6,261,455줄 · CP949",
      "설비 MCT_MCT3호 1대가 5~6초마다 남긴 텍스트 로그. 측정 1회가 13줄에 나뉘어 기록되어 있습니다."],
    ["proc", "Parsing", "src/parse_mct_log.py",
      "FEED_RATE에서 묶음을 시작하고 STATUS에서 한 행을 확정합니다. % 기호·공백을 제거하고 STATUS 줄을 6개 항목으로 분해합니다."],
    ["file", "mct_parsed.csv", "중간 산출물 (현재 output 폴더에 없음)",
      "파싱 과정에서 생성되는 중간 산출물입니다. 재현하려면 parse_mct_log.py를 다시 실행해야 합니다."],
    ["proc", "Validation / Cleaning", "src/validate_mct_data.py + build_clean_dataset.py",
      "중복·시간 변환·수집 간격·결측을 전수 검증하고, 파생 열과 상태 플래그를 추가합니다. 원본 센서값은 삭제·수정하지 않습니다."],
    ["file", "mct_clean.csv", "481,648행 × 31열",
      "분석의 기준이 되는 정제 데이터. 파싱 후 행 수가 그대로 유지되었음을 코드가 검증합니다."],
    ["proc", "AI용 EDA", "notebooks/model_eda.ipynb",
      "STATUS 분포, RUN 내부 구성, 공구별 프로파일, RPM_ERROR, Window 크기, ALARM 사전 징후를 분석했습니다."],
    ["proc", "실제 절삭 구간 추출", "STATUS_RUN & RPM>0 & FEED>0",
      "RUN 243,319행 중 실제로 가공이 진행 중인 219,195행(90.09%)만 정상 기준 산출 모집단으로 사용합니다."],
    ["proc", "공구별 정상 상태 분석", "공구 19종",
      "공구마다 정상 RPM이 6,990~25,990으로 크게 달라, 단일 전역 임계값이 불가능함을 확인했습니다."],
    ["file", "normal_baseline_by_tool.csv", "19행 × 20열 · 참고 기준",
      "공구별 정상 분포의 참고 기준입니다. 확정된 고장 threshold가 아니며, 파일 안에 reference_only_not_threshold로 명시되어 있습니다."],
    ["proc", "실제 고장 데이터 부족 확인", "ALARM 2건 · emergency 0건",
      "실제 알람 2건 모두 설비 정지 상태에서 기록되어, 센서 기반 사전 열화 패턴을 학습할 수 없었습니다."],
    ["synth", "가상 이상 3종 설계", "load_rise / rpm_instability / rpm_error_drift",
      "공구별 정상 기준에 대한 상대 변형으로 설계했습니다. 절대값을 코드에 하드코딩하지 않았습니다."],
    ["synth", "Synthetic Seed 선택", "462 seed · 각 10분",
      "10분 이상 연속 절삭 segment에서 segment당 1개씩 선택했습니다. 날짜 기준 시간순으로 train/val/test에 배정됩니다."],
    ["file", "synthetic_anomaly_data.csv", "48,240행 × 38열",
      "원본 mct_clean.csv는 수정하지 않고, seed 구간 복사본만 변형하여 별도 파일로 저장했습니다."],
    ["proc", "1분 Window", "4,620 Window",
      "seed 시작 기준 경과초를 60으로 나눠 window_id를 만듭니다. seed당 정확히 10개, stage가 섞인 Window는 0개입니다."],
    ["proc", "Feature 14종", "기본 11종 + 공구 상대 3종",
      "상대 Feature의 기준은 train + stage=normal Window 1,089개에서만 계산하여 Data Leakage를 막았습니다."],
    ["file", "training_dataset.csv", "4,560행 × 24열",
      "공구 15·16의 60 Window는 Train 기간에 정상 기준이 없어 제외되었습니다. NaN 0 · inf 0 · 스케일링 없음."],
    ["model", "XGBoost", "Early Stopping(logloss) · best_iteration 509",
      "기존 split을 그대로 사용하고 재분할·스케일링 없이 학습했습니다. Test는 마지막에 1회만 평가했습니다."],
    ["model", "가상 이상 탐지 프로토타입", "TEST Accuracy 0.9964",
      "이 수치는 동일 방식으로 생성된 synthetic anomaly test set에 대한 결과이며, 실제 설비 고장 예측 정확도가 아닙니다."]
  ];

  function renderFlow(id, list) {
    const el = $(id);
    if (!el) return;
    el.innerHTML =
      '<div class="flow-track">' +
      list
        .map(
          (n, i) =>
            (i > 0 ? '<div class="flow-arrow">▼</div>' : "") +
            `<div class="flow-node" tabindex="0">
               <div class="f-kind ${n[0]}">${
                 { raw: "원본", proc: "처리", file: "산출물", synth: "가상", model: "모델" }[n[0]]
               }</div>
               <div><div class="f-name">${n[1]}</div><div class="f-meta">${n[2]}</div></div>
               <div class="flow-detail">${n[3]}</div>
             </div>`
        )
        .join("") +
      "</div>";
    el.querySelectorAll(".flow-node").forEach((node) => {
      const toggle = () => node.classList.toggle("open");
      node.addEventListener("click", toggle);
      node.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      });
    });
  }

  /* ---- Section 18 추론 파이프라인 ---- */
  const INFER_FLOW = [
    ["raw", "기업 원본 로그 업로드 <span class='badge todo'>향후 구현</span>", "사용자가 .log 파일을 그대로 올림",
      "사용자가 training_dataset.csv 형태를 직접 만들 필요가 없어야 합니다. 최종 목표는 기업 원본 로그 형식을 그대로 업로드하는 것입니다."],
    ["proc", "parse_mct_log와 동일한 Parsing <span class='badge file'>로직 구현됨</span>", "src/parse_mct_log.py 재사용",
      "학습 때와 완전히 동일한 파싱 규칙을 써야 Feature 정의가 어긋나지 않습니다. 현재 스크립트가 이미 존재합니다."],
    ["proc", "clean 전처리 <span class='badge file'>로직 구현됨</span>", "src/build_clean_dataset.py 재사용",
      "interval_seconds, mm 변환, 상태 플래그를 동일하게 생성합니다."],
    ["proc", "절삭 구간 필터 <span class='badge file'>로직 구현됨</span>", "STATUS_RUN & RPM>0 & FEED>0",
      "정지·전환 구간을 이상으로 오탐하지 않도록, 학습 때와 같은 조건으로 절삭 구간만 판정 대상에 넣습니다."],
    ["proc", "1분 Window <span class='badge file'>로직 구현됨</span>", "60초 non-overlap",
      "실시간에서는 최근 1분을 슬라이딩하며 판정하도록 바꿔야 합니다."],
    ["proc", "Feature 14개 <span class='badge file'>로직 구현됨</span>", "기본 11 + 상대 3",
      "notebooks/feature_engineering.ipynb의 계산 로직을 스크립트로 옮겨야 합니다."],
    ["file", "Train normal baseline 적용 <span class='badge todo'>저장 필요</span>", "공구별 load_center / rpm_std_center / rpmerr_center / rpmerr_tol",
      "상대 Feature를 계산하려면 학습 때 쓴 공구별 기준값을 파일로 저장해 두어야 합니다. 현재는 Notebook 메모리에만 존재합니다."],
    ["model", "저장된 XGBoost <span class='badge todo'>미구현</span>", "output에 모델 파일 없음",
      "현재 모델이 파일로 저장되어 있지 않아, 커널을 닫으면 사라집니다. 가장 먼저 해결할 항목입니다."],
    ["model", "Anomaly Score <span class='badge todo'>미구현</span>", "predict_proba 기반 이상 점수",
      "이 값을 '실제 고장 확률'이라고 부르지 않습니다. '모델 이상 점수(Anomaly Score)'로 표현합니다."],
    ["model", "정상 / 주의 / 이상 <span class='badge todo'>임계값 미정</span>", "현재는 0.5 단일 기준뿐",
      "3단계 구분 임계값은 아직 결정되지 않았습니다. 오탐·미탐 비용을 기업과 협의해 정해야 합니다."],
    ["model", "주요 센서 변화 그래프 <span class='badge todo'>미구현</span>", "LOAD / RPM / RPM_ERROR / Anomaly Score",
      "왜 그렇게 판정했는지 근거를 함께 보여주어야 현장에서 신뢰를 얻을 수 있습니다."]
  ];

  /* ==================================================================== */
  /* 02. 원본 로그 샘플                                                    */
  /* ==================================================================== */
  function renderRawLog() {
    const el = $("rawLogSample");
    if (!el) return;
    el.innerHTML = N.rawLogSample
      .map((l, i) =>
        i === 12
          ? `<span style="color:#7fd4a8">${l}</span>   <span class="cm">← 여기서 CSV 1행 확정</span>`
          : i === 0
          ? `<span style="color:#8ab0ff">${l}</span>   <span class="cm">← 측정 묶음 시작</span>`
          : l
      )
      .join("\n");
  }

  /* ==================================================================== */
  /* 03. 검증 / 플래그                                                     */
  /* ==================================================================== */
  function renderValidate() {
    const iv = R.interval;
    table("tblValidate",
      [{ t: "검증 항목" }, { t: "결과", n: true }, { t: "해석" }],
      [
        ["전체 행 수", fmt(R.clean_shape[0]), "파싱 후 = 최종 clean, 동일"],
        ["완전 중복 행", "0", "중복 레코드 없음"],
        ["timestamp 변환 실패", "0", "모든 시간값이 정상 파싱됨"],
        ["평균 수집 간격", fmt(iv.mean, 2) + " 초", "설계상 약 5~6초 주기"],
        ["중앙 수집 간격", fmt(iv.median, 2) + " 초", "대부분의 관측이 이 근처"],
        ["최소 수집 간격", fmt(iv.min, 2) + " 초", "정상 범위"],
        ["최대 수집 간격", fmt(iv.max, 2) + " 초", "특정 시각의 수집 공백"],
        ["비정상 간격(>15초 또는 ≤0)", "17 건", "새벽 03:03에 반복 발생하는 패턴 존재"],
        ["error_message 결측", fmt(481632), "알람이 없어 메시지가 없는 정상적 결측"],
        ["interval_seconds 결측", "1", "설비의 첫 관측치라 이전 값 없음"]
      ]);

    const f = R.flags;
    table("tblFlags",
      [{ t: "플래그" }, { t: "정의" }, { t: "건수", n: true }],
      [
        ["is_alarm", "alarm ≠ 0", f.is_alarm],
        ["is_emergency", "emergency = 1 (실제 비상)", { v: f.is_emergency, cls: f.is_emergency === 0 ? "" : "" }],
        ["is_reset", "emergency = 2 (RESET)", f.is_reset],
        ["is_spindle_overload", "spindle_load > 100", f.is_spindle_overload],
        ["is_abnormal_interval", "interval_seconds > 15", f.is_abnormal_interval],
        ["has_event", "위 조건 중 하나라도 해당", f.has_event]
      ],
      { rowClass: (i) => (i === 5 ? "is-hl" : "") });
  }

  /* ==================================================================== */
  /* 04. EDA                                                              */
  /* ==================================================================== */
  const DESC_META = {
    all: { title: "A. 전체 Clean Data (481,648행)",
      desc: "READY/정지 상태가 49.5% 포함된 전체 데이터입니다. feed_rate·spindle_rate·spindle_load의 중앙값이 모두 0인 것은 정지 구간 때문이며, 정상 절삭 상태를 대표하지 않습니다.",
      src: "output/mct_clean.csv 재계산", key: "desc_all" },
    run: { title: "B. STATUS_RUN (243,319행)",
      desc: "운전 상태로 기록된 행입니다. 그러나 이 안에도 RPM 또는 FEED가 0인 정지·전환 구간이 약 10% 섞여 있어 여전히 정상 절삭을 대표하지 못합니다.",
      src: "output/mct_clean.csv 재계산", key: "desc_run" },
    cut: { title: "C. 실제 절삭 Data (219,195행)",
      desc: "STATUS_RUN & RPM>0 & FEED>0. 이 프로젝트에서 '정상'을 정의하는 기준 모집단입니다. spindle_load 중앙값이 11로, 전체 기준의 0과 완전히 다릅니다.",
      src: "output/mct_clean.csv 재계산 (model_eda.ipynb Cell 3과 동일 조건)", key: "desc_cut" },
    tool: { title: "D. 공구별 Normal Cutting (19종)",
      desc: "절삭 모집단을 공구별로 나눈 정상 운전점입니다. 같은 정상 절삭인데도 공구마다 값이 크게 다르다는 것이 이 표의 핵심입니다.",
      src: "output/mct_clean.csv 재계산 · reports/model_eda/normal_baseline_by_tool.csv", key: "tool_profile" },
    win: { title: "E. Train Normal Window (1,089 Window)",
      desc: "최종 학습 데이터에서 split=train & stage=normal 인 1분 Window의 Feature 통계입니다. 상대 Feature의 기준이 바로 이 모집단에서 계산되었습니다.",
      src: "output/training_dataset.csv 재계산", key: "desc_train_normal_window" }
  };

  function renderDesc(tabsId, tblId, descId, srcId, pop) {
    const meta = DESC_META[pop];
    if (descId && $(descId)) $(descId).innerHTML = "<b>" + meta.title + "</b> — " + meta.desc;
    if (srcId && $(srcId)) $(srcId).textContent = meta.src;

    if (pop === "tool") {
      const rows = R.tool_profile
        .slice()
        .sort((a, b) => b.n - a.n)
        .map((t) => {
          const b = R.baseline.find((x) => x.current_tool === t.current_tool) || {};
          return [
            "공구 " + t.current_tool, t.n, t.rpm_median, t.load_median,
            b.load_p95, b.load_p99, t.feed_median, t.rpmerr_median,
            b.insufficient_data ? "표본 부족" : "ok"
          ];
        });
      table(tblId,
        [{ t: "공구" }, { t: "절삭 행 수", n: true }, { t: "RPM 중앙", n: true },
         { t: "LOAD 중앙", n: true }, { t: "LOAD P95", n: true }, { t: "LOAD P99", n: true },
         { t: "FEED 중앙", n: true }, { t: "RPM_ERROR 중앙", n: true }, { t: "신뢰도" }],
        rows,
        { rowClass: (i, r) => (r[8] !== "ok" ? "is-muted" : "") });
      return;
    }

    const rows = R[meta.key].map((d) => [
      d.feature, d.count, d.mean, d.median, d.std, d.min, d.p05, d.p95, d.p99, d.max
    ]);
    table(tblId,
      [{ t: "변수" }, { t: "count", n: true }, { t: "mean", n: true }, { t: "median", n: true },
       { t: "std", n: true }, { t: "min", n: true }, { t: "P05", n: true },
       { t: "P95", n: true }, { t: "P99", n: true }, { t: "max", n: true }],
      rows);
  }

  function bindDescTabs(tabsId, tblId, descId, srcId) {
    const box = $(tabsId);
    if (!box) return;
    box.addEventListener("click", (e) => {
      const btn = e.target.closest(".tab-btn");
      if (!btn) return;
      box.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderDesc(tabsId, tblId, descId, srcId, btn.dataset.pop);
    });
    renderDesc(tabsId, tblId, descId, srcId, "all");
  }

  function renderEda() {
    const sc = R.status_counts;
    chart("chartStatus", {
      type: "doughnut",
      data: {
        labels: ["STATUS_RUN", "STATUS_READY", "STATUS_ERROR"],
        datasets: [{
          data: [sc.STATUS_RUN, sc.STATUS_READY, sc.STATUS_ERROR],
          backgroundColor: [PALETTE.blue, PALETTE.greySoft, PALETTE.red],
          borderWidth: 2, borderColor: "#fff"
        }]
      },
      options: {
        cutout: "58%",
        plugins: {
          legend: { position: "bottom" },
          tooltip: {
            callbacks: {
              label: (c) => {
                const tot = sc.STATUS_RUN + sc.STATUS_READY + sc.STATUS_ERROR;
                return ` ${c.label}: ${fmt(c.parsed)}행 (${((c.parsed / tot) * 100).toFixed(2)}%)`;
              }
            }
          }
        }
      }
    });

    // 상관계수 바
    const pairs = [
      ["spindle_rate ↔ spindle_load", 0.947],
      ["x_axis_mm ↔ y_axis_mm", 0.747],
      ["feed_rate ↔ spindle_load", 0.475],
      ["feed_rate ↔ z_axis_mm", -0.624],
      ["spindle_load ↔ z_axis_mm", -0.905],
      ["spindle_rate ↔ z_axis_mm", -0.924]
    ];
    chart("chartCorr", {
      type: "bar",
      data: {
        labels: pairs.map((p) => p[0]),
        datasets: [{
          data: pairs.map((p) => p[1]),
          backgroundColor: pairs.map((p) => (p[1] > 0 ? PALETTE.blue : PALETTE.red)),
          borderRadius: 3
        }]
      },
      options: {
        indexAxis: "y",
        scales: { x: { min: -1, max: 1, grid: { color: "#eef1f6" } }, y: { grid: { display: false } } },
        plugins: { legend: { display: false } }
      }
    });

    // 상관행렬
    const cc = R.corr_cols;
    table("tblCorr",
      [{ t: "" }].concat(cc.map((c) => ({ t: c, n: true }))),
      R.corr_matrix.map((row, i) => [cc[i]].concat(row.map((v) => ({ v: v, cls: Math.abs(v) >= 0.7 && i !== row.indexOf(v) ? "" : "" })))),
      { caption: "reports/eda/correlation.csv 와 동일한 계산 (전체 clean 데이터 기준)" });

    if ($("xyzSame")) $("xyzSame").textContent = fmt(R.xyz_load_identical_rows);
  }

  /* ==================================================================== */
  /* 05. 절삭 구간                                                        */
  /* ==================================================================== */
  function renderCutting() {
    const q = R.run_quadrants;
    chart("chartRunSplit", {
      type: "bar",
      data: {
        labels: q.map((x) => x.label),
        datasets: [{
          label: "행 수",
          data: q.map((x) => x.count),
          backgroundColor: q.map((x) =>
            x.label === "RPM>0 & FEED>0" ? PALETTE.blue
              : x.label === "RPM>0 & FEED=0" ? PALETTE.amber : PALETTE.greySoft),
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y",
        scales: { x: { type: "logarithmic", grid: { color: "#eef1f6" } }, y: { grid: { display: false } } },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (c) => {
                const d = q[c.dataIndex];
                return [` ${fmt(d.count)}행 (${d.pct}%)`, ` 스핀들 부하 평균 ${d.load_mean} / 최대 ${d.load_max}`];
              }
            }
          }
        }
      }
    });

    table("tblRunQuad",
      [{ t: "RUN 내부 조합" }, { t: "행 수", n: true }, { t: "비율", n: true },
       { t: "LOAD 평균", n: true }, { t: "LOAD 중앙", n: true }, { t: "LOAD 최대", n: true }, { t: "판정" }],
      q.map((x) => [
        x.label, x.count, x.pct + "%", x.load_mean, x.load_median, x.load_max,
        x.label === "RPM>0 & FEED>0" ? "실제 절삭 → 모집단 채택"
          : x.label === "RPM>0 & FEED=0" ? "판정 보류 → 모집단 제외"
          : "절삭 아님 → 모집단 제외"
      ]),
      { rowClass: (i, r) => (r[0] === "RPM>0 & FEED>0" ? "is-hl" : "") });

    // 861행 상세
    const b = N.rows861;
    table("tbl861",
      [{ t: "항목" }, { t: "값", n: true }, { t: "해석" }],
      [
        ["대상 행 수", b.totalRows, "STATUS_RUN & RPM>0 & FEED=0"],
        ["연속 구간 수", b.segments, "인덱스 연속성 기준"],
        ["1행짜리 구간", b.singleRowSegments, "전체 구간의 98.2%"],
        ["2행 이상 구간", b.multiRowSegments, ""],
        ["최장 구간", b.longestRows + " 행", b.longestSeconds + "초 (약 24.7분)"],
        ["구간 직전 FEED>0 비율", b.beforeFeedPosPct + "%", "직전에 가공 중이던 비율"],
        ["구간 직후 FEED>0 비율", b.afterFeedPosPct + "%", "곧바로 가공이 이어짐 → 전환 성격"],
        ["직전이 절삭 중이던 비율", b.beforeCuttingPct + "%", ""],
        ["직후가 절삭 중이던 비율", b.afterCuttingPct + "%", ""],
        ["LOAD 0", b.loadBins["0"], ""],
        ["LOAD 1~10", b.loadBins["1-10"], "대부분이 저부하"],
        ["LOAD 11~50", b.loadBins["11-50"], ""],
        ["LOAD 51~100", b.loadBins["51-100"], ""],
        ["LOAD >100 (과부하)", b.loadBins[">100"], "전체 과부하 60행이 100% 이 구간에 존재"]
      ],
      { rowClass: (i, r) => (r[0] === "LOAD >100 (과부하)" || r[0] === "구간 직후 FEED>0 비율" ? "is-hl" : "") });
  }

  /* ==================================================================== */
  /* 06. 공구별 baseline                                                  */
  /* ==================================================================== */
  function renderTools() {
    const tp = R.tool_profile.slice().sort((a, b) => b.n - a.n);
    const insuff = new Set(R.baseline.filter((b) => b.insufficient_data).map((b) => b.current_tool));

    const mk = (id, key, color, label) =>
      chart(id, {
        type: "bar",
        data: {
          labels: tp.map((t) => "T" + t.current_tool),
          datasets: [{
            label: label,
            data: tp.map((t) => t[key]),
            backgroundColor: tp.map((t) => (insuff.has(t.current_tool) ? PALETTE.greySoft : color)),
            borderRadius: 3
          }]
        },
        options: {
          scales: { y: { beginAtZero: true, grid: { color: "#eef1f6" } }, x: { grid: { display: false } } },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                afterLabel: (c) => {
                  const t = tp[c.dataIndex];
                  return `절삭 ${fmt(t.n)}행` + (insuff.has(t.current_tool) ? " · 표본 부족(회색)" : "");
                }
              }
            }
          }
        }
      });

    mk("chartToolRpm", "rpm_median", PALETTE.blue, "RPM 중앙값");
    mk("chartToolLoad", "load_median", PALETTE.green, "LOAD 중앙값");

    // 대표 공구 프로파일
    table("tblToolProfile",
      [{ t: "공구" }, { t: "절삭 행 수", n: true }, { t: "RPM 중앙", n: true },
       { t: "LOAD 중앙", n: true }, { t: "FEED 중앙", n: true }, { t: "RPM_ERROR 중앙", n: true }, { t: "비고" }],
      tp.map((t) => [
        "공구 " + t.current_tool, t.n, t.rpm_median, t.load_median, t.feed_median, t.rpmerr_median,
        insuff.has(t.current_tool) ? "표본 500행 미만 → 기준 신뢰 불가" : ""
      ]),
      { rowClass: (i, r) => (r[6] ? "is-muted" : "") });

    // baseline 전체
    const BCOLS = [
      ["current_tool", "공구"], ["sample_count", "표본"], ["rpm_median", "RPM med"],
      ["rpm_p95", "RPM P95"], ["rpm_p99", "RPM P99"], ["load_median", "LOAD med"],
      ["load_p95", "LOAD P95"], ["load_p99", "LOAD P99"], ["feed_median", "FEED med"],
      ["feed_p95", "FEED P95"], ["feed_p99", "FEED P99"], ["rpm_error_median", "ERR med"],
      ["rpm_error_p95", "ERR P95"], ["rpm_error_p99", "ERR P99"],
      ["rpm_std_1min_p95", "rpm_std 1min P95"], ["load_std_1min_p95", "load_std 1min P95"],
      ["rpmerr_std_1min_p95", "rpmerr_std 1min P95"], ["insufficient_data", "표본부족"]
    ];
    const brows = R.baseline.map((b) =>
      BCOLS.map((c) => (c[0] === "insufficient_data" ? (b[c[0]] ? "True" : "False") : b[c[0]])));
    const bheaders = BCOLS.map((c, i) => ({ t: c[1], n: i > 0 && c[0] !== "insufficient_data" }));
    const bopt = {
      rowClass: (i) => (R.baseline[i].insufficient_data ? "is-muted" : ""),
      caption: "표본 500행 미만(회색 행)은 insufficient_data = True. 이 표는 참고 기준이며 확정된 고장 threshold가 아닙니다."
    };
    table("tblBaseline", bheaders, brows, bopt);
    table("tblBaseline2", bheaders, brows, bopt);

    table("tblInsufficient",
      [{ t: "공구" }, { t: "절삭 표본", n: true }, { t: "기준 신뢰도" }, { t: "문제" }],
      R.baseline.filter((b) => b.insufficient_data).map((b) => [
        "공구 " + b.current_tool, b.sample_count, b.reliability_note,
        b.sample_count < 10 ? "표본이 극소수라 분위수 자체가 무의미" : "분포 추정이 불안정"
      ]));
  }

  /* ==================================================================== */
  /* 07. RPM_ERROR                                                        */
  /* ==================================================================== */
  function renderRpmError() {
    const h = R.rpm_error_hist;
    chart("chartRpmErr", {
      type: "bar",
      data: {
        labels: h.bins.map((b) => String(b)),
        datasets: [{
          label: "행 수",
          data: h.counts,
          backgroundColor: h.bins.map((b) => (Math.abs(b) <= 5 ? PALETTE.blue : PALETTE.redSoft)),
          borderRadius: 2
        }]
      },
      options: {
        scales: {
          y: { beginAtZero: true, grid: { color: "#eef1f6" }, ticks: { callback: (v) => fmt(v) } },
          x: { grid: { display: false }, title: { display: true, text: "RPM_ERROR (지령 − 실제)" } }
        },
        plugins: { legend: { display: false } }
      }
    });

    const e = R.rpm_error_cut;
    table("tblRpmErr",
      [{ t: "지표" }, { t: "값", n: true }],
      [
        ["corr(SPINDLE_RATE, SPINDLE_RPM)", e.corr_rate_rpm],
        ["RPM_ERROR = 0 비율", e.zero_pct + "%"],
        ["|RPM_ERROR| ≤ 5 비율", e.abs_le5_pct + "%"],
        ["평균", e.mean],
        ["표준편차", e.std],
        ["P01 / P99", e.p01 + " / " + e.p99],
        ["최소 / 최대", fmt(e.min) + " / " + fmt(e.max)]
      ],
      { caption: "절삭 구간 219,195행 기준 · output/mct_clean.csv 재계산" });
  }

  /* ==================================================================== */
  /* 08. Window                                                           */
  /* ==================================================================== */
  function renderWindow() {
    const w = N.windowCompare;
    const keys = ["rpm_std", "rpm_range", "load_std", "load_range", "rpmerr_std", "rpmerr_absmax"];
    const rows = w.rows.filter((r) => keys.indexOf(r.feature) >= 0);
    chart("chartWindow", {
      type: "bar",
      data: {
        labels: rows.map((r) => r.feature),
        datasets: [
          { label: "30초", data: rows.map((r) => r.s30 || 0.01), backgroundColor: PALETTE.greySoft, borderRadius: 3 },
          { label: "1분 (채택)", data: rows.map((r) => r.m1 || 0.01), backgroundColor: PALETTE.blue, borderRadius: 3 },
          { label: "3분", data: rows.map((r) => r.m3 || 0.01), backgroundColor: PALETTE.amberSoft, borderRadius: 3 },
          { label: "5분", data: rows.map((r) => r.m5 || 0.01), backgroundColor: PALETTE.red, borderRadius: 3 }
        ]
      },
      options: {
        scales: {
          y: { type: "logarithmic", grid: { color: "#eef1f6" }, title: { display: true, text: "정상 구간 P95 (로그 스케일)" } },
          x: { grid: { display: false } }
        },
        plugins: {
          legend: { position: "bottom" },
          tooltip: { callbacks: { label: (c) => ` ${c.dataset.label}: ${c.parsed.y === 0.01 ? "0.00" : fmt(c.parsed.y)}` } }
        }
      }
    });

    const wn = w.winN;
    const trows = [[
      "Window당 행 수 (중앙)", wn.s30.median, wn.m1.median, wn.m3.median, wn.m5.median, "표본이 적으면 std 추정이 불안정"
    ]].concat(w.rows.map((r) => [
      r.feature, r.s30, r.m1, r.m3, r.m5,
      r.feature === "rpm_std" ? "30초는 0으로 죽고, 5분은 1012로 과대"
        : r.feature === "rpm_range" ? "5분은 정상인데도 3000까지 벌어짐"
        : ""
    ]));
    const wheaders = [{ t: "지표" }, { t: "30초", n: true }, { t: "1분", n: true },
                      { t: "3분", n: true }, { t: "5분", n: true }, { t: "관찰" }];
    const wopt = {
      rowClass: (i, r) => (r[0] === "rpm_std" || r[0] === "rpm_range" ? "is-hl" : ""),
      caption: "정상 절삭 구간에서 계산한 변동성 Feature의 P95. 값이 작을수록 '정상은 조용하다'는 뜻이며, 이상이 두드러집니다."
    };
    table("tblWindow", wheaders, trows, wopt);
    table("tblWindow2", wheaders, trows, wopt);
  }

  /* ==================================================================== */
  /* 09. ALARM / 라벨 후보                                                */
  /* ==================================================================== */
  function renderAlarm() {
    N.alarmEvents.events.forEach((ev, idx) => {
      table("tblAlarm" + (idx + 1),
        [{ t: "구간" }, { t: "표본 수", n: true }, { t: "절삭행 수", n: true },
         { t: "RPM 평균", n: true }, { t: "LOAD 평균", n: true }, { t: "FEED 평균", n: true }],
        ev.segments.map((s) => [s.seg, s.n, s.run_n, s.rpm, s.load, s.feed]),
        {
          rowClass: (i, r) => (r[2] === 0 ? "is-muted" : ""),
          caption: idx === 0
            ? "5분 전부터 이미 모든 값이 0 — 알람 시점에 설비는 멈춰 있었습니다. (회색 = 절삭행 0)"
            : "30분 전부터 발생 중까지 전 구간 0 — 센서 기반 사전 징후를 관찰할 수 없습니다."
        });
    });

    table("tblLabelCand",
      [{ t: "라벨 후보" }, { t: "규모", n: true }, { t: "문제점" }, { t: "사용 가능?" }],
      [
        ["has_event", "97 행", "alarm / reset / overload / interval 을 합친 규칙 기반 보조 플래그. 고장 정답 아님", "✕"],
        ["ALARM", "16 행 / 2 건", "두 건 모두 설비 정지 상태에서 발생. 사전 열화 패턴 관찰 불가", "✕"],
        ["Emergency", "0 건", "실제 비상 정지 사례가 전혀 없음", "✕"],
        ["Spindle overload", "60 행", "전부 RPM>0 & FEED=0 구간. 절삭 이상인지 공정 전환인지 구분 불가", "✕"],
        ["비정상 수집 간격", "17 건", "새벽 03:03에 반복 → 설비 고장이 아니라 수집 시스템 특성으로 보임", "✕"],
        ["Maintenance History", "없음", "프로젝트에 파일 자체가 존재하지 않음", "✕"],
        ["제품 QC OK/NG", "없음", "프로젝트에 파일 자체가 존재하지 않음", "✕"]
      ],
      { caption: "지도학습에 사용할 수 있는 실제 고장 라벨이 사실상 0개였습니다." });
  }

  /* ==================================================================== */
  /* 10. Synthetic                                                        */
  /* ==================================================================== */
  function renderSynthetic() {
    const s = N.segments;
    table("tblSegment",
      [{ t: "항목" }, { t: "값", n: true }, { t: "의미" }],
      [
        ["총 연속 절삭 segment", s.total, "839개 구간으로 분리됨"],
        ["segment 행 수 합계", s.rowsSum, "절삭 모집단 219,195와 정확히 일치 (코드가 검증)"],
        ["segment 행 수 중앙", s.rowsMedian, ""],
        ["segment 최대 행 수", s.rowsMax, ""],
        ["지속시간 중앙", s.durMedianMin + " 분", ""],
        ["지속시간 최대", s.durMaxMin + " 분", "약 8.7시간 연속 가공"],
        ["3분 이상 segment", s.over3min, ""],
        ["5분 이상 segment", s.over5min, ""],
        ["10분 이상 segment", s.over10min, "→ 이 중 462개가 seed로 확정"]
      ],
      { rowClass: (i, r) => (r[0] === "10분 이상 segment" || r[0] === "segment 행 수 합계" ? "is-hl" : "") });

    // stage 색상 구분 라인차트 생성기
    function stageLine(id, ex, label, extra) {
      const bg = ex.stage.map((st) =>
        st === "normal" ? PALETTE.green : st === "warning" ? PALETTE.amber : PALETTE.red);
      const ds = [{
        label: label,
        data: ex.y,
        borderColor: PALETTE.ink,
        borderWidth: 1.2,
        pointRadius: 2.2,
        pointBackgroundColor: bg,
        pointBorderWidth: 0,
        tension: 0.15
      }];
      if (extra) extra.forEach((e) => ds.push(e));
      return chart(id, {
        type: "line",
        data: { labels: ex.x.map((v) => v.toFixed(1)), datasets: ds },
        options: {
          scales: {
            x: { title: { display: true, text: "seed 시작 후 경과 (분)" }, grid: { display: false },
                 ticks: { maxTicksLimit: 11 } },
            y: { grid: { color: "#eef1f6" } }
          },
          plugins: {
            legend: { position: "bottom" },
            tooltip: { callbacks: { afterLabel: (c) => " stage: " + ex.stage[c.dataIndex] } }
          }
        }
      });
    }

    const lr = R.ex_load_rise;
    stageLine("chartLoadRise", lr, "SPINDLE_LOAD (공구 " + lr.tool + ")", [
      { label: "정상 load_p95 = " + lr.load_p95, data: lr.x.map(() => lr.load_p95),
        borderColor: PALETTE.grey, borderDash: [3, 3], borderWidth: 1, pointRadius: 0 },
      { label: "정상 load_p99 = " + lr.load_p99, data: lr.x.map(() => lr.load_p99),
        borderColor: PALETTE.violet, borderDash: [6, 3], borderWidth: 1, pointRadius: 0 }
    ]);

    const ri = R.ex_rpm_instability;
    stageLine("chartRpmInstab", ri, "SPINDLE_RPM (공구 " + ri.tool + ")");

    const dr = R.ex_rpm_error_drift;
    chart("chartDriftRpm", {
      type: "line",
      data: {
        labels: dr.x.map((v) => v.toFixed(1)),
        datasets: [
          { label: "SPINDLE_RATE (지령) — 불변", data: dr.rate, borderColor: PALETTE.blue,
            borderWidth: 1.6, pointRadius: 0, tension: 0.1 },
          { label: "SPINDLE_RPM (실제) — 이동", data: dr.rpm, borderColor: PALETTE.red,
            borderWidth: 1.4, pointRadius: 0, tension: 0.1 }
        ]
      },
      options: {
        scales: {
          x: { title: { display: true, text: "seed 시작 후 경과 (분)" }, grid: { display: false }, ticks: { maxTicksLimit: 11 } },
          y: { grid: { color: "#eef1f6" } }
        },
        plugins: { legend: { position: "bottom" } }
      }
    });
    stageLine("chartDriftErr", dr, "RPM_ERROR (공구 " + dr.tool + ")", [
      { label: "정상 중앙값 = " + dr.err_median, data: dr.x.map(() => dr.err_median),
        borderColor: PALETTE.grey, borderDash: [3, 3], borderWidth: 1, pointRadius: 0 },
      { label: "정상 P99 = " + dr.err_p99, data: dr.x.map(() => dr.err_p99),
        borderColor: PALETTE.violet, borderDash: [6, 3], borderWidth: 1, pointRadius: 0 }
    ]);

    const c = R.syn_checks;
    table("tblSynCheck",
      [{ t: "검증 항목" }, { t: "결과", n: true }, { t: "판정" }],
      [
        ["SPINDLE_RPM < 0", c.rpm_neg + " 행", "물리적으로 불가능한 값 없음"],
        ["SPINDLE_LOAD < 0", c.load_neg + " 행", "물리적으로 불가능한 값 없음"],
        ["NaN 총 개수", "0", "결측 없음"],
        ["inf 총 개수", "0", "무한대 없음"],
        ["synthetic LOAD 최대", c.load_max, "원본 절삭 최대 " + c.src_load_max + " 이내"],
        ["synthetic RPM 최대", fmt(c.rpm_max), "원본 절삭 최대 " + fmt(c.src_rpm_max) + " 초과"],
        ["원본 최대 초과 행", c.rpm_over_src + " 행", "전체 48,240행의 약 0.1% (clipping 하지 않음)"]
      ],
      { rowClass: (i, r) => (r[0] === "원본 최대 초과 행" ? "is-hl" : "") });

    const order = { normal: 0, warning: 1, anomaly: 2 };
    const sd = R.syn_stage_detail;
    const stRows = [];
    sd.forEach((d) => {
      const useLoad = d.type === "load_rise";
      const src = useLoad ? d.load_mean : d.rpmerr_absmean;
      stRows.push([
        d.type,
        useLoad ? "SPINDLE_LOAD 평균" : "|RPM_ERROR| 평균",
        src.normal, src.warning, src.anomaly,
        (src.anomaly / (src.normal || 1)).toFixed(1) + "배"
      ]);
    });
    table("tblSynStage",
      [{ t: "이상 유형" }, { t: "관측 지표" }, { t: "normal", n: true },
       { t: "warning", n: true }, { t: "anomaly", n: true }, { t: "증가율", n: true }],
      stRows,
      { caption: "rpm_instability는 RPM 평균을 유지하도록 설계했으나, 지령(RATE)이 고정이므로 |RPM_ERROR| 축에서도 신호가 나타납니다." });
  }

  /* ==================================================================== */
  /* 11-12. Feature / Leakage                                             */
  /* ==================================================================== */
  function renderFeature() {
    const trHeaders = [{ t: "공구" }, { t: "train+normal Window", n: true },
                       { t: "load_center", n: true }, { t: "rpm_std_center", n: true },
                       { t: "rpmerr_center", n: true }, { t: "rpmerr_tol", n: true }];
    const trRows = R.train_ref.map((t) => [
      "공구 " + t.current_tool, t.n_window, t.load_center, t.rpm_std_center, t.rpmerr_center, t.rpmerr_tol
    ]);
    const trOpt = {
      caption: "정상 rpm_std_center가 대부분 0이기 때문에, 0으로 나누기를 막는 안전 하한 1.0이 반드시 필요했습니다. " +
               "공구 15·16은 이 표에 없어서 상대 Feature를 계산할 수 없었습니다."
    };
    table("tblTrainRef", trHeaders, trRows, trOpt);
    table("tblTrainRef2", trHeaders, trRows, trOpt);

    const fHeaders = [{ t: "그룹" }, { t: "Feature" }, { t: "계산 의미" },
                      { t: "왜 넣었는가" }, { t: "겨냥한 이상 유형" }];
    const fRows = N.features.map((f) => [f.g, f.f, f.d, f.why, f.tgt]);
    const fOpt = { rowClass: (i) => (N.features[i].g === "상대" ? "is-hl" : "") };
    table("tblFeatures", fHeaders, fRows, fOpt);
    table("tblFeatures2", fHeaders, fRows, fOpt);

    // stage별 상대 Feature
    const sr = R.stage_rel.slice().sort((a, b) =>
      ({ normal: 0, warning: 1, anomaly: 2 })[a.stage] - ({ normal: 0, warning: 1, anomaly: 2 })[b.stage]);
    chart("chartStageRel", {
      type: "bar",
      data: {
        labels: sr.map((s) => s.stage),
        datasets: [
          { label: "load_vs_normal", data: sr.map((s) => s.load_vs_normal), backgroundColor: PALETTE.green, borderRadius: 3 },
          { label: "rpm_std_vs_normal", data: sr.map((s) => s.rpm_std_vs_normal), backgroundColor: PALETTE.blue, borderRadius: 3 },
          { label: "rpm_error_vs_normal", data: sr.map((s) => s.rpm_error_vs_normal), backgroundColor: PALETTE.violet, borderRadius: 3 }
        ]
      },
      options: {
        scales: { y: { beginAtZero: true, grid: { color: "#eef1f6" } }, x: { grid: { display: false } } },
        plugins: { legend: { position: "bottom" } }
      }
    });

    table("tblStageRel",
      [{ t: "stage" }, { t: "load_vs_normal", n: true }, { t: "rpm_std_vs_normal", n: true }, { t: "rpm_error_vs_normal", n: true }],
      sr.map((s) => [s.stage, s.load_vs_normal, s.rpm_std_vs_normal, s.rpm_error_vs_normal]),
      { caption: "output/training_dataset.csv 재계산 (4,560 Window)" });

    // split × label
    const sl = R.td_split_label;
    table("tblTdSplit",
      [{ t: "split" }, { t: "Window", n: true }, { t: "label 0", n: true }, { t: "label 1", n: true }, { t: "이상 비율", n: true }],
      ["train", "validation", "test"].map((k) => {
        const d = sl[k];
        const n = d["0"] + d["1"];
        return [k, n, d["0"], d["1"], ((d["1"] / n) * 100).toFixed(1) + "%"];
      }).concat([[
        "합계", R.td_shape[0], R.td_label["0"], R.td_label["1"],
        ((R.td_label["1"] / R.td_shape[0]) * 100).toFixed(1) + "%"
      ]]),
      { rowClass: (i) => (i === 3 ? "total-row" : "") });

    // seed 배정
    const cr = N.seedSplit.cross;
    table("tblSeedCross",
      [{ t: "split" }, { t: "load_rise", n: true }, { t: "rpm_instability", n: true },
       { t: "rpm_error_drift", n: true }, { t: "합계", n: true }],
      ["train", "validation", "test"].map((k) => [
        k, cr[k].load_rise, cr[k].rpm_instability, cr[k].rpm_error_drift,
        cr[k].load_rise + cr[k].rpm_instability + cr[k].rpm_error_drift
      ]).concat([[
        "합계", N.seedSplit.byType.load_rise, N.seedSplit.byType.rpm_instability,
        N.seedSplit.byType.rpm_error_drift, N.seedSplit.total
      ]]),
      { rowClass: (i) => (i === 3 ? "total-row" : ""),
        caption: "split 내부 라운드로빈 배정으로 유형 편중을 방지했습니다." });

    const bt = N.seedSplit.byTool;
    table("tblSeedTool",
      [{ t: "공구" }, { t: "seed 수", n: true }, { t: "비고" }],
      Object.keys(bt).map(Number).sort((a, b) => a - b).map((k) => [
        "공구 " + k, bt[k],
        k === 15 || k === 16 ? "seed가 전부 validation 기간 → 최종 제외" : ""
      ]),
      { rowClass: (i, r) => (r[2] ? "is-hl" : "") });
  }

  /* ==================================================================== */
  /* 14-15. 학습 / 평가                                                    */
  /* ==================================================================== */
  function renderTrain() {
    const pHeaders = [{ t: "파라미터" }, { t: "값", n: true }, { t: "의도" }];
    const pRows = N.xgbParams.params.map((p) => [p[0], p[1], p[2]]);
    const pOpt = { rowClass: (i, r) => (r[0] === "eval_metric" || r[0] === "early_stopping_rounds" ? "is-hl" : "") };
    table("tblXgbParams", pHeaders, pRows, pOpt);
    table("tblXgbParams2", pHeaders, pRows, pOpt);

    const m = N.metrics;
    const mHeaders = [{ t: "지표" }, { t: "TRAIN", n: true }, { t: "VALIDATION", n: true }, { t: "TEST", n: true }];
    const keys = [["n", "표본 수"], ["acc", "Accuracy"], ["prec", "Precision"],
                  ["rec", "Recall"], ["f1", "F1-score"], ["prauc", "PR-AUC"]];
    const mRows = keys.map((k) => [
      k[1],
      k[0] === "n" ? fmt(m.train.n) : m.train[k[0]].toFixed(4),
      k[0] === "n" ? fmt(m.validation.n) : m.validation[k[0]].toFixed(4),
      k[0] === "n" ? fmt(m.test.n) : m.test[k[0]].toFixed(4)
    ]);
    mRows.push(["오분류 건수",
      m.train.cm[0][1] + m.train.cm[1][0],
      m.validation.cm[0][1] + m.validation.cm[1][0],
      m.test.cm[0][1] + m.test.cm[1][0]]);
    const mOpt = {
      caption: "모든 수치는 합성한 가상 이상 3종을 정상과 구분하는 성능입니다. 실제 설비 고장 예측 정확도가 아닙니다."
    };
    table("tblMetrics", mHeaders, mRows, mOpt);
    table("tblMetrics2", mHeaders, mRows, mOpt);

    // Confusion matrices
    function cm(id, mm) {
      const el = $(id);
      if (!el) return;
      const [[tn, fp], [fn, tp]] = mm;
      el.innerHTML =
        `<div class="cm-grid">
          <div class="cm-h"></div><div class="cm-h">예측 정상</div><div class="cm-h">예측 이상</div>
          <div class="cm-h">실제 정상</div><div class="cm-tn">TN ${fmt(tn)}</div><div class="${fp ? "cm-fp" : "cm-zero"}">FP ${fmt(fp)}</div>
          <div class="cm-h">실제 이상</div><div class="${fn ? "cm-fn" : "cm-zero"}">FN ${fmt(fn)}</div><div class="cm-tp">TP ${fmt(tp)}</div>
        </div>`;
    }
    cm("cmTrain", m.train.cm);
    cm("cmVal", m.validation.cm);
    cm("cmTest", m.test.cm);

    // TEST 유형별
    const bt = N.testDetail.byType;
    chart("chartTestType", {
      type: "bar",
      data: {
        labels: bt.map((b) => b.name),
        datasets: [{
          data: bt.map((b) => b.rate * 100),
          backgroundColor: [PALETTE.grey, PALETTE.green, PALETTE.amber, PALETTE.violet],
          borderRadius: 4
        }]
      },
      options: {
        indexAxis: "y",
        scales: { x: { min: 95, max: 100, grid: { color: "#eef1f6" }, ticks: { callback: (v) => v + "%" } },
                  y: { grid: { display: false } } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => ` ${c.parsed.x.toFixed(2)}% (n=${bt[c.dataIndex].n})` } }
        }
      }
    });

    table("tblTestStage",
      [{ t: "stage" }, { t: "Window", n: true }, { t: "정답 label", n: true }, { t: "정탐률", n: true }],
      N.testDetail.byStage.map((s) => [s.name, s.n, s.target, (s.rate * 100).toFixed(2) + "%"]),
      { rowClass: (i, r) => (r[0] === "warning" ? "is-hl" : ""),
        caption: "warning = 정상 경계에 가장 가까운 초기 이상 구간 (167/168 탐지)" });

    // Feature importance
    const imp = N.importance.rows;
    chart("chartImportance", {
      type: "bar",
      data: {
        labels: imp.map((r) => r.f + (r.rel ? " ★" : "")),
        datasets: [{
          data: imp.map((r) => r.v),
          backgroundColor: imp.map((r) => (r.rel ? PALETTE.violet : PALETTE.blue)),
          borderRadius: 3
        }]
      },
      options: {
        indexAxis: "y",
        scales: { x: { grid: { color: "#eef1f6" }, ticks: { callback: (v) => v + "%" } }, y: { grid: { display: false } } },
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (c) => ` gain ${c.parsed.x}%` + (imp[c.dataIndex].rel ? " · 상대 Feature" : "") } }
        }
      }
    });

    const iHeaders = [{ t: "순위", n: true }, { t: "Feature" }, { t: "gain %", n: true }, { t: "구분" }, { t: "주로 겨냥한 신호" }];
    const tgt = {
      rpm_error_absmax: "rpm_error_drift", load_vs_normal: "load_rise", load_max: "load_rise",
      rpm_error_vs_normal: "rpm_error_drift", load_mean: "load_rise", rpm_range: "rpm_instability",
      rpm_std: "rpm_instability", rpm_mean: "운전점 식별", rpm_max: "rpm_instability",
      load_std: "load_rise", rpm_min: "rpm_instability", rpm_std_vs_normal: "rpm_instability",
      rpm_error_mean: "rpm_error_drift", rpm_error_std: "rpm_error_drift / rpm_instability"
    };
    const iRows = imp.map((r, i) => [i + 1, r.f, r.v.toFixed(2), r.rel ? "★ 상대 Feature" : "기본", tgt[r.f] || ""]);
    const iOpt = {
      rowClass: (i) => (imp[i].rel ? "is-hl" : ""),
      caption: "상위 4개가 84.5%를 차지합니다. 상대 Feature 3종 합계는 29.36%입니다."
    };
    table("tblImportance", iHeaders, iRows, iOpt);
    table("tblImportance2", iHeaders, iRows, iOpt);
  }

  /* ==================================================================== */
  /* 부록 A. 데이터 계보                                                   */
  /* ==================================================================== */
  const LINEAGE = [
    ["원본 로그 라인", "31개 .log 파일", 6261455, "raw", "측정 1회가 13줄"],
    ["mct_parsed.csv", "13줄 → 1행 구조화 · 중간 산출물", 481648, "", "21열"],
    ["mct_clean.csv", "파생 열 + 상태 플래그 추가", 481648, "", "31열 · 행 수 보존"],
    ["STATUS_RUN", "운전 상태로 기록된 행", 243319, "", "전체의 50.5%"],
    ["실제 절삭 구간", "RUN & RPM>0 & FEED>0", 219195, "", "RUN의 90.09%"],
    ["연속 절삭 segment", "공구/시간/비절삭/날짜 기준 분리", 839, "", "행 합계 219,195 일치"],
    ["Synthetic Seed", "10분 이상 segment에서 1개씩", 462, "synth", "각 10분"],
    ["synthetic_anomaly_data.csv", "가상 이상 3종 생성", 48240, "synth", "38열 · 462 seed"],
    ["1분 Window", "seed당 10개", 4620, "synth", "stage 혼합 0건"],
    ["training_dataset.csv", "공구 15·16의 60 Window 제외", 4560, "model", "24열 · Feature 14종"],
    ["Train / Val / Test", "3,630 / 370 / 560", 4560, "model", "시간순 분할"]
  ];

  function renderLineage() {
    const el = $("lineageChart");
    if (!el) return;
    const max = Math.log10(6261455);
    el.innerHTML = LINEAGE.map((l) => {
      const w = (Math.log10(Math.max(l[2], 10)) / max) * 100;
      return `<div class="lin-row">
        <div class="lin-name">${l[0]}<small>${l[1]}</small></div>
        <div class="lin-bar-track"><div class="lin-bar ${l[3]}" style="width:${w.toFixed(1)}%"></div></div>
        <div class="lin-val">${fmt(l[2])}</div>
      </div>`;
    }).join("");

    table("tblLineage",
      [{ t: "단계" }, { t: "행 / 개수", n: true }, { t: "변환 내용" }, { t: "비고" }],
      LINEAGE.map((l) => [l[0], l[2], l[1], l[4]]),
      { caption: "막대 길이는 로그 스케일입니다 (626만 → 4,560까지의 규모 차이를 한 화면에 표시하기 위함)." });
  }

  /* ==================================================================== */
  /* 네비게이션 / UI                                                       */
  /* ==================================================================== */
  function initNav() {
    const links = Array.from(document.querySelectorAll(".nav-list a"));
    const sections = links
      .map((a) => document.querySelector(a.getAttribute("href")))
      .filter(Boolean);

    function onScroll() {
      const y = window.scrollY + 120;
      let idx = 0;
      sections.forEach((s, i) => { if (s.offsetTop <= y) idx = i; });
      links.forEach((a, i) => a.classList.toggle("active", i === idx));
    }
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();

    const toggle = $("navToggle");
    const sidebar = $("sidebar");
    if (toggle && sidebar) {
      toggle.addEventListener("click", () => sidebar.classList.toggle("open"));
      links.forEach((a) => a.addEventListener("click", () => sidebar.classList.remove("open")));
    }
  }

  /* ==================================================================== */
  /* 실행                                                                  */
  /* ==================================================================== */
  function init() {
    if (!R || !N) {
      console.error("데이터 파일(assets/data.js, assets/notebook_data.js)을 불러오지 못했습니다.");
      return;
    }
    renderKpi();
    renderFlow("flowDiagram", FLOW);
    renderFlow("inferFlow", INFER_FLOW);
    renderRawLog();
    renderValidate();
    renderEda();
    bindDescTabs("descTabs", "tblDesc", "descTabDesc", "descSrc");
    bindDescTabs("descTabs2", "tblDesc2", null, "descSrc2");
    renderCutting();
    renderTools();
    renderRpmError();
    renderWindow();
    renderAlarm();
    renderSynthetic();
    renderFeature();
    renderTrain();
    renderLineage();
    initNav();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
