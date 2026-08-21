# -*- coding: utf-8 -*-
"""
M2* vs XGBoost 최종 비교 화면
- 공통 전처리 적용 후 생성한 비교 결과 CSV를 읽어서 표시
- 실제 고장 라벨이 없으므로 실제 로그 결과는 '알람/탐지 후보'로 표현
"""

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
RESULT_DIR = PROJECT_ROOT / "reports" / "model_comparison"

T = {
    "blue": "#3987e5",
    "orange": "#d95926",
    "aqua": "#199e70",
    "yellow": "#c98500",
    "critical": "#d03b3b",
    "good": "#0ca30c",
    "ink": "#ffffff",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#232322",
    "axis": "#383835",
    "surface": "#1a1a19",
}

FONT = '"Malgun Gothic", "Segoe UI", system-ui, sans-serif'
NOBAR = {"displayModeBar": False}


def _kpi_row(items):
    tiles = []
    for it in items:
        dot = (
            f'<span style="display:inline-block;width:11px;height:11px;'
            f'border-radius:50%;background:{it["dot"]};margin-right:7px;"></span>'
            if it.get("dot")
            else ""
        )
        unit = (
            f'<span style="font-size:.95rem;font-weight:500;color:{T["ink2"]};">'
            f'{it["unit"]}</span>'
            if it.get("unit")
            else ""
        )
        sub = (
            f'<div style="font-size:.74rem;color:{T["muted"]};margin-top:5px;">'
            f'{it["sub"]}</div>'
            if it.get("sub")
            else ""
        )
        tiles.append(
            f'<div style="flex:1 1 0;min-width:135px;background:{T["surface"]};'
            'border:1px solid rgba(255,255,255,.08);border-radius:12px;'
            'padding:12px 14px 10px;">'
            f'<div style="font-size:.79rem;color:{T["ink2"]};margin-bottom:5px;">'
            f'{it["label"]}</div>'
            f'<div style="font-size:1.55rem;font-weight:600;color:{T["ink"]};'
            f'line-height:1.15;">{dot}{it["value"]} {unit}</div>{sub}</div>'
        )
    st.markdown(
        '<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px;">'
        + "".join(tiles)
        + "</div>",
        unsafe_allow_html=True,
    )


def _style(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=12, t=40, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=T["ink2"]),
        legend=dict(
            orientation="h",
            x=1,
            xanchor="right",
            y=1.02,
            yanchor="bottom",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(gridcolor=T["grid"], linecolor=T["axis"], zeroline=False)
    fig.update_yaxes(gridcolor=T["grid"], linecolor=T["axis"], zeroline=False)
    return fig


@st.cache_data(show_spinner=False)
def _load_results():
    summary_path = RESULT_DIR / "comparison_summary_common.csv"
    detail_path = RESULT_DIR / "comparison_calibrated_p99_common.csv"
    synthetic_path = RESULT_DIR / "synthetic_same_scenario_comparison.csv"

    missing = [
        p for p in [summary_path, detail_path, synthetic_path]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "필요한 비교 결과 파일이 없습니다:\n"
            + "\n".join(str(p) for p in missing)
        )

    summary = pd.read_csv(summary_path)
    detail = pd.read_csv(detail_path)
    synthetic = pd.read_csv(synthetic_path)

    return summary, detail, synthetic


def render_comparison_view():
    st.markdown("##### 모델 비교")
    st.caption(
        "비지도학습 M2*와 지도학습 XGBoost를 공통 전처리 기준으로 비교합니다. "
        "실제 로그에는 고장 정답 라벨이 없으므로 아래 결과는 정확도 비교가 아니라 "
        "두 모델이 어떤 구간에 서로 다르게 반응했는지를 확인하는 화면입니다."
    )

    st.markdown(
        f"""
<div style="
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:12px;
    margin:.8rem 0 1rem;
">
  <div style="background:{T['surface']};border:1px solid rgba(255,255,255,.08);
              border-radius:12px;padding:14px 16px;">
    <div style="font-size:.76rem;color:{T['muted']};margin-bottom:5px;">M2*</div>
    <div style="font-size:1rem;font-weight:700;color:{T['ink']};">정상 기준 이탈 감시</div>
    <div style="font-size:.78rem;color:{T['ink2']};margin-top:6px;line-height:1.5;">
      정상 운전조건에서 벗어나는 변화와 지속성을 중심으로 봅니다.
    </div>
  </div>
  <div style="background:{T['surface']};border:1px solid rgba(255,255,255,.08);
              border-radius:12px;padding:14px 16px;">
    <div style="font-size:.76rem;color:{T['muted']};margin-bottom:5px;">XGBoost</div>
    <div style="font-size:1rem;font-weight:700;color:{T['ink']};">학습 패턴 유사도 분석</div>
    <div style="font-size:.78rem;color:{T['ink2']};margin-top:6px;line-height:1.5;">
      학습한 가상 이상 패턴과 유사한 1분 절삭 Window를 찾습니다.
    </div>
  </div>
  <div style="background:{T['surface']};border:1px solid rgba(255,255,255,.08);
              border-radius:12px;padding:14px 16px;">
    <div style="font-size:.76rem;color:{T['muted']};margin-bottom:5px;">해석 기준</div>
    <div style="font-size:1rem;font-weight:700;color:{T['ink']};">많이 탐지 = 더 정확함 아님</div>
    <div style="font-size:.78rem;color:{T['ink2']};margin-top:6px;line-height:1.5;">
      실제 고장 라벨이 없어 Precision·Recall 같은 현장 정확도는 계산할 수 없습니다.
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    try:
        summary, detail, synthetic = _load_results()
    except Exception as exc:
        st.error(f"비교 결과를 불러오지 못했습니다: {exc}")
        return

    # --------------------------------------------------
    # 1. 실제 시험 로그 비교
    # --------------------------------------------------
    p99 = summary[
        summary["비교 기준"].astype(str).str.contains(
            "calibration P99", na=False
        )
    ]

    if p99.empty:
        p99 = summary.iloc[[0]]

    row = p99.iloc[0]

    xgb_ep = int(row["XGBoost Episode"])
    m2_ep = int(row["M2* Episode"])
    xgb_overlap = int(row["XGBoost 중 겹침"])
    m2_overlap = int(row["M2* 중 겹침"])
    threshold = float(row["XGBoost 기준"])

    _kpi_row(
        [
            {
                "label": "XGBoost 고득점 Episode",
                "value": f"{xgb_ep}",
                "unit": "건",
                "dot": T["blue"],
                "sub": f"5/22–5/31 · calibration P99 {threshold:.6f}",
            },
            {
                "label": "M2* 알람",
                "value": f"{m2_ep}",
                "unit": "건",
                "dot": T["aqua"],
                "sub": "5/22–5/31 시험기간",
            },
            {
                "label": "XGBoost 중 M2*와 겹침",
                "value": f"{xgb_overlap}",
                "unit": "건",
                "sub": "같은 공구 · ±90초 후보",
            },
            {
                "label": "M2* 중 XGBoost와 겹침",
                "value": f"{m2_overlap}",
                "unit": "건",
                "sub": "같은 공구 · ±90초 후보",
            },
        ]
    )

    with st.container(border=True):
        st.markdown("##### 실제 로그 반응 비교")
        st.caption(
            "동일 시험기간에서 XGBoost와 M2*가 어느 구간에 반응했는지를 비교합니다. "
            "반응 후보의 수는 모델의 우열이나 실제 고장 적중 수를 의미하지 않습니다."
        )

        comparison_labels = {
            "XGBoost만 탐지": "XGBoost 추가 반응 후보",
            "둘 다 탐지": "두 모델 공통 반응 후보",
            "M2*만 탐지": "M2* 추가 반응 후보",
        }

        counts = detail["comparison"].value_counts()

        count_df = pd.DataFrame(
            [
                {
                    "구분": comparison_labels["XGBoost만 탐지"],
                    "매칭 행 수": int(counts.get("XGBoost만 탐지", 0)),
                },
                {
                    "구분": comparison_labels["둘 다 탐지"],
                    "매칭 행 수": int(counts.get("둘 다 탐지", 0)),
                },
                {
                    "구분": comparison_labels["M2*만 탐지"],
                    "매칭 행 수": int(counts.get("M2*만 탐지", 0)),
                },
            ]
        )

        c1, c2 = st.columns([1.15, 1])

        with c1:
            if go is not None:
                fig = go.Figure(
                    go.Bar(
                        x=count_df["구분"],
                        y=count_df["매칭 행 수"],
                        marker_color=[
                            T["blue"],
                            T["yellow"],
                            T["aqua"],
                        ],
                        text=count_df["매칭 행 수"],
                        textposition="outside",
                        hovertemplate="%{x}<br>%{y}행<extra></extra>",
                    )
                )
                fig.update_yaxes(rangemode="tozero", title="매칭 행 수")
                st.plotly_chart(
                    _style(fig, 300),
                    width="stretch",
                    config=NOBAR,
                )
            else:
                st.dataframe(count_df, width="stretch", hide_index=True)

        with c2:
            tool_col = "tool" if "tool" in detail.columns else None

            if tool_col:
                tool_table = (
                    detail.groupby(["comparison", tool_col])
                    .size()
                    .reset_index(name="건수")
                    .rename(
                        columns={
                            "comparison": "구분",
                            tool_col: "공구",
                        }
                    )
                )
                tool_table["구분"] = tool_table["구분"].replace(
                    comparison_labels
                )
                tool_table["공구"] = tool_table["공구"].map(
                    lambda x: f"Tool {int(x)}" if pd.notna(x) else "-"
                )
                st.dataframe(
                    tool_table,
                    width="stretch",
                    hide_index=True,
                )

        st.caption(
            "※ '두 모델 공통 반응 후보'의 막대는 매칭 행 수입니다. "
            "한 M2* 알람이 여러 XGBoost Episode와 겹칠 수 있으므로 고유 건수와 다를 수 있습니다. "
            f"고유 겹침은 XGBoost {xgb_overlap}건 / M2* {m2_overlap}건입니다."
        )

    # --------------------------------------------------
    # 2. 동일 합성 이상 비교
    # --------------------------------------------------
    with st.container(border=True):
        st.markdown("##### 모델 개발용 동일 합성 이상 실험")
        st.caption(
            "공통 전처리 후 같은 seed와 같은 주입 위치에서 비교한 모델 개발용 실험입니다. "
            "실제 설비 고장 성능이나 현장 정확도를 의미하지 않습니다."
        )

        syn = synthetic.copy()

        m2_rate_col = "M2*_탐지율"
        xgb_rate_col = "XGBoost_탐지율_분석가능기준"

        if (
            go is not None
            and m2_rate_col in syn.columns
            and xgb_rate_col in syn.columns
        ):
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=syn["이상 유형"],
                    y=syn[m2_rate_col],
                    name="M2*",
                    marker_color=T["aqua"],
                    hovertemplate="%{x}<br>M2* %{y:.1f}%<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=syn["이상 유형"],
                    y=syn[xgb_rate_col],
                    name="XGBoost",
                    marker_color=T["blue"],
                    hovertemplate="%{x}<br>XGBoost %{y:.1f}%<extra></extra>",
                )
            )
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0, 110], title="탐지율 (%)")
            st.plotly_chart(
                _style(fig, 330),
                width="stretch",
                config=NOBAR,
            )

        cols = [
            "이상 유형",
            "M2*_탐지n",
            "M2*_탐지율",
            "M2*_평균탐지초",
            "XGBoost_분석가능n",
            "XGBoost_분석불가n",
            "XGBoost_탐지n",
            "XGBoost_탐지율_분석가능기준",
            "XGBoost_평균탐지초",
        ]
        cols = [c for c in cols if c in syn.columns]

        display = syn[cols].copy()
        display = display.rename(
            columns={
                "M2*_탐지n": "M2* 탐지",
                "M2*_탐지율": "M2* 탐지율(%)",
                "M2*_평균탐지초": "M2* 평균 탐지(초)",
                "XGBoost_분석가능n": "XGB 분석 가능",
                "XGBoost_분석불가n": "XGB 분석 불가",
                "XGBoost_탐지n": "XGB 탐지",
                "XGBoost_탐지율_분석가능기준": "XGB 탐지율(%)",
                "XGBoost_평균탐지초": "XGB 평균 탐지(초)",
            }
        )

        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "평균 탐지시간은 M2*가 cycle 단위, XGBoost가 1분 Window 단위이므로 "
            "절대적인 속도 우열로 직접 비교하지 않습니다."
        )

    # --------------------------------------------------
    # 3. 최종 해석
    # --------------------------------------------------
    with st.container(border=True):
        st.markdown("##### 현재 데이터 기준 활용 제안")

        st.markdown(
            f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:.6rem;">
  <div style="background:{T['surface']};border:1px solid rgba(25,158,112,.42);
              border-radius:12px;padding:16px 18px;">
    <div style="font-size:.78rem;color:{T['aqua']};font-weight:700;margin-bottom:5px;">
      MAIN ANALYSIS
    </div>
    <div style="font-size:1.05rem;font-weight:800;color:{T['ink']};margin-bottom:9px;">
      비지도학습 M2*
    </div>
    <div style="color:{T['ink2']};font-size:.84rem;line-height:1.75;">
      • 정상 운전조건에서 얼마나 벗어났는지 지속적으로 감시<br>
      • 합성 공구 파손형 88.3%, 지속 과부하·완만한 저하 100%<br>
      • 시험 로그 알람 후보 빈도 1.4건/일<br>
      • 학습하지 않은 형태도 정상 기준 이탈 관점으로 확인
    </div>
  </div>

  <div style="background:{T['surface']};border:1px solid rgba(57,135,229,.42);
              border-radius:12px;padding:16px 18px;">
    <div style="font-size:.78rem;color:{T['blue']};font-weight:700;margin-bottom:5px;">
      SUPPORT ANALYSIS
    </div>
    <div style="font-size:1.05rem;font-weight:800;color:{T['ink']};margin-bottom:9px;">
      지도학습 XGBoost
    </div>
    <div style="color:{T['ink2']};font-size:.84rem;line-height:1.75;">
      • 학습한 가상 이상 패턴과의 유사도를 1분 Window 단위로 분석<br>
      • 지속 과부하 100%, 완만한 저하 98.2% (분석 가능 구간 기준)<br>
      • 합성 공구 파손형은 9.4%로 상대적으로 약함<br>
      • Feature Contribution으로 높은 점수의 판단 근거 확인
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.info(
            "현재 데이터와 모델 개발 실험을 기준으로는 M2*를 주 분석 방식, "
            "XGBoost를 학습된 이상 패턴 및 Feature 해석을 위한 보조 분석 방식으로 제안합니다. "
            "실제 고장 라벨이 확보되면 두 모델의 현장 성능을 다시 검증해야 합니다."
        )
