# -*- coding: utf-8 -*-
"""MCT 3호 예지보전 PoC 대시보드 (산학 프로젝트)
실행:  streamlit run app.py
데이터: data/ 폴더의 사전 계산 결과만 읽는다. (모델 연산 없음 → 어떤 PC에서든 즉시 구동)
"""
import json, os, time, datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="MCT 3호 예지보전 PoC", page_icon="🛠️", layout="wide")

# ── 디자인 토큰 (다크 서피스 기준 검증 팔레트) ────────────────────────────────
T = dict(
    blue="#3987e5", orange="#d95926", aqua="#199e70", yellow="#c98500",
    critical="#d03b3b", warn="#fab219", good="#0ca30c",
    ink="#ffffff", ink2="#c3c2b7", muted="#898781",
    grid="#232322", axis="#383835",
    band="rgba(57,135,229,0.16)", surface="#1a1a19",
)
FONT = '"Malgun Gothic", "Segoe UI", system-ui, sans-serif'
NOBAR = {"displayModeBar": False}

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.4rem; max-width: 1240px; }}
h1 {{ font-size: 1.7rem !important; letter-spacing: -0.01em; }}
.chips {{ display:flex; gap:8px; flex-wrap:wrap; margin: 2px 0 6px; }}
.chip {{ font-size:.78rem; color:{T['ink2']}; background:{T['surface']};
        border:1px solid rgba(255,255,255,.08); border-radius:999px; padding:3px 11px; }}
.kpi-row {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom: 4px; }}
.kpi {{ flex:1 1 0; min-width:118px; background:{T['surface']};
       border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:12px 14px 10px; }}
.kpi-label {{ font-size:.79rem; color:{T['ink2']}; margin-bottom:5px; }}
.kpi-value {{ font-size:1.62rem; font-weight:600; color:{T['ink']};
             line-height:1.15; display:flex; align-items:baseline; gap:7px; white-space:nowrap; }}
.kpi-unit {{ font-size:.95rem; font-weight:500; color:{T['ink2']}; }}
.kpi-sub {{ font-size:.74rem; color:{T['muted']}; margin-top:5px; }}
.dot {{ width:11px; height:11px; border-radius:50%; display:inline-block; align-self:center; }}
[data-testid="stDataFrame"] * {{ font-variant-numeric: tabular-nums; }}
</style>
""", unsafe_allow_html=True)


def kpi_row(items):
    """스탯 타일 한 줄. item: label, value, unit?, sub?, dot?(상태색), small?(긴 텍스트값용 축소 폰트)"""
    tiles = []
    for it in items:
        dot = f'<span class="dot" style="background:{it["dot"]}"></span>' if it.get("dot") else ""
        unit = f'<span class="kpi-unit">{it["unit"]}</span>' if it.get("unit") else ""
        sub = f'<div class="kpi-sub">{it["sub"]}</div>' if it.get("sub") else ""
        fs = ' style="font-size:1.22rem"' if it.get("small") else ""
        tiles.append(f'<div class="kpi"><div class="kpi-label">{it["label"]}</div>'
                     f'<div class="kpi-value"{fs}>{dot}{it["value"]}{unit}</div>{sub}</div>')
    st.markdown(f'<div class="kpi-row">{"".join(tiles)}</div>', unsafe_allow_html=True)


def style(fig, h, legend=True, top=48):
    """공통 차트 스타일 — 투명 배경, 헤어라인 그리드, 우상단 가로 범례."""
    fig.update_layout(
        height=h, margin=dict(l=8, r=12, t=top, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12.5, color=T["ink2"]),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=T["surface"], bordercolor="rgba(255,255,255,.15)",
                        font=dict(family=FONT, size=12, color=T["ink"])),
        showlegend=legend,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11.5, color=T["ink2"]),
                    itemwidth=30),
    )
    fig.update_xaxes(gridcolor=T["grid"], linecolor=T["axis"], zeroline=False,
                     tickfont=dict(size=11, color=T["muted"]))
    fig.update_yaxes(gridcolor=T["grid"], linecolor=T["axis"], zeroline=False,
                     tickfont=dict(size=11, color=T["muted"]))
    # 서브플롯 제목(가운데 정렬로 생성됨) → 좌측 정렬 소제목
    for a in fig.layout.annotations:
        if a.xref == "paper" and a.x == 0.5 and a.xanchor == "center":
            a.update(x=0, xanchor="left", font=dict(size=13, color=T["ink"], family=FONT))
    return fig


@st.cache_data(show_spinner="데이터 불러오는 중…")
def load():
    df = pd.read_csv("data/scored.csv.gz", parse_dates=["ts"])
    alarms = pd.read_csv("data/alarms.csv", parse_dates=["시작", "종료"])
    novel = pd.read_csv("data/novel.csv", parse_dates=["시작", "종료"])
    events = pd.read_csv("data/events.csv", parse_dates=["시각"])
    daily = pd.read_csv("data/daily.csv", parse_dates=["date"])
    meta = json.load(open("data/meta.json", encoding="utf-8"))
    return df, alarms, novel, events, daily, meta

df, alarms, novel, events, daily, meta = load()
DATES = sorted(df.ts.dt.date.unique())
WD = "월화수목금토일"

st.title("MCT 3호 예지보전 PoC 대시보드")
st.markdown(
    '<div class="chips">'
    '<span class="chip">분석 기간 2026-05-01 – 05-31 (31일)</span>'
    f'<span class="chip">수집 데이터 {len(df):,} 사이클</span>'
    f'<span class="chip">정상 기준 학습 {meta["train_range"].replace("~", "–")}</span>'
    '</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["① 실시간 관제 (리플레이)", "② 이상 알람", "③ 월간 운영 리포트", "④ 분석 방법과 신뢰성"])

# ──────────────────────────────── ① 실시간 관제 ────────────────────────────────
STATE_KR = {"steady": ("가공 중", T["good"]), "transient": ("가감속", T["warn"]),
            "ready": ("대기", T["muted"]), "error": ("알람", T["critical"])}

def monitor_figure(day_df, t_cur, win_min=30, alarm_spans=None):
    w = day_df[(day_df.ts <= t_cur) & (day_df.ts > t_cur - pd.Timedelta(minutes=win_min))]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                        vertical_spacing=0.10,
                        subplot_titles=("스핀들 부하 (%)", "스핀들 회전수 (rpm)"))
    sd = w[w.state == "steady"]
    if len(sd):  # 기대 부하 밴드 (컨텍스트 기대치 ±3σ) — 은은한 워시
        fig.add_trace(go.Scatter(x=sd.ts, y=(sd.ms + 3*sd.ss), line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"), 1, 1)
        fig.add_trace(go.Scatter(x=sd.ts, y=(sd.ms - 3*sd.ss).clip(lower=0), fill="tonexty",
                                 fillcolor=T["band"], line=dict(width=0), name="기대 범위",
                                 hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=w.ts, y=w.sload, name="부하",
                             line=dict(color=T["blue"], width=2, shape="spline", smoothing=0.4),
                             hovertemplate="부하 %{y:.0f}%<extra></extra>"), 1, 1)
    tr_ = w[w.state == "transient"]
    if len(tr_):
        fig.add_trace(go.Scatter(x=tr_.ts, y=tr_.sload, mode="markers", name="가감속 중",
                                 marker=dict(color=T["yellow"], size=7, symbol="diamond",
                                             line=dict(color="#0d0d0d", width=1)),
                                 hovertemplate="가감속 중 %{y:.0f}%<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=w.ts, y=w.rpm, name="실제 rpm",
                             line=dict(color=T["aqua"], width=2),
                             hovertemplate="실제 %{y:,.0f} rpm<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=w.ts, y=w.srate, name="지령 rpm",
                             line=dict(color=T["muted"], width=1.2, dash="dot"),
                             hovertemplate="지령 %{y:,.0f} rpm<extra></extra>"), 2, 1)
    # 알람 구간: 시작 세로선 + 지속 구간 워시
    spans = alarm_spans if alarm_spans is not None else \
        alarms[(alarms.시작 <= t_cur) & (alarms.종료 >= t_cur - pd.Timedelta(minutes=win_min))]
    for _, a in spans.iterrows():
        fig.add_vrect(x0=a.시작, x1=a.종료, fillcolor="rgba(208,59,59,0.14)", line_width=0)
        fig.add_vline(x=a.시작, line_color=T["critical"], line_width=1.5, line_dash="dash")
    fig.update_xaxes(tickformat="%H:%M")
    fig.update_yaxes(rangemode="tozero")
    return style(fig, 440, top=56)

with tab1:
    left, right = st.columns([1, 3], gap="large")
    with left:
        day = st.selectbox("날짜", DATES, index=DATES.index(dt.date(2026, 5, 12)),
                           format_func=lambda d: f"{d:%m월 %d일} ({WD[d.weekday()]})", key="mon_day")
        tsel = st.slider("시작 시각", value=dt.time(19, 30), step=dt.timedelta(minutes=5),
                         format="HH:mm", key="mon_time")
        speed = st.select_slider("재생 속도", ["느리게", "보통", "빠르게"], value="보통")
        play = st.toggle("▶ 리플레이 재생", key="play")
        st.caption("5월 기록을 관제 화면처럼 재생합니다. 빨간 점선·음영이 알람 지점과 지속 구간.")
    day_df = df[df.ts.dt.date == day]
    t0 = pd.Timestamp.combine(day, tsel)
    ph = right.empty()

    def draw(t_cur):
        with ph.container():
            upto = day_df[day_df.ts <= t_cur]
            cur = upto.iloc[-1] if len(upto) else day_df.iloc[0]
            run_pct = (upto.status == "STATUS_RUN").mean() * 100 if len(upto) else 0
            label, color = STATE_KR[cur.state]
            kpi_row([
                dict(label="설비 상태", value=label, dot=color),
                dict(label="현재 공구", value=f"T{int(cur.tool):02d}"),
                dict(label="스핀들 회전수", value=f"{cur.rpm:,.0f}", unit="rpm"),
                dict(label="스핀들 부하", value=f"{cur.sload:.0f}", unit="%"),
                dict(label="금일 가동률", value=f"{run_pct:.1f}", unit="%", sub="00시부터 현재까지"),
            ])
            st.plotly_chart(monitor_figure(day_df, t_cur), width="stretch", config=NOBAR)

    if play:
        delay = {"느리게": 1.0, "보통": 0.5, "빠르게": 0.15}[speed]
        t_cur, end = t0, day_df.ts.max()
        while st.session_state.get("play", False) and t_cur < end:
            draw(t_cur); t_cur += pd.Timedelta(minutes=1); time.sleep(delay)
    else:
        draw(t0)

# ──────────────────────────────── ② 이상탐지 ────────────────────────────────
with tab2:
    with st.container(border=True):
        st.markdown("##### 5월 전체 이상 점수 추이")
        st.caption("같은 공구·회전수 조건의 평소 부하 대비 이탈 점수. 점선을 넘는 상태가 이어지면 알람, "
                   "회색 음영은 기준값을 뽑은 학습 구간.")
        sd_all = df[df.state == "steady"].set_index("ts")
        z5 = sd_all.zload.abs().resample("5min").max().dropna()
        e5 = sd_all.ewma.abs().resample("5min").max().dropna()
        ymax = float(max(z5.max(), e5.max()) * 1.02)
        fig = go.Figure()
        fig.add_vrect(x0="2026-05-01", x1="2026-05-22", fillcolor="rgba(137,135,129,0.09)",
                      line_width=0)
        fig.add_trace(go.Scatter(x=z5.index, y=z5.values, name=f"순간 변동 (기준 {meta['thr_z']:.1f})",
                                 line=dict(color=T["blue"], width=1.3),
                                 hovertemplate="%{y:.1f}<extra>순간 변동</extra>"))
        fig.add_trace(go.Scatter(x=e5.index, y=e5.values, name=f"점진적 변화 (기준 {meta['thr_ew']:.1f})",
                                 line=dict(color=T["aqua"], width=1.3),
                                 hovertemplate="%{y:.1f}<extra>점진적 변화</extra>"))
        fig.add_hline(y=meta["thr_z"], line_dash="dash", line_color=T["blue"], line_width=1)
        fig.add_hline(y=meta["thr_ew"], line_dash="dash", line_color=T["aqua"], line_width=1)
        if len(alarms):
            fig.add_trace(go.Scatter(x=alarms.시작, y=[ymax * 1.06] * len(alarms), mode="markers",
                                     name=f"알람 ({len(alarms)}건)",
                                     marker=dict(color=T["critical"], symbol="triangle-down", size=9),
                                     customdata=alarms.설명, hovertemplate="%{customdata}<extra></extra>"))
        fig.update_yaxes(range=[0, ymax * 1.14], title=None)
        fig.update_xaxes(tickformat="%m/%d")
        st.plotly_chart(style(fig, 330), width="stretch", config=NOBAR)

    st.markdown(f"##### 이상 알람 내역 — 총 {len(alarms)}건")
    st.caption("순간: 부하가 기준을 크게 벗어난 상태가 30초 이상 지속 · 드리프트: 평소 수준에서 서서히 멀어지는 변화")
    ch = st.multiselect("알람 유형", ["순간", "드리프트"], default=["순간", "드리프트"])
    view = alarms[alarms.채널.apply(lambda s: any(c in s for c in ch))]
    st.dataframe(
        view[["시작", "지속초", "채널", "공구", "RPM대역", "기대부하", "실측", "최대z", "설명"]],
        width="stretch", hide_index=True,
        height=min(423, 35 * (len(view) + 1) + 3),
        column_config={
            "시작": st.column_config.DatetimeColumn("시작", format="MM-DD HH:mm:ss", width="medium"),
            "지속초": st.column_config.NumberColumn("지속(초)", width="small"),
            "채널": st.column_config.TextColumn("유형", width="small"),
            "기대부하": st.column_config.NumberColumn("평소 부하(%)", format="%.1f", width="small"),
            "최대z": st.column_config.NumberColumn("이탈 크기", format="%.1f", width="small",
                                                  help="평소 대비 벗어난 정도 (z점수)"),
            "설명": st.column_config.TextColumn("설명", width="large"),
        })

    if len(view):
        pick = st.selectbox("알람 상세 보기", view.index,
                            format_func=lambda i: f"{view.loc[i, '시작']:%m-%d %H:%M} · {view.loc[i, '설명']}")
        a = view.loc[pick]
        with st.container(border=True):
            kpi_row([
                dict(label="알람 유형", value=a.채널, dot=T["critical"], small=True),
                dict(label="운전 조건", value=f"{a.공구} · {a.RPM대역}", unit="rpm", small=True),
                dict(label="평소 부하", value=f"{a.기대부하:.0f}", unit="%"),
                dict(label="실측 부하", value=a.실측, small=True),
                dict(label="지속 시간", value=f"{a.지속초:,}", unit="초"),
                dict(label="이탈 크기", value=f"{a.최대z:.1f}", sub="평소 대비 벗어난 정도"),
            ])
            around = df[(df.ts >= a.시작 - pd.Timedelta("10min")) & (df.ts <= a.종료 + pd.Timedelta("10min"))]
            span_min = (around.ts.max() - around.ts.min()).total_seconds() / 60 + 1
            st.plotly_chart(monitor_figure(around, around.ts.max(), win_min=span_min,
                                           alarm_spans=view.loc[[pick]]),
                            width="stretch", config=NOBAR)

    dt_cfg = {c: st.column_config.DatetimeColumn(c, format="MM-DD HH:mm:ss") for c in ["시작", "종료", "시각"]}
    c1, c2 = st.columns(2)
    with c1.expander(f"새로운 운전조건 감지 {len(novel)}건 — 학습 기간에 없던 작업 조건 (고장 신호 아님)"):
        st.dataframe(novel, width="stretch", hide_index=True, column_config=dt_cfg)
    with c2.expander(f"설비 컨트롤러 이벤트 {len(events)}건 — 기계가 직접 남긴 알람·비상신호 원본"):
        st.dataframe(events, width="stretch", hide_index=True, column_config=dt_cfg)

# ──────────────────────────────── ③ 월간 리포트 ────────────────────────────────
with tab3:
    kpi_row([
        dict(label="월평균 가동률", value=f"{daily.가동률.mean():.1f}", unit="%"),
        dict(label="최장 연속가동", value="48.3", unit="시간", sub="5/16 09:54 – 5/18 10:12 무인 주말가공"),
        dict(label="이상 알람", value=f"{int(daily.이상알람.sum())}", unit="건"),
        dict(label="가감속 구간 사이클", value=f"{(df.state == 'transient').sum():,}",
             sub="부하 90% 이상 스파이크는 전부 이 구간에서 발생 (설비 이상 아님)"),
    ])

    with st.container(border=True):
        st.markdown("##### 일자별 가동률과 이상 알람")
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28],
                            vertical_spacing=0.14, subplot_titles=("가동률 (%)", "이상 알람 (건)"))
        fig.add_trace(go.Bar(x=daily.date, y=daily.가동률, marker_color=T["blue"],
                             marker=dict(cornerradius=4), width=0.62 * 86400000,
                             hovertemplate="%{y:.1f}%<extra>가동률</extra>"), 1, 1)
        al_d = daily[daily.이상알람 > 0]
        fig.add_trace(go.Bar(x=al_d.date, y=al_d.이상알람, marker_color=T["critical"],
                             marker=dict(cornerradius=2), width=0.62 * 86400000,
                             hovertemplate="%{y}건<extra>이상 알람</extra>"), 2, 1)
        fig.update_xaxes(tickformat="%m/%d")
        fig.update_yaxes(range=[0, 105], row=1)
        fig.update_yaxes(dtick=2, rangemode="tozero", row=2)
        st.plotly_chart(style(fig, 380, legend=False, top=40), width="stretch", config=NOBAR)

    c1, c2 = st.columns(2)
    with c1, st.container(border=True):
        st.markdown("##### 시간대별 평균 가동률 (%)")
        st.caption("낮에는 셋업, 야간에는 무인 가공이 진행되는 패턴이 나타납니다.")
        hourly = df.assign(h=df.ts.dt.hour).groupby("h").status.apply(lambda s: (s == "STATUS_RUN").mean() * 100)
        f1 = go.Figure(go.Scatter(x=hourly.index, y=hourly.values, fill="tozeroy",
                                  fillcolor=T["band"], line=dict(color=T["blue"], width=2),
                                  hovertemplate="%{x}시 · %{y:.0f}%<extra></extra>"))
        f1.update_xaxes(dtick=3, ticksuffix="시", range=[0, 23])
        f1.update_yaxes(rangemode="tozero")
        st.plotly_chart(style(f1, 250, legend=False, top=16), width="stretch", config=NOBAR)
    with c2, st.container(border=True):
        st.markdown("##### 공구 사용 분포")
        st.caption("실제 가공에 사용된 사이클 수 기준입니다.")
        tools = df[df.status == "STATUS_RUN"].tool.value_counts().sort_index()
        f2 = go.Figure(go.Bar(x=[f"T{t:02d}" for t in tools.index], y=tools.values,
                              marker_color=T["blue"], marker=dict(cornerradius=3),
                              hovertemplate="%{x} · %{y:,} 사이클<extra></extra>"))
        f2.update_yaxes(tickformat="~s")
        st.plotly_chart(style(f2, 250, legend=False, top=16), width="stretch", config=NOBAR)

    with st.container(border=True):
        st.markdown("""
##### 데이터 품질 진단과 개선 제안
- **축별 부하 기록 이슈** — X/Y/Z 축 부하가 전 기간 동일한 값으로 기록되어 있어, 축 단위 진단은 현재 데이터로는 불가능합니다. 로거 설정 수정만으로 개선할 수 있습니다.
- **알람 코드 미기록** — 알람 발생 시 "Alarm"이라는 문구만 남고 코드가 기록되지 않아 원인 분석이 어렵습니다. 이 역시 로거 설정으로 개선 가능합니다.
- **부하 90% 이상 스파이크 69건은 모두 스핀들 감속(제동) 순간**으로 확인되었습니다. 설비 이상이 아니며, 단순 임계값 감시를 쓰면 이것 때문에 하루 2회꼴 오탐이 발생합니다. 본 분석에서는 가감속 구간을 분리해 이 오탐을 제거했습니다.
- **매주 일요일 03시경 약 3.3분의 수집 공백**이 있습니다(로거 정기 재시작으로 추정). 그 외 불규칙 공백 3건.
- **다음 단계 제안** — 고장 시점을 예측하는 단계로 나아가려면 알람 코드, 정비 이력(CMMS), 스핀들 진동·전류 센서, NC 프로그램명 로깅이 축적되어야 합니다.
""")

# ──────────────────────────────── ④ 모델 검증 ────────────────────────────────
# 표·차트의 수치는 experiments/benchmark.py가 산출한 data/bench.csv를 그대로 읽는다.
INJ_LABEL = {"지속과부하": "지속 과부하", "공구파손": "공구 파손형", "완만한저하": "완만한 저하"}
SHORT = {"단순 임계값 감시 (SPC)": "단순 임계값", "머신러닝 일괄 탐지 (Isolation Forest)": "머신러닝 일괄",
         "운전조건별 비교 — 채택": "채택 방식", "운전조건별 Isolation Forest (보조)": "조건별 IF",
         "패턴 재구성 (PCA)": "패턴 재구성", "신경망 (오토인코더)": "신경망",
         "신경망 (LSTM 오토인코더)": "LSTM"}

def fmt_detect(rate, sec, n=None):
    """탐지율 + (최초 알람까지 평균 시간) 셀 문자열. 표본(n) 5개 미만이면 병기해 표본 부족을 알린다."""
    if rate <= 0 or pd.isna(sec):
        return f"{rate:.0f}%"
    t = f"{sec/60:.0f}분" if sec >= 90 else f"{sec:.0f}초"
    tag = f", n={int(n)} 표본 부족" if n is not None and n < 5 else ""
    return f"{rate:.0f}% ({t}{tag})"

with tab4:
    if not os.path.exists("data/bench.csv"):
        st.info("벤치마크 결과 파일(data/bench.csv)이 아직 없습니다. "
                "`python experiments/benchmark.py`를 실행하면 이 탭이 채워집니다.")
    else:
        bench = pd.read_csv("data/bench.csv")
        adopted = bench[bench.채택 == 1].iloc[0]
        test_from = pd.Timestamp(meta["train_range"].split("~")[1].strip()) + pd.Timedelta("1D")
        test_sd = df[(df.ts >= test_from) & (df.state == "steady")]
        novel_share = 100 * test_sd.novel.mean()
        st.markdown(f"""
##### 이 알람은 얼마나 믿을 수 있는가
실제 고장 기록이 없는 데이터 → 고장과 비슷한 가상 이상 신호를 심어 놓고 방식별로 얼마나
잡는지 시험함. {len(bench)}가지 방식 모두 같은 조건(학습 5/1–21, 시험 5/22–31, 기준값 산출 규칙 동일).

이상 유형 3가지 × 각 60회 — 지속 과부하(+12%p, 11분) · 공구 파손형(부하 소실, 3분) ·
완만한 저하(2.3시간에 +8%p).

주입은 학습 기간에 있던 운전조건 구간에만 함. 시험 구간의 {novel_share:.0f}%는 학습에 없던
새 조건이라 알람 대신 "새 운전조건 알림"(탭 ②)으로 빠지며, 탐지율 시험 대상이 아님.
실운영에서는 기준값을 주기적으로 재학습하면 이 비율이 줄어듦.
""")
        disp = pd.DataFrame({
            "분석 방식": bench.방식,
            "오탐 (건/일)": bench.오탐_건일,
            **{INJ_LABEL[k]: [fmt_detect(r, s, n) for r, s, n in
                              zip(bench[f"{k}_탐지율"], bench[f"{k}_평균탐지초"],
                                  bench[f"{k}_탐지n"] if f"{k}_탐지n" in bench.columns
                                  else [None] * len(bench))]
               for k in INJ_LABEL}})
        st.dataframe(disp, width="stretch", hide_index=True,
                     column_config={
                         "분석 방식": st.column_config.TextColumn("분석 방식", width="medium"),
                         "오탐 (건/일)": st.column_config.NumberColumn("오탐 (건/일)", format="%.2f", width="small"),
                     })
        st.caption("괄호 안은 이상 발생부터 알람 확정까지의 평균 시간. 채택 방식은 두 채널 중 먼저 확정된 쪽 "
                   "기준이라 30초 지속 조건보다 빠를 수 있음(큰 이탈은 점진적 변화 채널이 2~3사이클 만에 확정). "
                   "오탐(건/일)은 시험 구간 달력일수(약 10일) 기준 — 가동 여부와 무관하게 하루에 몇 번 울리는가. "
                   "'n=x 표본 부족' 표기는 60회 주입 중 성공 사례가 5건 미만이라 평균 시간이 통계적으로 "
                   "의미가 약하다는 뜻(예: n=1이면 그 시간은 평균이 아니라 유일한 성공 사례 값).")

        with st.container(border=True):
            st.markdown("##### 이상 유형별 탐지율 (%)")
            short = [SHORT.get(m, m) for m in bench.방식]
            fig = go.Figure()
            for k, col in zip(INJ_LABEL, [T["blue"], T["orange"], T["aqua"]]):
                fig.add_trace(go.Bar(x=short, y=bench[f"{k}_탐지율"], name=INJ_LABEL[k],
                                     marker_color=col, marker=dict(cornerradius=3),
                                     hovertemplate="%{y:.0f}%<extra>" + INJ_LABEL[k] + "</extra>"))
            i_adopt = int(bench.index[bench.채택 == 1][0])
            fig.add_vrect(x0=i_adopt - 0.5, x1=i_adopt + 0.5,
                          fillcolor="rgba(255,255,255,0.045)", line_width=0)
            fig.add_annotation(x=i_adopt, y=113, text="채택", showarrow=False,
                               font=dict(size=12, color=T["ink"], family=FONT))
            fig.update_layout(barmode="group", bargap=0.32, bargroupgap=0.12)
            fig.update_yaxes(range=[0, 120], tickvals=[0, 25, 50, 75, 100])
            st.plotly_chart(style(fig, 320), width="stretch", config=NOBAR)

        rate_cols = [f"{k}_탐지율" for k in INJ_LABEL]
        adopt_min = bench.loc[bench.채택 == 1, rate_cols].min(axis=1).iloc[0]
        rival_fp = bench.loc[(bench.채택 == 0) & (bench[rate_cols] >= 50).all(axis=1), "오탐_건일"]
        rival_txt = (f"세 유형을 다 잡은 방식이 하나 더 있으나 오탐이 하루 {rival_fp.min():.2f}건이라 "
                     "현장 적용은 어려움." if len(rival_fp) else
                     "다른 방식들은 세 유형 중 일부를 놓침.")
        st.markdown(f"""
##### 채택 방식을 선택한 이유
세 유형 전부를 {adopt_min:.0f}% 이상 잡으면서 오탐을 하루 {adopted.오탐_건일:.1f}건으로 막은 것은
채택한 운전조건별 비교 방식뿐. {rival_txt}

전역 방식들이 놓치는 건 "그 운전조건에서는 비정상이지만 값 자체는 흔한" 이상 —
저속 가공에서 부하 15%는 문제 신호일 수 있지만 고속 가공에서는 정상 수치인 식.

오탐을 줄인 장치는 셋: 가감속 순간 분리, 처음 보는 조건은 별도 알림, 순간 이탈은 30초 지속 시에만
알람(이 규칙만으로 순간 채널 오탐 하루 3.0건 → 0건). 남은 오탐 하루 {adopted.오탐_건일:.1f}건은 전부
'점진적 변화' 쪽이며, 주 단위 기준값 갱신으로 관리 가능한 수준.
""")
