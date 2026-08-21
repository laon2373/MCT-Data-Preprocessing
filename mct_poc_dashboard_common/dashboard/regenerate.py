# -*- coding: utf-8 -*-
"""원본 로그 폴더에서 대시보드 데이터(data/)를 처음부터 재생성한다.
사용:  python regenerate.py <로그폴더>              예) python regenerate.py ./logs
       python regenerate.py <로그폴더> --save-model  # 학습 산출물을 data/model_artifacts.joblib로
                                                    # 추가 저장 (api.py 서빙용 — 기본 출력물은 불변)
파이프라인: 파싱 → 상태분류(정상/가감속) → 컨텍스트 정상통계 학습(앞 3주)
          → M2*(조건부 잔차+지속성+EWMA) & M3(조건부 IsolationForest) 스코어링
          → 알람/신규조건/설비이벤트/일별요약 저장
상태분류·통계학습·스코어링 수식은 pipeline_core.py가 단일 소스다(api.py와 공유 — CLAUDE.md §8.4).
"""
import re, sys, glob, json, os
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline_core as core

args = [a for a in sys.argv[1:] if not a.startswith("--")]
SAVE_MODEL = "--save-model" in sys.argv
CLEAN_CSV = args[0] if args else "./mct_clean.csv"
TRAIN_END = pd.Timestamp("2026-05-22")   # 이 이전을 정상통계 학습 구간으로 사용
os.makedirs("data", exist_ok=True)

# ── 1) 공통 전처리 데이터 로드 ────────────────────────────────────────────────
# 지도학습 프로젝트에서 생성한 mct_clean.csv를 그대로 사용한다.
raw = pd.read_csv(CLEAN_CSV, low_memory=False)

# 혹시 대문자/소문자가 달라도 맞춰지도록 통일
raw.columns = [
    str(c).strip().lower()
    for c in raw.columns
]

required = [
    "timestamp",
    "status",
    "current_tool",
    "feed_rate",
    "spindle_rate",
    "spindle_load",
    "spindle_rpm",
    "delta_spindle_rpm",
]

missing = [
    c for c in required
    if c not in raw.columns
]

assert not missing, (
    "mct_clean.csv에 필요한 컬럼이 없습니다: "
    + ", ".join(missing)
)

# M2*가 기존에 사용하던 컬럼 이름으로 연결
df = raw.rename(
    columns={
        "timestamp": "ts",
        "current_tool": "tool",
        "feed_rate": "feed",
        "spindle_rate": "srate",
        "spindle_load": "sload",
        "spindle_rpm": "rpm",
        "delta_spindle_rpm": "drpm",
    }
).copy()

df["ts"] = pd.to_datetime(
    df["ts"],
    errors="coerce",
)

for col in [
    "tool",
    "feed",
    "srate",
    "sload",
    "rpm",
    "drpm",
]:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )

# 기존 M2* 코드에서 사용하는 이벤트 컬럼 연결
if "is_emergency" in raw.columns:
    df["emg"] = pd.to_numeric(
        raw["is_emergency"],
        errors="coerce",
    ).fillna(0).astype(int)
else:
    df["emg"] = 0

if "is_alarm" in raw.columns:
    df["alarm"] = pd.to_numeric(
        raw["is_alarm"],
        errors="coerce",
    ).fillna(0).astype(int)
else:
    df["alarm"] = 0

df = (
    df
    .dropna(
        subset=[
            "ts",
            "tool",
            "feed",
            "srate",
            "sload",
            "rpm",
            "drpm",
            "status",
        ]
    )
    .sort_values("ts")
    .reset_index(drop=True)
)

df["tool"] = df["tool"].astype(int)

# 전체 로그는 설비 이벤트/가동률 계산용으로 보존
df_all = df.copy()

print(
    f"공통 전처리 데이터 로드 완료: "
    f"{len(df):,}행 "
    f"({df.ts.min()} ~ {df.ts.max()})"
)

# ── 2) 공통 실제 절삭 기준 + 상태 분류 ─────────────────────────────────────

# 전체 데이터는 기존 대시보드 화면을 위해 그대로 유지
df = core.classify(df_all.copy())

# 지도학습과 비지도학습이 동일한 실제 절삭 데이터에서 출발하도록 통일
cutting_mask = (
    (df["status"] == "STATUS_RUN")
    & (df["rpm"] > 0)
    & (df["feed"] > 0)
)

# M2* 학습/이상탐지에만 사용할 데이터
# 원본 index를 유지해야 나중에 결과를 전체 데이터에 다시 붙일 수 있음
df_model = df[cutting_mask].copy()

print(
    f"실제 절삭 데이터 필터링: "
    f"{len(df):,}행 → {len(df_model):,}행"
)

# ── 3) 학습 구간 정의: 앞 3주, 설비 이벤트 주변 제외 ────────────────────────
# 전체 로그에서 설비 이벤트 시각은 그대로 가져옴
ev_ts = list(df.loc[df.status == "STATUS_ERROR", "ts"])
emg_ts = list(df.loc[df.emg != 0, "ts"])

# M2* 학습 데이터는 실제 절삭 데이터만 사용
train_mask = core.train_mask(
    df_model,
    TRAIN_END,
    ev_ts,
    emg_ts,
)

tr = df_model[train_mask]

# ── 4) 컨텍스트(공구×RPM밴드)별 정상 통계 — 희소 조건은 공구→전역 폴백 ──────
stats = core.fit_stats(tr)

# M2* 이상탐지도 실제 절삭 데이터에서만 수행
sd = df_model[
    df_model.state == "steady"
].copy()

sd = core.score_steady(
    sd,
    stats,
)

# ── 5) 임계값(학습 점수 99.9분위) + M3 보조모델 ────────────────────────────
is_tr = sd.index.isin(tr.index)
thr, if3 = core.fit_thresholds(sd, is_tr, seed=0)
sd = core.flag_channels(sd, thr)   # f_pt(순간·30초 지속) / f_ew(드리프트) / f_m3(보조)
thr_z, thr_ew, thr_m3 = thr["thr_z"], thr["thr_ew"], thr["thr_m3"]

if SAVE_MODEL:
    core.save_artifacts("data/model_artifacts.joblib", stats, thr, if3,
                        meta=dict(device="MCT 3호",
                                  train_range=f"{df.ts.min():%Y-%m-%d} ~ {TRAIN_END - pd.Timedelta('1D'):%Y-%m-%d}",
                                  data_end=f"{df.ts.max():%Y-%m-%d %H:%M:%S}"))
    print("모델 산출물 저장: data/model_artifacts.joblib (api.py 서빙용)")

for c in ["zload","ewma","m3","ms","ss"]: df[c] = np.nan
for c in ["novel","f_pt","f_ew","f_m3"]:  df[c] = False
df.loc[sd.index, ["zload","ewma","m3","ms","ss","novel","f_pt","f_ew","f_m3"]] = \
    sd[["zload","ewma","m3","ms","ss","novel","f_pt","f_ew","f_m3"]].values

# ── 6) 에피소드화(알람/신규조건/설비이벤트) 및 저장 ─────────────────────────
def episodes(sub, gap=60):
    out, cur_, last = [], [], None
    for _, r in sub.iterrows():
        if last is not None and (r.ts - last).total_seconds() > gap:
            out.append(pd.DataFrame(cur_)); cur_ = []
        cur_.append(r); last = r.ts
    if cur_: out.append(pd.DataFrame(cur_))
    return out
al = []
for e in episodes(df[df.f_pt | df.f_ew]):
    ch = "+".join([c for c, f in [("순간", e.f_pt.any()), ("드리프트", e.f_ew.any())] if f])
    t0, b0 = e.tool.mode().iat[0], e.band.mode().iat[0]
    exp, z = e.ms.median(), e.zload.abs().max()
    dur = (e.ts.iloc[-1] - e.ts.iloc[0]).total_seconds() + 5.5
    dirn = "높음" if e.zload.mean() > 0 else "낮음"
    al.append(dict(시작=e.ts.iloc[0], 종료=e.ts.iloc[-1], 지속초=round(dur), 채널=ch,
                   공구=f"T{t0:02d}", RPM대역=f"{b0}k", 기대부하=round(exp, 1),
                   실측=f"{e.sload.min():.0f}~{e.sload.max():.0f}%", 최대z=round(z, 1),
                   설명=f"T{t0:02d}·{b0}k rpm: 기대 {exp:.0f}% 대비 부하 {dirn} 상태 {dur:.0f}초 지속 (최대|z|={z:.1f})"))
nv = [dict(시작=e.ts.iloc[0], 종료=e.ts.iloc[-1], 사이클수=len(e),
           조건=", ".join(sorted({f"T{t:02d}·{b}k" for t, b in zip(e.tool, e.band)})[:4]))
      for e in episodes(df[df.novel], gap=300)]
evd = df[(df.status == "STATUS_ERROR") | (df.emg != 0)]
evs = [dict(시각=e.ts.iloc[0], 유형="설비 알람(ERROR)" if (e.status == "STATUS_ERROR").any() else "비상신호(EMG)",
            지속초=round((e.ts.iloc[-1] - e.ts.iloc[0]).total_seconds() + 5.5))
       for e in episodes(evd, gap=120)]
alarms, novel, events = pd.DataFrame(al), pd.DataFrame(nv), pd.DataFrame(evs)

df["date"] = df.ts.dt.date
daily = df.groupby("date").agg(사이클=("ts", "size"),
                               가동률=("status", lambda s: (s == "STATUS_RUN").mean()*100)).reset_index()
for name, tab in [("이상알람", alarms), ("신규조건", novel)]:
    cnt = tab.set_index(tab["시작"].dt.date).groupby(level=0).size() if len(tab) else pd.Series(dtype=int)
    daily[name] = daily["date"].map(cnt).fillna(0).astype(int)

df[["ts","status","state","tool","band","srate","rpm","sload","feed","drpm",
    "novel","zload","ewma","m3","ms","ss","f_pt","f_ew","f_m3"]] \
    .to_csv("data/scored.csv.gz", index=False, compression="gzip")
alarms.to_csv("data/alarms.csv", index=False); novel.to_csv("data/novel.csv", index=False)
events.to_csv("data/events.csv", index=False); daily.to_csv("data/daily.csv", index=False)
json.dump(dict(thr_z=thr_z, thr_ew=thr_ew, thr_m3=thr_m3, device="MCT 3호",
               train_range=f"{df.ts.min():%Y-%m-%d} ~ {TRAIN_END - pd.Timedelta('1D'):%Y-%m-%d}",
               note="임계값은 학습구간 점수 99.9분위"),
          open("data/meta.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"완료: 알람 {len(alarms)}건 · 신규조건 {len(novel)}건 · 설비이벤트 {len(events)}건 → data/")
