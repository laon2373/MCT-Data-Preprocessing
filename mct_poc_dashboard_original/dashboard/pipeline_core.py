# -*- coding: utf-8 -*-
"""M2* 파이프라인 공용 코어 — regenerate.py(배치)와 api.py(실시간 API), experiments/(실험)가 임포트한다.

컨텍스트(공구×RPM밴드) 통계 학습 · 조회 · 폴백 · novel 판정 · 채널 스코어링의 단일 소스.
regenerate.py의 §2~§5 수식을 그대로 옮긴 것이며, 양쪽 로직이 어긋나면 안 된다(CLAUDE.md §8.4).
app.py는 이 모듈을 임포트하지 않는다(대시보드 단독 구동 원칙).
"""
import numpy as np
import pandas as pd

CF = ["zload", "zfeed", "drpm_l", "rstd", "rmax_z"]  # M3(보조 다변량 채널) 특징
PERSIST_K = 6      # 순간 채널 지속 판정: 6사이클(≈30초) 연속 초과 시 알람
GAP_RPM = 200      # steady 판정: |rpm-srate| ≤ 200 그리고 drpm ≤ 200
Q = 0.999          # 임계값 분위수 — 학습구간 점수의 99.9분위


# ── 상태 분류 (regenerate §2) ────────────────────────────────────────────────
def classify(df):
    """band·state 컬럼 부여. df에는 status·srate·rpm·drpm 컬럼이 있어야 한다."""
    df["band"] = (df.srate / 1000).round().astype(int)
    gap = (df.rpm - df.srate).abs()
    df["state"] = "ready"
    run = df.status == "STATUS_RUN"
    df.loc[run, "state"] = np.where((gap[run] <= GAP_RPM) & (df.drpm[run] <= GAP_RPM),
                                    "steady", "transient")
    df.loc[df.status == "STATUS_ERROR", "state"] = "error"
    return df


# ── 학습 구간 마스크 (regenerate §3) ─────────────────────────────────────────
def train_mask(df, train_end, ev_ts, emg_ts):
    """train_end 이전의 steady만, 설비 이벤트(ERROR) ±10분 · EMG ±5분 제외."""
    m = (df.ts < train_end) & (df.state == "steady")
    for t in ev_ts:
        m &= (df.ts - t).abs() > pd.Timedelta("10min")
    for t in emg_ts:
        m &= (df.ts - t).abs() > pd.Timedelta("5min")
    return m


# ── 컨텍스트 통계 학습·조회·폴백 (regenerate §4) ────────────────────────────
def robust(s):
    med = s.median()
    mad = (s - med).abs().median()
    return float(med), float(max(1.4826 * mad, 0.5))


def fit_stats(tr, min_cycles=50):
    """컨텍스트(공구×밴드)별 robust 통계. 희소 조건은 공구 → 전역 통계로 폴백."""
    return dict(
        ctx={c: (robust(g.sload), robust(g.feed))
             for c, g in tr.groupby([tr.tool, tr.band]) if len(g) >= min_cycles},
        tool={t: (robust(g.sload), robust(g.feed)) for t, g in tr.groupby("tool")},
        glob=(robust(tr.sload), robust(tr.feed)),
    )


def lookup(stats, tool, band):
    """컨텍스트 → 공구 → 전역 폴백. ((부하 med, 부하 scale), (피드 med, 피드 scale))."""
    return stats["ctx"].get((tool, band)) or stats["tool"].get(tool) or stats["glob"]


def is_novel(stats, tool, band):
    """학습된 컨텍스트가 없으면 novel — 이상이 아니라 '판단 불가' (CLAUDE.md §1-3)."""
    return (tool, band) not in stats["ctx"]


def score_steady(sd, stats):
    """steady 프레임에 novel·ms·ss·zload·zfeed·drpm_l·rstd·rmax_z·ewma 컬럼 부여."""
    pairs = [lookup(stats, t, b) for t, b in zip(sd.tool, sd.band)]
    sd["novel"] = [is_novel(stats, t, b) for t, b in zip(sd.tool, sd.band)]
    sd["ms"] = [p[0][0] for p in pairs]
    sd["ss"] = [p[0][1] for p in pairs]
    sd["zload"] = (sd.sload - sd.ms) / sd.ss
    sd["zfeed"] = (sd.feed - np.array([p[1][0] for p in pairs])) / \
                  np.array([p[1][1] for p in pairs])
    sd["drpm_l"] = np.log1p(sd.drpm)
    sd["rstd"] = sd.sload.rolling(12, min_periods=3).std().fillna(0)
    sd["rmax_z"] = sd.zload.rolling(12, min_periods=3).max().fillna(0)
    sd["ewma"] = sd.groupby([sd.tool, sd.band], sort=False)["zload"] \
                   .transform(lambda s: s.ewm(span=60).mean())
    return sd


# ── 임계값·M3 학습과 채널 판정 (regenerate §5) ──────────────────────────────
def fit_thresholds(sd, is_tr, seed=0):
    """학습 점수 99.9분위 임계값 3종 + M3(조건부 IsolationForest) 학습."""
    from sklearn.ensemble import IsolationForest
    thr_z = float(np.quantile(sd.loc[is_tr, "zload"].abs(), Q))
    thr_ew = float(np.quantile(sd.loc[is_tr, "ewma"].abs(), Q))
    if3 = IsolationForest(n_estimators=200, max_samples=1024,
                          random_state=seed).fit(sd.loc[is_tr, CF])
    sd["m3"] = -if3.score_samples(sd[CF])
    thr_m3 = float(np.quantile(sd.loc[is_tr, "m3"], Q))
    return dict(thr_z=thr_z, thr_ew=thr_ew, thr_m3=thr_m3), if3


def persist(flags, k=PERSIST_K):
    """k사이클 연속 초과 시에만 True (순간 채널 지속성 규칙 — 오탐 3.31→0.00/일)."""
    s = pd.Series(np.asarray(flags).astype(int))
    return (s.rolling(k).sum() == k).fillna(False).values


def flag_channels(sd, thr):
    """채널 3개 판정. novel 컨텍스트는 어떤 채널에서도 알람으로 승격하지 않는다."""
    known = ~sd.novel.values
    sd["f_pt"] = persist((sd.zload.abs() > thr["thr_z"]).values) & known
    sd["f_ew"] = (sd.ewma.abs() > thr["thr_ew"]) & known
    sd["f_m3"] = (sd.m3 > thr["thr_m3"]) & known
    return sd


# ── 학습 산출물 저장/로드 (api.py 서빙용 — CLAUDE.md §8.4) ──────────────────
def save_artifacts(path, stats, thr, if3, meta=None):
    import joblib
    joblib.dump(dict(stats=stats, thr=thr, if3=if3, meta=meta or {}), path)


def load_artifacts(path):
    import joblib
    return joblib.load(path)
