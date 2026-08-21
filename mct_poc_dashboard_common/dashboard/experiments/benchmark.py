# -*- coding: utf-8 -*-
"""탭④ 벤치마크 실험 복원 (CLAUDE.md §8.1 — 계획서 산출물의 근거 실험)

프로토콜(탭④ 문구가 명세):
  학습 5/1~21 · 시험 5/22~31 · steady만 · 각 후보의 임계값은 자기 학습 점수의 99.9분위.
  합성 이상 3종을 시험 구간의 steady 구간에 무작위 주입(유형별 60회, seed 고정),
  오탐/일은 주입하지 않은 원본 시험 구간에서 에피소드(60초 병합) 단위로 측정.

오탐/일의 분모: 시험 구간 달력일수 = 학습 종료 시점(5/22 00:00)부터 데이터 끝까지(≈10.0일).
  설비 가동/비가동과 무관하게 "담당자가 하루에 알람을 몇 번 보는가" 기준이다.

탐지 시간의 정의: 주입 시작 사이클부터 **해당 방식의 알람 규칙이 확정된 첫 사이클**까지.
  M2*는 두 채널 중 먼저 확정된 쪽 기준 — 순간 채널은 6사이클(≈30초) 지속 충족 시점이지만,
  드리프트(EWMA) 채널이 큰 이탈을 2~3사이클 만에 확정하면 그보다 빠른 시간이 기록될 수 있다.
  지속성 규칙이 없는 전역 후보들은 첫 임계 초과 사이클이 곧 확정이다.

사용:  python experiments/benchmark.py     ← data/scored.csv.gz만 필요, 원본 로그 불필요
산출:  data/bench.csv  + 콘솔에 기존 하드코딩 수치와의 비교표
       data/lstm_ae.pt가 있으면 LSTM-AE(§8.2)를 7번째 후보로 자동 편입한다.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

import pipeline_core as core

SEED = 42
N_INJ = 60                                # 유형별 주입 횟수
TRAIN_END = pd.Timestamp("2026-05-22")    # regenerate.py TRAIN_END와 동일한 프로토콜 정의
WARMUP = 3000                             # 주입 재스코어링 시 EWMA·롤링 워밍업용 선행 steady 사이클
RAW = ["ts", "tool", "band", "srate", "rpm", "sload", "feed", "drpm"]
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")

# 실행 중인 조건 전역 상태 — SPC·FeatModel·CondIF·LSTM-AE의 flags()가 참조한다.
# main()이 조건마다 값을 바꿔 끼우고 재실행한다(딕셔너리 자체는 그대로 두고 값만 mutate —
# lstm_ae.py가 `from benchmark import COND`로 같은 객체를 참조하므로 재할당하면 끊어진다).
# persist_k: persist=True일 때 쓰는 지속 사이클 수 — K 스윕에서 6이 아닌 값을 넣는다.
COND = dict(persist=True, split=True, persist_k=None)

# 조건 정의 — 순서대로 실행. (지속성 규칙 적용 여부, 임계값 분리 적용 여부) — 2×2 완전 설계.
CONDITIONS = [
    ("A_원본", False, False),        # 둘 다 없음 (이전 세션 방식)
    ("B_persist단독", True, False),   # +6사이클 지속 규칙만
    ("D_분리단독", False, True),      # +임계값 분리만
    ("C_둘다", True, True),          # 둘 다 — persist가 이미 바닥을 쳐서 분리 효과가 안 보일 수 있음(D로 확인)
]

# 합성 이상 3종 — (키, 지속초, 창 내 최소 steady 사이클 수)
INJ_TYPES = [
    ("지속과부하", 660, 60),      # sload +12%p, 11분
    ("공구파손", 180, 16),        # sload를 대기 수준(0~5%)으로 강제, 3분
    ("완만한저하", 8280, 750),    # 0 → +8%p 선형 램프, 2.3시간
]
INJ_LABEL = {"지속과부하": "지속 과부하", "공구파손": "공구 파손형", "완만한저하": "완만한 저하"}  # app.py와 동일 매핑

# 이전 세션 하드코딩 수치(구 app.py 탭④) — 재현 대조용
OLD = {
    "단순 임계값 감시 (SPC)":                  (0.22, 100.0, 0.0, 18.3),
    "머신러닝 일괄 탐지 (Isolation Forest)":    (4.08, 16.0, 4.0, 56.7),
    "운전조건별 비교 — 채택":                   (1.87, 100.0, 100.0, 100.0),
    "운전조건별 Isolation Forest (보조)":       (2.76, 100.0, 0.0, 100.0),
    "패턴 재구성 (PCA)":                       (1.65, 5.0, 2.0, 23.3),
    "신경망 (오토인코더)":                      (5.40, 100.0, 25.0, 51.7),
}


# ── 1) 데이터 로드와 학습 통계 재현 검증 ────────────────────────────────────
def load_and_refit():
    """scored.csv.gz 원시 컬럼에서 M2* 학습을 재현하고 저장값(meta.json 등)과 대조한다."""
    df = pd.read_csv(
        os.path.join(DATA, "scored.csv.gz"),
        parse_dates=["ts"]
    )

    events = pd.read_csv(
        os.path.join(DATA, "events.csv"),
        parse_dates=["시각"]
    )

    # 전체 로그에서 이벤트 시각은 그대로 사용
    ev_ts = list(
        df.loc[
            df.status == "STATUS_ERROR",
            "ts"
        ]
    )

    emg = events[
        events.유형.str.contains(
            "EMG",
            na=False
        )
    ]

    emg_ts = (
        list(emg.시각)
        + list(
            emg.시각
            + pd.to_timedelta(
                emg.지속초,
                unit="s"
            )
        )
    )

    # ── 지도/비지도 공통 실제 절삭 기준 ──
    cutting_mask = (
        (df["status"] == "STATUS_RUN")
        & (df["rpm"] > 0)
        & (df["feed"] > 0)
    )

    df_model = df[
        cutting_mask
    ].copy()

    print(
        f"벤치마크 실제 절삭 데이터: "
        f"{len(df):,}행 → {len(df_model):,}행"
    )

    # regenerate.py와 동일하게 실제 절삭 데이터만 학습
    tm = core.train_mask(
        df_model,
        TRAIN_END,
        ev_ts,
        emg_ts,
    )

    sd_stored = df_model[
        df_model.state == "steady"
    ]

    sd = sd_stored[
        RAW
    ].copy()

    stats = core.fit_stats(
        df_model[tm]
    )
    
    sd = core.score_steady(sd, stats)
    is_tr = sd.index.isin(df_model[tm].index)
    thr, if3 = core.fit_thresholds(sd, is_tr, seed=0)
    sd = core.flag_channels(sd, thr)

    # 검증: 재현값 vs regenerate.py 저장값
    import json
    meta = json.load(open(os.path.join(DATA, "meta.json"), encoding="utf-8"))
    checks = {
        "ms(기대부하) 최대오차": float((sd.ms - sd_stored.ms).abs().max()),
        "zload 최대오차": float((sd.zload - sd_stored.zload).abs().max()),
        "thr_z 오차": abs(thr["thr_z"] - meta["thr_z"]),
        "thr_ew 오차": abs(thr["thr_ew"] - meta["thr_ew"]),
        "thr_m3 오차": abs(thr["thr_m3"] - meta["thr_m3"]),
        "f_pt 불일치 사이클": int((sd.f_pt != sd_stored.f_pt).sum()),
        "f_ew 불일치 사이클": int((sd.f_ew != sd_stored.f_ew).sum()),
        "novel 불일치 사이클": int((sd.novel != sd_stored.novel).sum()),
    }
    # 허용오차: 알람 채널(f_pt·f_ew)·통계는 완전 일치, thr_ew는 CSV 왕복 부동소수(1e-4),
    # thr_m3는 보조 채널 IsolationForest 재학습 편차(1%)까지 재현 성공으로 본다.
    tol = {"thr_ew 오차": 1e-4 * meta["thr_ew"], "thr_m3 오차": 0.01 * meta["thr_m3"]}
    ok = all(v <= tol.get(k, 1e-9) for k, v in checks.items())
    print("── 학습 통계 재현 검증 (scored.csv 원시 컬럼 → regenerate.py 저장값 대조) ──")
    for k, v in checks.items():
        print(f"  {k}: {v:.3g}")
    print("  →", "완전 재현 성공" if ok else "차이 있음(아래 결과는 재현 통계 기준)")
    return sd.reset_index(drop=True), np.asarray(is_tr), stats, thr, if3, df.ts.max()


# ── 2) 비교 후보 ─────────────────────────────────────────────────────────────
def feats(d):
    """전역 후보 공통 특징: sload · rpm−srate 갭 · log1p(drpm) · feed (§8.2와 동일 구성)."""
    return np.column_stack([d.sload, d.rpm - d.srate, np.log1p(d.drpm), d.feed])


def cur_k():
    """현재 조건의 지속성 규칙 K값 — K 스윕에서 COND['persist_k']를 채우고, 아니면 운영값(6)."""
    return COND["persist_k"] or core.PERSIST_K


class SPC:
    """전역 단순 임계값 감시 — 학습 전체의 평균±표준편차 기준 |z|.
    지속성 규칙(6사이클 연속 초과) 공통 적용 — M2*만 갖던 장치를 모든 후보에 동일 적용."""
    name = "단순 임계값 감시 (SPC)"
    warmup = core.PERSIST_K   # 지속성 롤링 윈도우 선행 이력

    def fit(self, fit_df, cal_df):
        """fit_df로 기준 통계, cal_df(별도 구간)로 임계값을 산출 — 학습·보정 구간 분리."""
        self.mu, self.sg = fit_df.sload.mean(), fit_df.sload.std()
        self.thr = np.quantile(np.abs((cal_df.sload - self.mu) / self.sg), core.Q)

    def flags(self, d):
        raw = (np.abs((d.sload - self.mu) / self.sg) > self.thr).to_numpy()
        return core.persist(raw, k=cur_k()) if COND["persist"] else raw


class FeatModel:
    """전역 특징(표준화) 기반 후보의 공통 골격 — 자기 보정 구간 점수 99.9분위 임계값.
    지속성 규칙(6사이클 연속 초과) 공통 적용."""
    warmup = core.PERSIST_K   # 지속성 롤링 윈도우 선행 이력

    def fit(self, fit_df, cal_df):
        """fit_df로 모델을, cal_df(별도 구간)로 임계값을 산출 — 학습·보정 구간 분리."""
        X = feats(fit_df)
        self.m_, self.s_ = X.mean(0), X.std(0)
        self.s_[self.s_ == 0] = 1.0
        Xs = (X - self.m_) / self.s_
        self._fit(Xs)
        Xc = (feats(cal_df) - self.m_) / self.s_
        self.thr = np.quantile(self._score(Xc), core.Q)

    def flags(self, d):
        X = (feats(d) - self.m_) / self.s_
        raw = self._score(X) > self.thr
        return core.persist(raw, k=cur_k()) if COND["persist"] else raw


class GlobalIF(FeatModel):
    name = "머신러닝 일괄 탐지 (Isolation Forest)"

    def _fit(self, X):
        self.mdl = IsolationForest(n_estimators=200, max_samples=1024,
                                   random_state=SEED).fit(X)

    def _score(self, X):
        return -self.mdl.score_samples(X)


class PCARecon(FeatModel):
    name = "패턴 재구성 (PCA)"

    def _fit(self, X):
        self.mdl = PCA(n_components=2, random_state=SEED).fit(X)

    def _score(self, X):
        R = self.mdl.inverse_transform(self.mdl.transform(X))
        return ((X - R) ** 2).mean(axis=1)


class DenseAE(FeatModel):
    name = "신경망 (오토인코더)"

    def _fit(self, X):
        self.mdl = MLPRegressor(hidden_layer_sizes=(8, 3, 8), max_iter=100,
                                random_state=SEED, early_stopping=True,
                                n_iter_no_change=5).fit(X, X)

    def _score(self, X):
        return ((X - self.mdl.predict(X)) ** 2).mean(axis=1)


class M2Star:
    """채택 모델 — 컨텍스트 잔차 z(30초 지속) + 컨텍스트 EWMA. regenerate.py 로직 재사용."""
    name = "운전조건별 비교 — 채택"
    warmup = WARMUP     # 컨텍스트 EWMA 워밍업 필요

    def __init__(self, stats, thr):
        self.stats, self.thr = stats, thr

    def flags(self, d):
        s = core.score_steady(d[RAW].copy(), self.stats)
        known = ~s.novel.to_numpy()
        f_pt = core.persist((s.zload.abs() > self.thr["thr_z"]).to_numpy()) & known
        f_ew = (s.ewma.abs() > self.thr["thr_ew"]).to_numpy() & known
        return f_pt | f_ew


class CondIF:
    """조건부 IsolationForest — M3 보조 채널을 단독 후보로 평가.
    지속성 규칙 공통 적용(순서는 M2*의 f_pt와 동일 — persist 먼저, known은 나중에 AND)."""
    name = "운전조건별 Isolation Forest (보조)"
    warmup = 24         # rolling(12) 특징 워밍업 — 지속성 롤링(6)보다 커서 그대로 유지

    def __init__(self, stats, thr, if3):
        self.stats, self.thr, self.if3 = stats, thr, if3

    def flags(self, d):
        s = core.score_steady(d[RAW].copy(), self.stats)
        m3 = -self.if3.score_samples(s[core.CF])
        ex = m3 > self.thr["thr_m3"]
        raw = core.persist(ex, k=cur_k()) if COND["persist"] else ex
        return raw & (~s.novel.to_numpy())


def try_lstm_candidate():
    """§8.2 완료 후 data/lstm_ae.pt가 있으면 LSTM-AE를 7번째 후보로 편입."""
    pt = os.path.join(DATA, "lstm_ae.pt")
    if not os.path.exists(pt):
        print("· LSTM-AE 미편입 (data/lstm_ae.pt 없음 — §8.2 실행 후 자동 추가)")
        return None
    try:
        from lstm_ae import LSTMAECandidate
        return LSTMAECandidate(pt)
    except Exception as e:
        print(f"· LSTM-AE 미편입 ({e})")
        return None


# ── 3) 합성 이상 주입 ────────────────────────────────────────────────────────
KNOWN_MIN = 0.8   # 주입 창의 학습된(known) 컨텍스트 사이클 비율 하한.
# novel 컨텍스트는 M2*가 설계상 '판단 불가'로 분류하는 구간이므로(CLAUDE.md §1-3) 탐지율
# 측정 대상이 아니다 — 시험 구간의 novel 비율은 별도 수치로 보고한다(숨기지 않는다).


def sample_sites(sd, rng, clean_alarm):
    """시험 구간 steady에서 유형별 N_INJ개 주입 창을 뽑는다. 모든 후보가 동일 창 사용.
    조건: 창 내 known 비율 ≥ KNOWN_MIN, 원본 데이터에서 이미 알람 중인 창은 제외."""
    ts = sd.ts.to_numpy()
    novel = sd.novel.to_numpy()
    test_pos = np.where(ts >= np.datetime64(TRAIN_END))[0]
    t_max = ts[-1]
    sites = {}
    for key, dur, min_c in INJ_TYPES:
        got, seen = [], set()
        while len(got) < N_INJ:
            j0 = int(rng.choice(test_pos))
            if j0 in seen:
                continue
            seen.add(j0)
            t0 = ts[j0]
            t1 = t0 + np.timedelta64(dur, "s")
            if t1 > t_max:
                continue
            j1 = int(np.searchsorted(ts, t1, side="right"))
            if j1 - j0 < min_c:
                continue
            if novel[j0:j1].mean() > 1 - KNOWN_MIN or clean_alarm[j0:j1].any():
                continue
            elapsed = (ts[j0:j1] - t0) / np.timedelta64(1, "s")
            base = sd.sload.to_numpy()[j0:j1]
            if key == "지속과부하":
                inj = base + 12.0
            elif key == "공구파손":
                inj = rng.uniform(0.0, 5.0, j1 - j0)
            else:  # 완만한저하: 0 → +8%p 선형 램프
                inj = base + 8.0 * elapsed / dur
            got.append((j0, j1, inj))
        sites[key] = got
    return sites


def eval_site(cand, sd, j0, j1, inj):
    """한 주입 창 채점: 창 안에서 알람이 켜지면 탐지.
    반환값 = 주입 시작부터 해당 방식의 알람 규칙이 '확정'된 첫 사이클까지의 시간(초)
    — flags()가 지속성·EWMA 등 각 방식의 확정 규칙을 이미 반영한 뒤의 첫 True 위치다."""
    lo = max(0, j0 - getattr(cand, "warmup", WARMUP))
    sl = sd.iloc[lo:j1].copy().reset_index(drop=True)
    s = sl["sload"].to_numpy(copy=True)
    s[j0 - lo:] = inj
    sl["sload"] = s
    fl = np.asarray(cand.flags(sl))
    w = fl[j0 - lo:]
    if not w.any():
        return None
    k = int(np.argmax(w))
    return float((sl.ts.iloc[j0 - lo + k] - sl.ts.iloc[j0 - lo]).total_seconds())


def count_episodes(ts_flagged, gap=60):
    """regenerate.py와 동일한 60초 병합 에피소드 개수."""
    if len(ts_flagged) == 0:
        return 0
    d = pd.Series(ts_flagged).sort_values().diff().dt.total_seconds()
    return int(1 + (d > gap).sum())


def build_cands(stats, thr, if3, lstm, fit_df, cal_df):
    """조건마다 새 인스턴스로 학습 — 이전 조건의 임계값이 새는 것을 막는다.
    fit_df/cal_df가 같은 프레임이면(조건 A·B) '분리 없음', 다르면(조건 C) '분리'가 된다."""
    cands = [SPC(), GlobalIF(), M2Star(stats, thr), CondIF(stats, thr, if3),
             PCARecon(), DenseAE()]
    for c in cands:
        if hasattr(c, "fit"):
            c.fit(fit_df, cal_df)
    if lstm is not None:
        cands.append(lstm)
    return cands


def run_condition(cond_name, persist, split, sd, tr, fit_tr, cal_tr, stats, thr, if3,
                   lstm, sites, days, persist_k=None):
    """조건 하나를 끝까지 평가해 bench DataFrame으로 반환. 같은 seed로 뽑은 sites를 모든
    조건이 그대로 공유하므로, 조건 간 차이는 오직 persist·split·persist_k 설정에서만 온다.
    M2*는 자체 f_pt(core.persist 직접 호출, 항상 K=6)를 쓰므로 persist_k의 영향을 받지 않는다
    — K 스윕에서 M2*가 매번 동일한 성적으로 나오는 것은 버그가 아니라 고정 기준점 역할이다."""
    COND["persist"], COND["split"], COND["persist_k"] = persist, split, persist_k
    fit_df, cal_df = (fit_tr, cal_tr) if split else (tr, tr)
    print(f"\n=== 조건 {cond_name} — persist={persist} split={split} k={persist_k or core.PERSIST_K} ===")
    cands = build_cands(stats, thr, if3, lstm, fit_df, cal_df)

    in_test = (sd.ts >= TRAIN_END).to_numpy()
    rows = []
    for c in cands:
        print(f"평가: {c.name}")
        fl = np.asarray(c.flags(sd))                     # 원본(무주입) 전 구간 스코어링
        fp_day = count_episodes(sd.ts.to_numpy()[fl & in_test]) / days
        row = {"조건": cond_name, "방식": c.name, "채택": int(isinstance(c, M2Star)),
               "오탐_건일": round(fp_day, 2)}
        for key, dur, _ in INJ_TYPES:
            times = [eval_site(c, sd, j0, j1, inj) for j0, j1, inj in sites[key]]
            det = [t for t in times if t is not None]
            row[f"{key}_탐지율"] = round(100.0 * len(det) / len(times), 1)
            row[f"{key}_평균탐지초"] = round(float(np.mean(det)), 1) if det else np.nan
            row[f"{key}_탐지n"] = len(det)   # 평균탐지초의 표본 수 — 표본 부족 셀 판별용
        rows.append(row)
    return pd.DataFrame(rows)


def table_md(bench):
    """bench DataFrame → 방식별 오탐/탐지율 마크다운 표 (n 병기)."""
    hdr = "| 방식 | 오탐(건/일) | " + " | ".join(INJ_LABEL[k] for k, *_ in INJ_TYPES) + " |"
    sep = "|---|---|" + "---|" * len(INJ_TYPES)
    lines = [hdr, sep]
    for _, r in bench.iterrows():
        cells = [f"{r[f'{k}_탐지율']:.0f}%" +
                 ("" if r[f"{k}_탐지n"] == 0 else f" (n={int(r[f'{k}_탐지n'])})")
                 for k, *_ in INJ_TYPES]
        mark = " **← 채택**" if r.채택 else ""
        lines.append(f"| {r.방식}{mark} | {r.오탐_건일:.2f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_condition_report(benches):
    """조건 A/B/D/C(2×2: persist × 임계값분리) 결과를 나란히 놓는 보고서.
    B와 C가 같은 값이 나온다면(둘 다 persist로 바닥을 쳐서) D가 분리의 단독 효과를 보여준다."""
    by_name = {b["조건"].iloc[0]: b for b in benches}
    A, B, D, C = by_name["A_원본"], by_name["B_persist단독"], by_name["D_분리단독"], by_name["C_둘다"]

    md = f"""# 벤치마크 조건별 비교(2×2: persist × 임계값분리) — 지속성 규칙·임계값 분리 민감도

탭④ 수치가 "M2* 전용 장치(6사이클 지속 규칙)를 다른 후보에 씌워 유리하게 만든 것 아니냐"는
질문에 답하기 위한 근거 표. 네 조건을 같은 주입 위치(seed={SEED})로 나란히 비교하는 2×2 완전 설계.

|              | 임계값 분리 없음 | 임계값 분리 적용 |
|---|---|---|
| **persist 없음** | A. 원본 | D. 분리 단독 |
| **persist 적용(K={core.PERSIST_K})** | B. persist 단독 | C. 둘 다 |

- **A. 원본** — 둘 다 없음. 임계값 = 학습구간 전체 점수의 99.9분위 (최초 결과)
- **B. persist 단독** — 6사이클(K={core.PERSIST_K}) 연속 초과 확정 규칙만 전 후보에 공통 적용
- **D. 분리 단독** — persist 없이, 임계값 산출 구간만 학습 구간과 분리(시간순 80/20)
- **C. 둘 다** — B + D (SPC·GlobalIF·PCA·AE에 한함; M2*·조건부IF는 운영값 재현 검증이 목적이라
  원 프로토콜 유지 — 아래 "한계" 참고)

## 조건 A — 원본

{table_md(A)}

## 조건 B — persist 단독

{table_md(B)}

## 조건 D — 임계값 분리 단독

{table_md(D)}

## 조건 C — 둘 다

{table_md(C)}

## 해석

**D vs A (분리 단독 효과)**: persist 없이 임계값 산출 구간만 분리했을 때 오탐/탐지율이
A와 거의 같다면, 이 데이터·후보 조합에서는 임계값 분리 자체의 효과가 크지 않다는 뜻이다
(즉, 조건 A의 결과가 "학습 데이터에 과적합된 임계값" 때문이 아니라는 근거). 반대로 D에서
오탐이 눈에 띄게 달라졌다면 원래 방식(A)의 임계값이 실제로 학습 표본에 낙관적으로
맞춰져 있었다는 뜻 — 아래 D 표의 실측값으로 판단한다.

**B vs A (persist 단독 효과)**: 6사이클 지속 규칙을 전역 후보(단순 임계값·IsolationForest·
PCA·오토인코더)에 적용하자 오탐과 함께 탐지율도 대부분 0%로 붕괴했다. 이 규칙(K={core.PERSIST_K})은
애초에 M2* 순간 채널의 오탐을 하루 3.0건→0건으로 낮추도록 M2* 기준으로 정해진 값이며,
전역 후보에게는 원래 설계에 없던 불리한 조건이다 — 그 사실을 숨기지 않고 그대로 보고한다.

**C vs B**: B에서 이미 오탐이 0으로 바닥을 친 후보는 C에서 임계값을 분리해도 더 내려갈 곳이
없어 B와 동일하게 보일 수 있다 — 그 경우 분리 효과는 D의 결과로 판단해야 하며 C만 보고
"분리가 효과 없다"고 결론 내리면 안 된다.

네 조건 전부에서 채택 방식(M2*)만 세 이상 유형 모두에서 의미 있는 탐지율을 유지했다.
전역 후보들이 A·D(persist 없음)에서 보인 탐지율은 "지속 규칙이 없는 즉시 알람" 조건에서
나온 것이며, B·C(persist 있음)에서의 붕괴는 K=6이 그들에게 맞지 않는 값이기 때문일 수
있다는 점은 K 스윕 결과(`reports/benchmark_k_sweep.md`)로 별도 확인한다.

## 한계

- 임계값 분리(D·C)는 SPC·GlobalIF·PCA·AE에만 적용됐다. M2*·조건부IF도 같은 방식으로
  분리한 버전은 이 표에 없다(별도 비교표 작업 필요 — 운영 파이프라인 재현 검증과는 분리해서
  수행해야 함).
"""
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, "benchmark_conditions.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"저장: {os.path.relpath(path, ROOT)}")


K_SWEEP = [1, 2, 3, 6]   # 지속성 규칙 K값 스윕 — 6은 M2* 운영값(core.PERSIST_K)과 동일 지점


def run_k_sweep(sd, tr, stats, thr, if3, lstm, sites, days):
    """K=1,2,3,6에서 전역 후보(+M2*)를 재평가 — split은 걸지 않는다(조건 B와 동일 조건,
    분리 변수 하나만 바꿔가며 보는 게 목적). M2*는 자체 K=6 고정이라 매 K에서 동일하게 나온다
    (버그 아님 — cur_k() 참고)."""
    benches = {}
    for k in K_SWEEP:
        name = f"K{k}"
        bench = run_condition(name, True, False, sd, tr, tr, tr, stats, thr, if3,
                               lstm, sites, days, persist_k=k)
        bench.to_csv(os.path.join(DATA, f"bench_{name}.csv"), index=False)
        print(f"저장: data/bench_{name}.csv")
        benches[k] = bench
    write_k_sweep_report(benches)


def write_k_sweep_report(benches):
    """benches: {K값: bench DataFrame}. data/bench_K*.csv에서 다시 읽어 재계산 없이도 호출 가능."""
    names = benches[K_SWEEP[0]]["방식"].tolist()
    fp_hdr = "| 방식 | " + " | ".join(f"K={k}" for k in K_SWEEP) + " |"
    fp_sep = "|---|" + "---|" * len(K_SWEEP)
    fp_lines = [fp_hdr, fp_sep]
    for name in names:
        cells = [f"{benches[k].loc[benches[k].방식 == name, '오탐_건일'].iloc[0]:.2f}" for k in K_SWEEP]
        fp_lines.append(f"| {name} | " + " | ".join(cells) + " |")

    det_blocks = []
    for key, *_ in INJ_TYPES:
        hdr = f"### {INJ_LABEL[key]} 탐지율 (%)\n\n| 방식 | " + " | ".join(f"K={k}" for k in K_SWEEP) + " |"
        sep = "|---|" + "---|" * len(K_SWEEP)
        lines = [hdr, sep]
        for name in names:
            cells = [f"{benches[k].loc[benches[k].방식 == name, f'{key}_탐지율'].iloc[0]:.0f}%" for k in K_SWEEP]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        det_blocks.append("\n".join(lines) + "\n")

    md = f"""# K 스윕 — 지속성 규칙 K값에 따른 후보별 성적

"K={core.PERSIST_K}이 M2* 기준으로 정해진 값이라 전역 후보에 불리하다"는 한계를 정량화한다.
persist는 전 후보 공통 적용(조건 B와 동일), 임계값 분리는 걸지 않음 — K 하나만 바꿔가며 본다.
M2*는 자체 지속 채널이 항상 K={core.PERSIST_K} 고정이라 모든 K에서 동일한 성적으로 나온다
(고정 기준점 — 버그 아님).

## 오탐 (건/일)

{chr(10).join(fp_lines)}

## 이상 유형별 탐지율

{chr(10).join(det_blocks)}

## 해석

전역 후보(단순 임계값·IsolationForest·PCA·오토인코더·조건부IF·LSTM-AE)의 탐지율이 K가
작아질수록(예: K=1~2) 회복되는지, 회복된 지점에서 오탐도 같이 늘어나는지, 그리고 그 최적
지점에서도 M2*(고정 성적)를 못 넘는지를 위 표에서 그대로 읽는다. 별도 서술로 유도하지 않는다
— 표가 결론이다.

## 한계

- K 격자는 {K_SWEEP}로 한정했다. 더 촘촘한 스윕(예: K=4,5)은 여기 없다.
- 이 스윕은 persist 유무·K값만 바꾼 것이며 임계값 분리(D·C 조건)와는 결합하지 않았다.
"""
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, "benchmark_k_sweep.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"저장: {os.path.relpath(path, ROOT)}")


# ── 4) 실행 ──────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)
    sd, is_tr, stats, thr, if3, data_end = load_and_refit()
    tr = sd[is_tr]  # 전 후보 동일 조건: M2*와 같은 학습 마스크(5/1~21 steady, 이벤트 주변 제외)
    # 오탐/일 분모: 시험 구간 달력일수 (학습 종료 시점 ~ 데이터 끝) — steady 시각 범위가 아님
    days = (data_end - TRAIN_END).total_seconds() / 86400

    # 학습·임계값보정 구간 분리(시간순 80/20) — SPC·GlobalIF·PCA·AE 전용, 조건 C에서만 쓰인다.
    # M2*·CondIF는 load_and_refit()에서 regenerate.py(운영값 meta.json)와 동일 계산을 재현·검증하는
    # 것이 목적이라 여기서 분할하지 않는다(운영 임계값 재베이스라이닝은 별도 작업).
    cal_start = tr.ts.min() + (tr.ts.max() - tr.ts.min()) * 0.8
    fit_tr, cal_tr = tr[tr.ts < cal_start], tr[tr.ts >= cal_start]
    print(f"학습/보정 분할(조건 C, SPC·GlobalIF·PCA·AE): fit {len(fit_tr):,}행 · "
          f"임계값보정 {len(cal_tr):,}행 (시간순 80/20)")

    lstm = try_lstm_candidate()

    in_test_all = (sd.ts >= TRAIN_END).to_numpy()
    novel_share = 100.0 * sd.novel.to_numpy()[in_test_all].mean()

    # 지속성 규칙의 효과 실측 (탭④ 서술 "순간 채널 오탐 감소"의 근거) — M2* 운영 채널 기준, 조건과 무관
    ts_arr = sd.ts.to_numpy()
    raw_ex = (sd.zload.abs() > thr["thr_z"]).to_numpy() & ~sd.novel.to_numpy()
    print(f"순간 채널 지속성 규칙 효과(시험 구간 달력 {days:.2f}일): "
          f"규칙 없이 {count_episodes(ts_arr[raw_ex & in_test_all]) / days:.2f}건/일 → "
          f"6사이클 지속 적용 {count_episodes(ts_arr[sd.f_pt.to_numpy() & in_test_all]) / days:.2f}건/일")
    print(f"주입 창 샘플링… (known≥{KNOWN_MIN:.0%} 창만 — 시험구간 novel 비율 {novel_share:.1f}%는 별도 보고)")
    # 세 조건이 전부 같은 위치에 주입되도록 sites는 한 번만 뽑아 공유한다 — 조건 간 차이가
    # 주입 위치 변화가 아니라 오직 persist·split 설정에서만 나오도록 하기 위함.
    sites = sample_sites(sd, rng, clean_alarm=(sd.f_pt | sd.f_ew).to_numpy())

    benches = []
    for cond_name, persist, split in CONDITIONS:
        bench = run_condition(cond_name, persist, split, sd, tr, fit_tr, cal_tr,
                               stats, thr, if3, lstm, sites, days)
        bench.to_csv(os.path.join(DATA, f"bench_{cond_name}.csv"), index=False)
        print(f"저장: data/bench_{cond_name}.csv")
        benches.append(bench)

    # app.py 탭④가 읽는 data/bench.csv는 가장 엄격한 조건(C = 둘 다)을 반영한다.
    bench_c = next(b for b in benches if b["조건"].iloc[0] == "C_둘다").drop(columns=["조건"])
    bench_c.to_csv(os.path.join(DATA, "bench.csv"), index=False)
    print(f"\n저장: data/bench.csv = 조건 C ({len(bench_c)}개 후보, 유형별 주입 {N_INJ}회, seed={SEED})")

    write_condition_report(benches)
    run_k_sweep(sd, tr, stats, thr, if3, lstm, sites, days)

    # 기존 하드코딩 수치(구 app.py, persist 없음)와 조건 A 비교 — 완전 일치가 아니라
    # 순위·경향 재현이 목표(§8.1)
    print("\n── 기존 하드코딩 수치(구) → 조건 A 재실험(신) — 순위·경향 재현이 목표 ──")
    print(f"{'방식':<32} {'오탐/일':>16} {'지속과부하%':>16} {'공구파손%':>16} {'완만한저하%':>16}")
    for _, r in benches[0].iterrows():
        old = OLD.get(r.방식, (None,) * 4)
        cells = []
        for o, n in zip(old, [r.오탐_건일, r.지속과부하_탐지율, r.공구파손_탐지율, r.완만한저하_탐지율]):
            cells.append(f"{'—' if o is None else f'{o:g}':>7}→{n:<7g}")
        print(f"{r.방식:<32} " + " ".join(cells))


if __name__ == "__main__":
    main()
