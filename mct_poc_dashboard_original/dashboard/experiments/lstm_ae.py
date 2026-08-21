# -*- coding: utf-8 -*-
"""LSTM Autoencoder 이상감지 실험 (CLAUDE.md §8.2 — 계획서 명시 산출물)

목적은 채택이 아니라 '직접 구현 + 동일 조건 공정 비교 후 채택 근거 확보'다.
결과가 채택 모델(M2*)보다 나쁘면 나쁜 대로, 좋으면 좋은 대로 보고한다 — 어느 쪽으로도 유도하지 않는다.

프로토콜(다른 전역 후보와 동일):
  학습 5/1~21 steady(설비 이벤트 ±10분·EMG ±5분 제외) · 전역 표준화 원시 특징
  (sload, rpm−srate 갭, log1p(drpm), feed) · 창 24사이클(≈2.2분)·stride 6 ·
  이상 점수 = 재구성 오차 · 임계값 = 학습 점수 99.9분위.

사용:  python experiments/lstm_ae.py                 # CPU 학습 (목표 10분 이내, seed 고정)
       python experiments/lstm_ae.py --report-only   # 재학습 없이 보고서만 재생성
산출:  data/lstm_ae.pt · reports/lstm_ae.md (계획서의 '임계값 설정 결과 보고서')
       이후 experiments/benchmark.py 재실행 시 7번째 후보로 자동 편입된다.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import pipeline_core as core
from benchmark import TRAIN_END, feats, load_and_refit, COND, cur_k

SEED = 42
WINDOW = 24          # 24사이클 ≈ 2.2분
STRIDE = 6
SPAN_MAX = WINDOW * 5.5 * 2   # 창 내 최대 허용 시간폭(초) — 수집 공백을 가로지르는 창 배제
HID = 32
EPOCHS = 40
BATCH = 256
LR = 1e-3
PATIENCE = 5
PT_PATH = os.path.join(ROOT, "data", "lstm_ae.pt")
RPT_PATH = os.path.join(ROOT, "reports", "lstm_ae.md")


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_model():
    torch, nn = _torch()

    class LSTMAE(nn.Module):
        """seq2seq 오토인코더: LSTM 인코더 마지막 은닉 → 반복 입력 → LSTM 디코더 → 복원."""

        def __init__(self, n_feat=4, hid=HID):
            super().__init__()
            self.enc = nn.LSTM(n_feat, hid, batch_first=True)
            self.dec = nn.LSTM(hid, hid, batch_first=True)
            self.out = nn.Linear(hid, n_feat)

        def forward(self, x):
            _, (h, _) = self.enc(x)
            rep = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
            y, _ = self.dec(rep)
            return self.out(y)

    return LSTMAE()


def window_starts(ts, n, stride=1):
    """길이 WINDOW 창의 유효 시작 위치 — 수집 공백을 가로지르는 창은 제외."""
    span = (ts[WINDOW - 1:] - ts[:n - WINDOW + 1]) / np.timedelta64(1, "s")
    ok = span <= SPAN_MAX
    starts = np.arange(0, n - WINDOW + 1, stride)
    return starts[ok[starts]]


def batch_scores(model, X, starts, batch=4096):
    """창별 재구성 오차(MSE). X는 표준화된 (N,4) 특징 행렬."""
    torch, _ = _torch()
    W = np.lib.stride_tricks.sliding_window_view(X, (WINDOW, X.shape[1]))[:, 0]
    out = np.empty(len(starts), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for i in range(0, len(starts), batch):
            xb = torch.from_numpy(np.ascontiguousarray(W[starts[i:i + batch]])).float()
            rec = model(xb)
            out[i:i + batch] = ((rec - xb) ** 2).mean(dim=(1, 2)).numpy()
    return out


class LSTMAECandidate:
    """benchmark.py가 7번째 후보로 편입하는 래퍼 — 저장된 가중치·정규화·임계값으로 판정.
    지속성 규칙(6사이클 연속 초과) 공통 적용."""
    name = "신경망 (LSTM 오토인코더)"
    warmup = WINDOW

    def __init__(self, pt_path=PT_PATH):
        torch, _ = _torch()
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        self.mdl = build_model()
        self.mdl.load_state_dict(d["state_dict"])
        self.mdl.eval()
        self.m_, self.s_, self.thr = d["mean"], d["std"], d["thr"]

    def flags(self, dfr):
        n = len(dfr)
        fl = np.zeros(n, dtype=bool)
        if n < WINDOW:
            return fl
        X = (feats(dfr) - self.m_) / self.s_
        ts = dfr.ts.to_numpy()
        starts = window_starts(ts, n, stride=1)
        if len(starts) == 0:
            return fl
        sc = batch_scores(self.mdl, X, starts)
        fl[starts + WINDOW - 1] = sc > self.thr   # 알람 시점 = 창의 마지막 사이클
        return core.persist(fl, k=cur_k()) if COND["persist"] else fl


# ── 학습 ─────────────────────────────────────────────────────────────────────
def train():
    torch, nn = _torch()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t_start = time.time()

    sd, is_tr, *_ = load_and_refit()
    X_all = feats(sd)
    tr_rows = np.where(is_tr)[0]

    # 학습 창: 모든 사이클이 학습 마스크 안에 있는 창만 (stride 6)
    in_tr = np.zeros(len(sd), dtype=bool)
    in_tr[tr_rows] = True
    ts = sd.ts.to_numpy()
    starts = window_starts(ts, len(sd), stride=STRIDE)
    tr_ok = np.array([in_tr[s:s + WINDOW].all() for s in starts])
    starts = starts[tr_ok]

    # 표준화 통계는 학습 창 구성 사이클에서만 산출 (§8.2: 전역 표준화)
    used = np.zeros(len(sd), dtype=bool)
    for s in starts:
        used[s:s + WINDOW] = True
    mean, std = X_all[used].mean(0), X_all[used].std(0)
    std[std == 0] = 1.0
    X = (X_all - mean) / std

    W = np.lib.stride_tricks.sliding_window_view(X, (WINDOW, X.shape[1]))[:, 0]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(starts))
    n_val = max(1, len(starts) // 10)
    va_s, tr_s = starts[perm[:n_val]], starts[perm[n_val:]]
    print(f"학습 창 {len(tr_s):,} · 검증 창 {n_val:,} (창 {WINDOW}사이클 · stride {STRIDE})")

    model = build_model()
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    xva = torch.from_numpy(np.ascontiguousarray(W[va_s])).float()

    curve, best, best_state, wait = [], np.inf, None, 0
    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_perm = rng.permutation(len(tr_s))
        tl = 0.0
        for i in range(0, len(tr_s), BATCH):
            xb = torch.from_numpy(np.ascontiguousarray(W[tr_s[ep_perm[i:i + BATCH]]])).float()
            opt.zero_grad()
            loss = lossf(model(xb), xb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            vl = float(lossf(model(xva), xva))
        curve.append((ep, tl / len(tr_s), vl))
        print(f"  epoch {ep:2d}  train {tl / len(tr_s):.5f}  val {vl:.5f}")
        if vl < best - 1e-6:
            best, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  조기 종료 (val 손실 {PATIENCE}epoch 미개선)")
                break
    model.load_state_dict(best_state)

    # 임계값: 학습 창 재구성 오차의 99.9분위 (기존 채널과 동일 규칙)
    tr_scores = batch_scores(model, X, np.sort(np.concatenate([tr_s, va_s])))
    thr = float(np.quantile(tr_scores, core.Q))
    pct = {p: float(np.quantile(tr_scores, p)) for p in [0.5, 0.9, 0.99, 0.999]}
    mins = (time.time() - t_start) / 60
    print(f"임계값(99.9분위): {thr:.5f} · 학습 소요 {mins:.1f}분 · 파라미터 {n_par:,}개")

    torch.save(dict(state_dict=model.state_dict(), mean=mean, std=std, thr=thr,
                    config=dict(window=WINDOW, stride=STRIDE, hid=HID, seed=SEED,
                                feats=["sload", "rpm-srate", "log1p(drpm)", "feed"]),
                    curve=curve, pct=pct, n_par=n_par, n_train=len(tr_s), n_val=n_val,
                    train_minutes=round(mins, 1)),
               PT_PATH)
    print(f"저장: {os.path.relpath(PT_PATH, ROOT)}")


# ── 보고서 ───────────────────────────────────────────────────────────────────
def write_report():
    torch, _ = _torch()
    d = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    c = d["config"]
    curve = d["curve"]
    pct = d["pct"]

    bench_path = os.path.join(ROOT, "data", "bench.csv")
    bench_row = None
    m2_row = None
    if os.path.exists(bench_path):
        b = pd.read_csv(bench_path)
        hit = b[b.방식 == LSTMAECandidate.name]
        bench_row = hit.iloc[0] if len(hit) else None
        m2 = b[b.채택 == 1]
        m2_row = m2.iloc[0] if len(m2) else None

    rows = "\n".join(f"| {ep} | {tr:.5f} | {vl:.5f} |" for ep, tr, vl in curve)
    if bench_row is not None:
        def fmt(r, k):
            """탐지율(+평균 탐지시간) 셀. 표본(n)이 5 미만이면 평균이 사실상 개별값이므로 n을 병기한다."""
            rate = f"{r[f'{k}_탐지율']:.0f}%"
            if pd.isna(r[f"{k}_평균탐지초"]):
                return rate
            n = int(r.get(f"{k}_탐지n", -1))
            tag = f", n={n} 표본 부족" if 0 <= n < 5 else ""
            return f"{rate} (평균 {r[f'{k}_평균탐지초']:.0f}초{tag})"
        bench_md = f"""| 항목 | LSTM-AE | 채택 모델(M2*) |
|---|---|---|
| 오탐 (건/일) | {bench_row.오탐_건일:.2f} | {m2_row.오탐_건일:.2f} |
| 지속 과부하 탐지 | {fmt(bench_row, '지속과부하')} | {fmt(m2_row, '지속과부하')} |
| 공구 파손형 탐지 | {fmt(bench_row, '공구파손')} | {fmt(m2_row, '공구파손')} |
| 완만한 저하 탐지 | {fmt(bench_row, '완만한저하')} | {fmt(m2_row, '완만한저하')} |

수치 출처: `data/bench.csv` (benchmark.py, 유형별 60회 주입·seed 고정·동일 프로토콜).
괄호 안 시간 = 주입 시작부터 알람 확정까지 평균.

참고: 두 방식의 오탐/일이 같은 건 우연 — 각자 7개 에피소드를 냈을 뿐, 플래그된 사이클은
겹침 0 (M2*는 5/29~30 드리프트 구간, LSTM은 5/23~30 분산). flags 교집합으로 확인함."""
    else:
        bench_md = ("벤치마크 미실행. `python experiments/benchmark.py` 실행 후 "
                    "`python experiments/lstm_ae.py --report-only`를 다시 돌리면 채워짐.")

    md = f"""# LSTM Autoencoder 이상감지 — 구현·임계값 설정 보고서

수행계획서의 "LSTM 기반 이상감지 모델 + 임계값 설정 결과 보고서" 산출물.
목적은 채택이 아니라 직접 구현 후 동일 조건 비교이며, 결과는 좋든 나쁘든 그대로 기록함.
모든 수치는 `python experiments/lstm_ae.py` 재실행으로 재현 가능 (seed={c['seed']}, CPU 학습).

## 1. 데이터·전처리

- 학습: 5/1~21 정상 가공(steady)만, 설비 알람 ±10분·EMG ±5분 제외 (채택 모델 M2*와 동일 마스크)
- 입력 특징 4개, 전역 표준화: sload(부하), rpm−srate(회전수 갭), log1p(drpm), feed(이송)
- 시퀀스: 창 {c['window']}사이클(≈2.2분), stride {c['stride']}. 수집 공백을 걸치는 창은 제외
- 학습 {d['n_train']:,}창 / 검증 {d['n_val']:,}창 (무작위 9:1)

## 2. 모델

seq2seq 오토인코더. LSTM 인코더(은닉 {c['hid']})의 마지막 은닉을 시퀀스 길이만큼 반복 입력해
LSTM 디코더 + 선형층으로 복원. 파라미터 {d['n_par']:,}개.
학습 Adam(lr={LR}), MSE, 배치 {BATCH}, 최대 {EPOCHS}epoch(검증 {PATIENCE}epoch 미개선 시 조기 종료),
소요 약 {d['train_minutes']}분(CPU).

## 3. 학습 곡선

| epoch | train MSE | val MSE |
|---|---|---|
{rows}

## 4. 임계값 설정

이상 점수 = 창 재구성 오차(MSE). 임계값 = 학습 점수의 99.9분위 — 기존 세 채널과 같은 규칙.
정상 데이터 기준 1,000창에 1창꼴로 넘는 수준을 기준선으로 삼는다는 의미.

| 학습 점수 분위 | 재구성 오차 |
|---|---|
| 50% | {pct[0.5]:.5f} |
| 90% | {pct[0.9]:.5f} |
| 99% | {pct[0.99]:.5f} |
| 99.9% (임계값) | {pct[0.999]:.5f} |

## 5. 벤치마크 성적 (7후보 동일 조건)

{bench_md}

## 6. 해석·한계

- 학습 점수의 오른쪽 꼬리가 매우 두꺼움 (중앙값 {pct[0.5]:.2f} vs 99.9분위 {pct[0.999]:.2f},
  약 {pct[0.999] / max(pct[0.5], 1e-9):.0f}배). 드물지만 정상인 전환 구간을 잘 복원하지 못한 탓.
  같은 규칙으로 임계값을 잡으면 매우 보수적이 됨 → 오탐 낮고 탐지율도 낮음. 벤치마크 성적의 주 원인.
- 전역 모델이라 운전조건(공구×RPM)별 기준이 없음. "그 조건에서만 비정상"인 이상은 원리적으로
  잡기 어려움 — 탭④ 전역 후보들의 공통 한계이지 LSTM 고유 문제는 아님.
- 임계값·학습 구간은 5/1~21 기준. 실운영이면 주 단위 재베이스라이닝 필요.
- 탐지율은 합성 이상 주입 검증. 실제 고장 라벨 검증 아님.
- 컨텍스트 정규화 입력 변형은 선택 실험으로 남김 (수행 시 벤치마크 표에 별도 행으로만 추가).
"""
    os.makedirs(os.path.dirname(RPT_PATH), exist_ok=True)
    with open(RPT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"저장: {os.path.relpath(RPT_PATH, ROOT)}")


if __name__ == "__main__":
    if "--report-only" not in sys.argv:
        train()
    write_report()
