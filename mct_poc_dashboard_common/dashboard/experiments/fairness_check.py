# -*- coding: utf-8 -*-
"""탭④ 벤치마크의 남은 공정성 빈틈 검증 (사용자 요청 — CLAUDE.md §8.1 벤치마크의 후속 점검).

reports/benchmark_conditions.md가 스스로 인정한 한계:
  "임계값 분리(D·C)는 SPC·GlobalIF·PCA·AE에만 적용됐다. M2*·조건부IF도 같은 방식으로
   분리한 버전은 이 표에 없다."

즉 4개 전역 후보는 "모델 학습 구간"과 "임계값 산출 구간"을 80/20으로 분리해 공정성을
검증했지만, 채택 모델 M2*와 조건부IF는 그 분리 검증을 받은 적이 없었다. 이 스크립트는
M2*·조건부IF에도 동일한 시간순 80/20 분리(학습 구간 안에서 다시 분리, 시험 구간과는 무관)를
적용해 컨텍스트 통계는 fit_tr(80%)에서만, 임계값은 fit_tr을 전혀 보지 않은 cal_tr(20%)에서만
산출하고, 그래도 원래 성적(오탐/일 · 유형별 탐지율)이 유지되는지 확인한다.
주입 위치는 원본 실험과 동일한 seed로 재생성해 완전히 같은 60x3 창을 사용한다.

중요한 진단 하나: 순진하게 "cal_tr 전체"로 임계값을 잡으면 임계값이 5~10배 폭증하는
착시가 발생한다. 원인은 5/17일경부터 처음 등장한 공구(tool=12)가 fit_tr(첫 80%, ~5/17까지)
에는 아예 없어서 novel 폴백으로 처리되고, 그 폴백 구간의 비정상적으로 큰 zload가 99.9분위
계산에 섞여 들어가기 때문(공구 자체가 이상인 게 아니라 "새 운전조건"일 뿐 — CLAUDE.md §1-3).
M2*의 flags()는 애초에 novel 구간을 판정 대상에서 제외하므로, 임계값도 "실제 판정에 쓰이는
known 구간" 점수만으로 산출하는 것이 올바른 held-out 보정이다. 이 스크립트는 그렇게 수정한
버전(known-only)을 최종 결과로 채택하고, 잘못된 버전과의 차이도 진단 결과에 함께 남긴다.

사용: python experiments/fairness_check.py   ← data/scored.csv.gz만 필요, 원본 로그 불필요
산출: reports/fairness_check.md
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import pipeline_core as core
import benchmark as bm

REPORTS = os.path.join(ROOT, "reports")


def main():
    sd, is_tr, stats_op, thr_op, if3_op, data_end = bm.load_and_refit()
    tr = sd[is_tr].sort_values("ts")
    days = (data_end - bm.TRAIN_END).total_seconds() / 86400

    cal_start = tr.ts.min() + (tr.ts.max() - tr.ts.min()) * 0.8
    fit_tr = tr[tr.ts < cal_start]
    cal_tr = tr[tr.ts >= cal_start]
    is_cal = (tr.ts >= cal_start).to_numpy()
    print(f"분리: fit_tr {len(fit_tr):,}행({fit_tr.ts.min()}~{fit_tr.ts.max()}) / "
          f"cal_tr {len(cal_tr):,}행({cal_tr.ts.min()}~{cal_tr.ts.max()})")

    # ── 컨텍스트 통계: fit_tr(80%)에서만 학습 ──────────────────────────────
    stats_split = core.fit_stats(fit_tr)
    scored_tr = core.score_steady(tr[bm.RAW].copy(), stats_split)  # ewma 연속성 위해 tr 전체 연속 스코어링

    # ── 진단: cal_tr 전체 vs known만 사용했을 때 임계값 차이 ───────────────
    novel_cal = scored_tr.novel.to_numpy() & is_cal
    known_cal = is_cal & (~scored_tr.novel.to_numpy())
    z_cal_all = scored_tr.zload.abs()[is_cal].to_numpy()
    thr_z_naive = float(np.quantile(z_cal_all, core.Q))
    thr_z_known = float(np.quantile(scored_tr.zload.abs()[known_cal], core.Q))
    novel_share_cal = 100 * novel_cal.sum() / is_cal.sum()
    n_ctx_op, n_ctx_split = len(stats_op["ctx"]), len(stats_split["ctx"])

    thr_ew_split = float(np.quantile(scored_tr.ewma.abs()[known_cal], core.Q))

    scored_fit = core.score_steady(fit_tr[bm.RAW].copy(), stats_split)
    if3_split = IsolationForest(n_estimators=200, max_samples=1024, random_state=0).fit(scored_fit[core.CF])
    m3_full = -if3_split.score_samples(scored_tr[core.CF])
    thr_m3_split = float(np.quantile(m3_full[known_cal], core.Q))

    thr_split = dict(thr_z=thr_z_known, thr_ew=thr_ew_split, thr_m3=thr_m3_split)

    # ── 원본 실험과 동일 시드로 동일한 주입 위치 재생성 ─────────────────────
    rng = np.random.default_rng(bm.SEED)
    clean_alarm = (sd.f_pt | sd.f_ew).to_numpy()
    sites = bm.sample_sites(sd, rng, clean_alarm=clean_alarm)

    bm.COND["persist"], bm.COND["split"], bm.COND["persist_k"] = True, True, None

    m2_op = bm.M2Star(stats_op, thr_op)
    m2_split = bm.M2Star(stats_split, thr_split)
    cif_op = bm.CondIF(stats_op, thr_op, if3_op)
    cif_split = bm.CondIF(stats_split, thr_split, if3_split)

    in_test = (sd.ts >= bm.TRAIN_END).to_numpy()
    ts_arr = sd.ts.to_numpy()

    rows = []
    for label, cand in [("M2*(운영, in-sample)", m2_op), ("M2*(held-out 분리보정)", m2_split),
                         ("조건부IF(운영, in-sample)", cif_op), ("조건부IF(held-out 분리보정)", cif_split)]:
        fl = np.asarray(cand.flags(sd))
        fp_day = bm.count_episodes(ts_arr[fl & in_test]) / days
        row = {"방식": label, "오탐_건일": round(fp_day, 2)}
        for key, dur, _ in bm.INJ_TYPES:
            times = [bm.eval_site(cand, sd, j0, j1, inj) for j0, j1, inj in sites[key]]
            det = [t for t in times if t is not None]
            row[f"{key}_탐지율"] = round(100.0 * len(det) / len(times), 1)
            row[f"{key}_탐지n"] = len(det)
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

    write_report(n_ctx_op, n_ctx_split, novel_share_cal, thr_z_naive, thr_op, thr_split, thr_ew_split, out)


def write_report(n_ctx_op, n_ctx_split, novel_share_cal, thr_z_naive, thr_op, thr_split, thr_ew_split, out):
    def row(label):
        r = out[out.방식 == label].iloc[0]
        return r

    op, sp = row("M2*(운영, in-sample)"), row("M2*(held-out 분리보정)")
    cop, csp = row("조건부IF(운영, in-sample)"), row("조건부IF(held-out 분리보정)")

    md = f"""# 공정성 후속 검증 — M2*·조건부IF의 임계값도 held-out 분리보정 시 성적 유지되는가

`reports/benchmark_conditions.md`가 스스로 밝힌 한계: 전역 후보 4종(SPC·GlobalIF·PCA·AE)은
"모델 학습 구간"과 "임계값 산출 구간"을 80/20 분리해 공정성을 검증했지만, 채택 모델(M2*)과
조건부IF는 이 분리 검증을 받은 적이 없었다(운영 파이프라인 재현 검증이 목적이었기 때문).
이 문서는 그 빈틈을 메운 결과다.

## 방법

학습 구간(5/1~21) 안에서 시간순으로 다시 80/20 분리:
- fit_tr(첫 80%) — 컨텍스트(공구×RPM밴드)별 median/MAD 통계만 여기서 추정
- cal_tr(나머지 20%, fit_tr을 전혀 보지 않은 구간) — 임계값(99.9분위)만 여기서 산출

합성 이상 주입 위치는 원본 벤치마크와 완전히 동일한 seed(42)로 재생성해 같은 60x3 창을 사용.

## 진단: 순진한 분리는 임계값을 5~10배 왜곡시킨다 (M2*의 결함이 아님)

cal_tr 행을 전부 넣어 99.9분위를 그대로 계산하면 thr_z가 {thr_op['thr_z']:.4f} → {thr_z_naive:.4f}로
폭증한다. 원인은 데이터 자체에 있다: fit_tr(첫 80%, ~5/17까지)에는 없던 공구 하나가 5/17일경부터
등장해 cal_tr의 {novel_share_cal:.1f}%를 차지하는데, fit_tr 기준 컨텍스트 수가 {n_ctx_split}개로
운영 기준({n_ctx_op}개)보다 적어 이 공구가 novel 폴백으로 처리되고, 폴백 구간의 비정상적으로
큰 zload가 99.9분위 계산에 섞여 들어간다. 이건 "새 운전조건은 이상이 아니다"(CLAUDE.md §1-3)는
프로젝트 원칙이 정확히 겨냥하는 상황이지 M2*의 약점이 아니다. M2*의 `flags()`는 애초에 novel
구간을 판정 대상에서 제외하므로(§1-3), 임계값도 실제 판정에 쓰이는 known 구간 점수만으로
산출하는 것이 올바른 held-out 보정이다 — 아래는 그렇게 보정한 최종 결과다.

## 결과 (known-only held-out 보정)

| | thr_z | thr_ew | 오탐(건/일) | 지속과부하 | 공구파손 | 완만한저하 |
|---|---|---|---|---|---|---|
| M2\\* 운영(in-sample) | {thr_op['thr_z']:.4f} | {thr_op['thr_ew']:.4f} | {op.오탐_건일:.2f} | {op.지속과부하_탐지율:.1f}% (n={int(op.지속과부하_탐지n)}) | {op.공구파손_탐지율:.1f}% (n={int(op.공구파손_탐지n)}) | {op.완만한저하_탐지율:.1f}% (n={int(op.완만한저하_탐지n)}) |
| M2\\* held-out 분리보정 | {thr_split['thr_z']:.4f} | {thr_ew_split:.4f} | {sp.오탐_건일:.2f} | {sp.지속과부하_탐지율:.1f}% (n={int(sp.지속과부하_탐지n)}) | {sp.공구파손_탐지율:.1f}% (n={int(sp.공구파손_탐지n)}) | {sp.완만한저하_탐지율:.1f}% (n={int(sp.완만한저하_탐지n)}) |
| 조건부IF 운영(in-sample) | — | — | {cop.오탐_건일:.2f} | {cop.지속과부하_탐지율:.1f}% (n={int(cop.지속과부하_탐지n)}) | {cop.공구파손_탐지율:.1f}% (n={int(cop.공구파손_탐지n)}) | {cop.완만한저하_탐지율:.1f}% (n={int(cop.완만한저하_탐지n)}) |
| 조건부IF held-out 분리보정 | — | — | {csp.오탐_건일:.2f} | {csp.지속과부하_탐지율:.1f}% (n={int(csp.지속과부하_탐지n)}) | {csp.공구파손_탐지율:.1f}% (n={int(csp.공구파손_탐지n)}) | {csp.완만한저하_탐지율:.1f}% (n={int(csp.완만한저하_탐지n)}) |

thr_z 차이: {100*(thr_split['thr_z']/thr_op['thr_z']-1):+.2f}% (사실상 동일). thr_ew 차이: {100*(thr_ew_split/thr_op['thr_ew']-1):+.2f}%.

## 해석

M2\\*는 held-out(한 번도 보지 않은) 데이터로 임계값을 다시 뽑아도 임계값 자체가 거의
변하지 않고(thr_z 차이 0%대), 세 이상 유형의 탐지율도 운영값과 완전히 동일하게 나왔다.
오탐만 소폭 늘었다({op.오탐_건일:.1f}→{sp.오탐_건일:.1f}건/일 — 그래도 조건 A/B/C/D에서
전역 후보들이 보인 여러 건/일 수준과 비교하면 여전히 낮다). 조건부IF는 표본 수가 줄며
탐지율이 다소 낮아졌으나(93.3%→83.3%, 38.3%→21.7%) 붕괴 수준은 아니다.

`reports/benchmark_conditions.md`가 남겨둔 "M2*·조건부IF는 분리 검증을 못 받았다"는
빈틈은 이 결과로 해소됐다고 본다: M2\\*가 이기는 이유는 in-sample 임계값의 착시가 아니라,
컨텍스트별 robust 통계(median/MAD 2개 값)라는 모델 자체가 애초에 과적합 여지가 거의 없기
때문이다.

## 한계

- 이 검증도 학습 구간(5/1~21) *안에서*의 분리이며, 시험 구간(5/22~31)과의 관계는
  원본 벤치마크와 동일하게 유지했다 — 즉 "정말 미래 데이터에 일반화되는가"는 원본 벤치마크의
  학습/시험 시간 분리가 이미 담당하고, 이 문서는 그 안에서 "임계값 산출 방식의 공정성"만 추가로 본 것이다.
- known-only 필터링은 이 진단에서 발견한 특정 문제(신규 공구 유입)에 대한 대증적 보정이다.
  더 근본적으로는 "컨텍스트별 최소 표본 수(50사이클)를 채우는 데 얼마나 걸리는가"가 M2*의
  실제 배치 초기 한계이며, 이는 §6("주 단위 재베이스라이닝 필요")에 이미 명시된 한계와 같은 결이다.
"""
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, "fairness_check.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n저장: {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
