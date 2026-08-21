# -*- coding: utf-8 -*-
"""기계 상태 4분류 ML 모델 (CLAUDE.md §8.3 — 계획서의 "정상·오류·알람·정지 4분류 모델")

실데이터의 대응 라벨은 state(steady/transient/ready/error)다.
입력은 센서 값만: sload, rpm, srate, drpm, feed. status·state 및 그 직접 파생 피처는 쓰지 않는다.
분할은 시간 기준(5/1~21 학습 / 5/22~31 평가) — 무작위 셔플 분할 금지(시계열 누수).

클래스 가중치는 적용/미적용 두 변형을 모두 실험해 그대로 보고한다:
balanced 가중치는 error(학습 표본 6개)에 극단 가중치를 주는데, error의 센서값 프로파일이
ready(대기)와 사실상 동일해 대기 전체가 error로 쏠리는 병리가 생긴다. 이것은 튜닝으로 감출
결함이 아니라 "error는 센서가 아니라 컨트롤러 플래그로만 구분된다"는 데이터의 발견 사항이다.

정직성 규칙(보고서에 명시):
  1) 라벨이 규칙 기반 산출이므로 높은 정확도는 "센서값만으로 상태 규칙을 복원했다"는 의미다.
  2) error 클래스 표본은 설비이벤트 유래로 극소다 — 그대로 보고하고 오버샘플링하지 않는다.

사용:  python experiments/classifier.py
산출:  reports/classifier.md (혼동행렬 · per-class F1 · 특징 중요도 · 한계 명시)
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

SEED = 42
TRAIN_END = pd.Timestamp("2026-05-22")
FEATURES = ["sload", "rpm", "srate", "drpm", "feed"]   # 센서 값만 (§8.3)
CLASSES = ["steady", "transient", "ready", "error"]
RPT_PATH = os.path.join(ROOT, "reports", "classifier.md")


def make_model(kind, weighted, cw):
    if kind == "RandomForest":
        return RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1,
                                      class_weight="balanced" if weighted else None)
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.2,
                             random_state=SEED, objective="multi:softprob",
                             num_class=len(CLASSES), n_jobs=-1, tree_method="hist",
                             verbosity=0)
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=SEED)


def boost_name():
    try:
        import xgboost  # noqa: F401
        return "XGBoost"
    except ImportError:
        return "GradientBoosting"


def cm_md(cm):
    """혼동행렬 markdown 표 (행=실제, 열=예측)."""
    head = "| 실제 \\ 예측 | " + " | ".join(CLASSES) + " |"
    sep = "|---" * (len(CLASSES) + 1) + "|"
    rows = [f"| **{c}** | " + " | ".join(f"{n:,}" for n in cm[i]) + " |"
            for i, c in enumerate(CLASSES)]
    return "\n".join([head, sep] + rows)


def main():
    df = pd.read_csv(os.path.join(ROOT, "data", "scored.csv.gz"), parse_dates=["ts"])
    y_all = pd.Categorical(df.state, categories=CLASSES).codes
    tr = (df.ts < TRAIN_END).to_numpy()
    Xtr, ytr = df.loc[tr, FEATURES].to_numpy(), y_all[tr]
    Xte, yte = df.loc[~tr, FEATURES].to_numpy(), y_all[~tr]

    dist = pd.DataFrame({
        "클래스": CLASSES,
        "학습": [int((ytr == i).sum()) for i in range(4)],
        "평가": [int((yte == i).sum()) for i in range(4)],
    })
    print(dist.to_string(index=False))

    # 클래스별 센서 프로파일(중앙값) — ready와 error가 센서로 구분 가능한지의 근거 자료
    prof = df.groupby("state")[FEATURES].median().reindex(CLASSES)
    print("\n클래스별 센서 중앙값:\n", prof.to_string())

    # balanced 가중치 (RandomForest 'balanced'와 동일 규칙: N / (K·n_k))
    cw = len(ytr) / (len(CLASSES) * np.bincount(ytr, minlength=len(CLASSES)).astype(float))
    print("\nbalanced 클래스 가중치:", {c: round(w, 1) for c, w in zip(CLASSES, cw)})

    results = []
    for kind in ["RandomForest", boost_name()]:
        for weighted in [False, True]:
            mdl = make_model(kind, weighted, cw)
            t0 = time.time()
            if kind != "RandomForest" and weighted:
                mdl.fit(Xtr, ytr, sample_weight=cw[ytr])
            else:
                mdl.fit(Xtr, ytr)
            yp = mdl.predict(Xte)
            p, r, f1, sup = precision_recall_fscore_support(
                yte, yp, labels=range(4), zero_division=0)
            res = dict(
                kind=kind, weighted=weighted, fit_sec=time.time() - t0,
                acc=accuracy_score(yte, yp),
                macro_f1=f1_score(yte, yp, average="macro", zero_division=0),
                cm=confusion_matrix(yte, yp, labels=range(4)),
                per_class=pd.DataFrame({"클래스": CLASSES, "정밀도": p, "재현율": r,
                                        "F1": f1, "표본수": sup}),
                imp=pd.Series(mdl.feature_importances_, index=FEATURES)
                      .sort_values(ascending=False),
            )
            results.append(res)
            tag = "balanced 가중치" if weighted else "가중치 없음"
            print(f"\n[{kind} · {tag}] 정확도 {res['acc']:.4f} · macro-F1 {res['macro_f1']:.4f}"
                  f" · 학습 {res['fit_sec']:.0f}초")
            print(res["per_class"].to_string(index=False))

    # ── 보고서 ──
    def pcf1(res, i):
        return res["per_class"].F1.iloc[i]

    summary = "\n".join(
        f"| {r['kind']} | {'balanced' if r['weighted'] else '없음'} | {r['acc']:.4f} | "
        f"{r['macro_f1']:.4f} | " + " | ".join(f"{pcf1(r, i):.3f}" for i in range(4)) + " |"
        for r in results)

    detail = []
    for r in results:
        tag = "balanced 가중치" if r["weighted"] else "가중치 없음"
        pc = "\n".join(
            f"| {row.클래스} | {row.정밀도:.3f} | {row.재현율:.3f} | {row.F1:.3f} | {int(row.표본수):,} |"
            for row in r["per_class"].itertuples())
        detail.append(f"""### {r['kind']} · {tag}

전체 정확도 {r['acc']:.4f} · macro-F1 {r['macro_f1']:.4f} (학습 {r['fit_sec']:.0f}초)

{cm_md(r['cm'])}

| 클래스 | 정밀도 | 재현율 | F1 | 평가 표본수 |
|---|---|---|---|---|
{pc}
""")

    imp_base = next(r for r in results if r["kind"] == "RandomForest" and not r["weighted"])
    imp_md = "\n".join(f"| {k} | {v:.3f} |" for k, v in imp_base["imp"].items())
    bf1 = [pcf1(imp_base, i) for i in range(4)]
    headline = f"""## 발표용 요약 — 반드시 세트로 인용

정확도 {imp_base['acc'] * 100:.1f}%만 단독으로 쓰지 말 것. 아래 셋이 한 세트:

1. 전체 정확도 {imp_base['acc'] * 100:.1f}% (RandomForest, 가중치 없음, 시간 기준 분할 평가)
2. 클래스별 F1 — steady {bf1[0]:.3f} / transient {bf1[1]:.3f} / ready {bf1[2]:.3f} / error {bf1[3]:.3f}
3. error는 센서값만으로 식별 불가 — ready(대기)와 센서값이 동일(회전 정지·부하 0·이송 0)해서
   구분 근거가 컨트롤러 상태 플래그뿐. 높은 정확도의 의미는 나머지 3개 상태
   (가공·가감속·대기)를 센서만으로 복원했다는 것.
"""
    dist_md = "\n".join(f"| {r.클래스} | {r.학습:,} | {r.평가:,} |" for r in dist.itertuples())
    prof_md = "\n".join(
        f"| {c} | " + " | ".join(f"{prof.loc[c, f]:,.0f}" for f in FEATURES) + " |"
        for c in CLASSES)
    cw_md = ", ".join(f"{c} {w:,.1f}" for c, w in zip(CLASSES, cw))

    md = f"""# 기계 상태 4분류 모델 — 성능 평가 보고서

수행계획서의 "기계 상태(정상·오류·알람·정지) 분류 ML 모델 + 성능 평가 보고서" 산출물.
실데이터의 대응 라벨은 파이프라인의 `state` 4분류: steady(정상 가공) / transient(가감속) /
ready(대기) / error(설비 알람). 수치는 `python experiments/classifier.py` 재실행으로 재현 가능
(seed={SEED}).

{headline}
## 실험 설계

- 입력: 센서 값 5개만 (sload, rpm, srate, drpm, feed). 라벨 산출에 쓰인 status·state와
  그 파생 피처는 배제
- 분할: 시간 기준 5/1~21 학습 / 5/22~31 평가. 무작위 셔플은 시계열 누수라 안 씀
- 불균형 대응: 가중치 적용/미적용 두 변형 모두 실험. 오버샘플링 안 씀(정직성 규칙 2)

| 클래스 | 학습(5/1~21) | 평가(5/22~31) |
|---|---|---|
{dist_md}

## 결과 요약

| 모델 | 클래스 가중치 | 정확도 | macro-F1 | steady F1 | transient F1 | ready F1 | error F1 |
|---|---|---|---|---|---|---|---|
{summary}

## 핵심 발견 — balanced 가중치가 무너지는 이유

balanced 가중치는 표본 수 역수 가중({cw_md}). error는 학습 표본 {int(dist.학습.iloc[3])}개뿐이라
사이클 하나가 수만 배 무게를 갖는데, error의 센서값이 ready와 사실상 같다는 게 문제:

| 클래스 | sload 중앙값 | rpm | srate | drpm | feed |
|---|---|---|---|---|---|
{prof_md}

둘 다 "회전 정지·부하 0·이송 0"이라 센서만으로는 구분 근거가 없고, 라벨 차이는 컨트롤러
상태 플래그(status)뿐. 그래서 balanced 변형에서는 대기 대부분이 error로 쏠려 정확도가 무너짐.
모델 결함이 아니라 "error는 센서가 아닌 컨트롤러 플래그로만 식별 가능"이라는 데이터 성질이며,
알람코드 미기록(대시보드 탭③)과 함께 로깅 개선이 필요한 지점.

결론: 대표 수치는 가중치 없음 변형으로 읽는 것이 타당. error는 센서 분류 대상이 아니라
status 플래그로 직접 판정하는 게 맞음(현 파이프라인 방식).

## 상세 결과

{chr(10).join(detail)}
## 특징 중요도 (RandomForest, 가중치 없음)

| 특징 | 중요도 |
|---|---|
{imp_md}

## 정직성 규칙 — 이 수치 읽는 법

1. 라벨은 사람 라벨이 아니라 규칙 기반 산출. state는 status·회전수 갭·drpm 규칙으로 계산된
   값이므로, 높은 정확도는 "고장 예지 능력"이 아니라 "센서값만으로 상태 규칙을 복원했다"는 의미.
   계획서의 4분류 요건은 충족하되 이 한계를 숨기지 않음.
2. error 표본은 극소 (학습 {int(dist.학습.iloc[3])} / 평가 {int(dist.평가.iloc[3])} 사이클, 설비이벤트
   유래). error 행 수치는 통계적 의미가 약하고, 오버샘플링으로 부풀리지 않고 그대로 보고.

## 추가 한계

- transient는 steady와의 경계가 rpm 갭 200 규칙으로 정의됨 → 경계 근처 사이클 오분류가 대부분
- 시간 분할이라 평가 구간에 학습에 없던 운전조건 포함 — 실운영 배포와 같은 조건, 의도된 설계
"""
    os.makedirs(os.path.dirname(RPT_PATH), exist_ok=True)
    with open(RPT_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n저장: {os.path.relpath(RPT_PATH, ROOT)}")


if __name__ == "__main__":
    main()
