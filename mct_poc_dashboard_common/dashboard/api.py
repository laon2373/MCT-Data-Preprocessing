# -*- coding: utf-8 -*-
"""MCT 3호 실시간 예측 API (CLAUDE.md §8.4 — 계획서의 'FastAPI 실시간 예측 API')

실행:  uvicorn api:app --port 8000        (대화형 문서: http://localhost:8000/docs)
필요:  data/model_artifacts.joblib  ←  python regenerate.py <로그폴더> --save-model 로 생성.

설계 원칙:
- 컨텍스트 lookup·폴백·novel 판정·채널 수식은 pipeline_core.py를 임포트한다 —
  regenerate.py(배치)와 동일 로직 보장(복붙 금지, §8.4).
- novel(학습되지 않은 공구×RPM 조합)은 이상이 아니다. 응답에 그 사실을 명시한다(§1-3).
- app.py(대시보드)는 이 모듈을 임포트하지 않는다 — 대시보드 단독 구동 원칙 유지.

채널 판정 범위 (README에도 명시):
- POST /predict  — 한 사이클: state·zload·novel·instant_exceed(순간 임계 초과 여부)·기준통계까지만.
  순간 채널의 알람 '확정'은 6사이클(≈30초) 연속 지속이 조건이고, 드리프트(EWMA) 채널은 이력이
  필요하므로 단일 사이클로는 판정 불가 → 두 판정 모두 /predict_batch에서만 반환한다.
- POST /predict_batch — 시간순 최근 N사이클 배열: 지속성 규칙·EWMA를 적용해 알람 확정을 반환.
  EWMA는 배열 안에서만 누적되므로 충분한 이력(권장 60사이클 이상)을 함께 보내야 한다.
"""
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import pipeline_core as core

ART_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "data", "model_artifacts.joblib")

app = FastAPI(
    title="MCT 3호 M2* 예측 API",
    description="조건부 잔차 이상탐지(M2*)의 실시간 서빙. 알람이 아닌 novel(신규 운전조건)은 "
                "'판단 불가'로 별도 구분해 반환한다.",
    version="1.0",
)

_art = None


def art():
    global _art
    if _art is None:
        if not os.path.exists(ART_PATH):
            raise HTTPException(status_code=503, detail=(
                "모델 산출물이 없습니다: data/model_artifacts.joblib — "
                "`python regenerate.py <로그폴더> --save-model`로 먼저 생성하세요."))
        _art = core.load_artifacts(ART_PATH)
    return _art


class Cycle(BaseModel):
    """한 수집 사이클(≈5.5초)의 센서 값."""
    status: str = Field("STATUS_RUN", description="설비 상태 문자열 (STATUS_RUN / STATUS_ERROR / 그 외=대기)")
    tool: int = Field(..., description="현재 공구 번호")
    srate: float = Field(..., description="지령 스핀들 회전수 (rpm)")
    rpm: float = Field(..., description="실제 스핀들 회전수 (rpm)")
    sload: float = Field(..., description="스핀들 부하 (%)")
    feed: float = Field(..., description="이송 속도")
    drpm: float = Field(0.0, description="회전수 변화량 |Δrpm|")


class BatchIn(BaseModel):
    cycles: list[Cycle] = Field(..., description="시간순(과거→현재) 사이클 배열. EWMA·지속성 판정에 "
                                                 "충분한 이력(권장 60사이클 이상)을 포함할 것")


def classify_one(c: Cycle):
    """단일 사이클 상태 판정 — pipeline_core.classify와 동일 규칙."""
    if c.status == "STATUS_ERROR":
        return "error"
    if c.status != "STATUS_RUN":
        return "ready"
    gap = abs(c.rpm - c.srate)
    return "steady" if (gap <= core.GAP_RPM and c.drpm <= core.GAP_RPM) else "transient"


@app.get("/health")
def health():
    a = art()
    return dict(status="ok", device=a["meta"].get("device"),
                train_range=a["meta"].get("train_range"), thresholds=a["thr"])


@app.post("/predict")
def predict(c: Cycle):
    """한 사이클 판정. 순간 채널의 30초 지속 확정과 드리프트(EWMA) 판정은 이력이 필요하므로
    여기서는 반환하지 않는다(null) — /predict_batch를 사용할 것."""
    a = art()
    state = classify_one(c)
    band = int(round(c.srate / 1000))
    base = dict(state=state, tool=c.tool, band=f"{band}k")
    if state != "steady":
        return dict(**base, zload=None, novel=None, instant_exceed=None, drift_exceed=None,
                    note="이상 판정은 steady(정상 가공)에서만 수행한다 — 현재 상태는 판정 대상 아님")
    novel = core.is_novel(a["stats"], c.tool, band)
    (ms, ss), (mf, sf) = core.lookup(a["stats"], c.tool, band)
    zload = (c.sload - ms) / ss
    기준통계 = dict(기대부하=round(ms, 2), 부하스케일=round(ss, 2),
                기대피드=round(mf, 2), 피드스케일=round(sf, 2))
    if novel:
        return dict(**base, zload=round(float(zload), 2), novel=True,
                    instant_exceed=None, drift_exceed=None, 기준통계=기준통계,
                    note="이상 아님 — 학습되지 않은 조건(공구×RPM 조합)이라 판단 불가. "
                         "알람이 아닌 '신규 운전조건 알림' 채널로 처리할 것")
    return dict(**base, zload=round(float(zload), 2), novel=False,
                instant_exceed=bool(abs(zload) > a["thr"]["thr_z"]),
                drift_exceed=None, 기준통계=기준통계,
                note="instant_exceed는 '순간 임계 초과 여부'까지만 — 알람 확정(6사이클≈30초 지속)과 "
                     "드리프트 판정은 POST /predict_batch 사용")


@app.post("/predict_batch")
def predict_batch(body: BatchIn):
    """시간순 사이클 배열 판정 — 지속성 규칙(6사이클)과 컨텍스트 EWMA를 적용해 알람을 확정한다."""
    a = art()
    if not body.cycles:
        raise HTTPException(status_code=422, detail="cycles가 비어 있습니다")
    df = pd.DataFrame([c.model_dump() for c in body.cycles])
    df = core.classify(df)
    sd = df[df.state == "steady"].copy()
    n_steady = len(sd)
    if n_steady == 0:
        return dict(n_cycles=len(df), n_steady=0, alarm_confirmed=False, cycles=[],
                    note="steady 사이클이 없어 이상 판정을 수행하지 않음")
    sd = core.score_steady(sd, a["stats"])
    sd["m3"] = -a["if3"].score_samples(sd[core.CF])
    sd = core.flag_channels(sd, a["thr"])

    out = df[["state", "tool", "band"]].to_dict("records")
    for i, r in zip(sd.index, sd.itertuples()):
        out[i].update(zload=round(r.zload, 2), ewma=round(r.ewma, 2), novel=bool(r.novel),
                      instant_alarm=bool(r.f_pt), drift_alarm=bool(r.f_ew), aux_m3=bool(r.f_m3))
        if r.novel:
            out[i]["note"] = "이상 아님 — 학습되지 않은 조건, 신규 운전조건 알림 채널"
    alarm = bool((sd.f_pt | sd.f_ew).any())
    return dict(n_cycles=len(df), n_steady=n_steady, alarm_confirmed=alarm,
                n_novel=int(sd.novel.sum()), thresholds=a["thr"], cycles=out,
                note="EWMA·지속성은 이 배열 안에서만 누적 계산됨 — 짧은 배열은 드리프트 판정이 "
                     "실제보다 보수적일 수 있음(권장 60사이클 이상)")
