# -*- coding: utf-8 -*-
"""
M2* 업로드 분석 파이프라인

목적
----
사용자가 올린 .log 또는 여러 .log가 들어 있는 .zip을
지도학습과 같은 공통 파서/정제 로직으로 읽은 뒤,
이미 학습되어 저장된 M2* 정상 기준(model_artifacts.joblib)을 이용해 추론한다.

중요
----
- 업로드 데이터로 M2*를 다시 학습하지 않는다.
- 공통 전처리의 실제 절삭 조건:
    STATUS_RUN & SPINDLE_RPM > 0 & FEED_RATE > 0
- 전체 원본 측정 행은 관제/운영 리포트용으로 보존한다.
- M2* 이상 스코어링만 실제 절삭 데이터에서 수행한다.
"""

from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
ARTIFACT_PATH = BASE_DIR / "data" / "model_artifacts.joblib"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# XGBoost가 사용하는 것과 동일한 공통 원본 로그 파서/정제 함수
from inference_pipeline import parse_raw_log, prepare_clean_data  # noqa: E402
import pipeline_core as core  # noqa: E402


MAX_ZIP_FILES = 100
MAX_ZIP_UNCOMPRESSED_BYTES = 1_500_000_000


# ---------------------------------------------------------------------
# 공통 schema 변환
# ---------------------------------------------------------------------

def clean_to_m2_schema(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    공통 clean 데이터 컬럼을 M2* 내부 컬럼명으로 연결한다.
    전체 행을 유지한다.
    """
    raw = clean_df.copy()
    raw.columns = [str(c).strip().lower() for c in raw.columns]

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

    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            "공통 전처리 데이터에 필요한 컬럼이 없습니다: "
            + ", ".join(missing)
        )

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

    if pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    else:
        df["ts"] = pd.to_datetime(
            df["ts"],
            format="%Y-%m-%d %H:%M:%S,%f",
            errors="coerce",
        )

    for col in ["tool", "feed", "srate", "sload", "rpm", "drpm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "is_emergency" in raw.columns:
        df["emg"] = (
            pd.to_numeric(raw["is_emergency"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        df["emg"] = 0

    if "is_alarm" in raw.columns:
        df["alarm"] = (
            pd.to_numeric(raw["is_alarm"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        df["alarm"] = 0

    df = (
        df.dropna(
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

    return df


# ---------------------------------------------------------------------
# LOG / ZIP 파싱
# ---------------------------------------------------------------------

def _parse_one_log_path(log_path: Path) -> tuple[pd.DataFrame, dict]:
    """
    원본 LOG를 21열 parsed DataFrame으로만 변환한다.

    ZIP에서는 모든 파일을 먼저 파싱한 뒤 한 번만 prepare_clean_data()를
    호출한다. 이렇게 해야 월간 ZIP에서 날짜별로 Cleaning을 31번 반복하지
    않고, 전체 월 데이터 기준 interval_seconds도 한 번에 계산된다.
    """
    parsed_df, written_rows, skipped_lines = parse_raw_log(log_path)

    return parsed_df, {
        "filename": log_path.name,
        "written_rows": int(written_rows),
        "skipped_lines": int(skipped_lines),
        "parsed_rows": int(len(parsed_df)),
    }


def parse_uploaded_log(
    filename: str,
    file_bytes: bytes,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """단일 .log bytes를 공통 파서로 정제한다."""
    with tempfile.TemporaryDirectory(prefix="m2_upload_") as tmp:
        path = Path(tmp) / Path(filename).name
        path.write_bytes(file_bytes)

        if progress_callback:
            progress_callback("parsing", 0, 1, Path(filename).name)

        parsed_df, info = _parse_one_log_path(path)

        if progress_callback:
            progress_callback("parsing", 1, 1, Path(filename).name)
            progress_callback("cleaning", 0, 1, "공통 전처리")

        clean_df = prepare_clean_data(parsed_df)
        info["clean_rows"] = int(len(clean_df))

        if progress_callback:
            progress_callback("cleaning", 1, 1, "공통 전처리")

    return clean_df, [info]


def parse_uploaded_zip(
    file_bytes: bytes,
    progress_callback=None,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    여러 .log가 들어 있는 ZIP을 분석한다.

    성능 개선:
    1) 31개 LOG를 각각 Parsing
    2) parsed DataFrame을 모두 합침
    3) 공통 Cleaning은 월 전체에 대해 딱 1번 수행

    기존 버전처럼 날짜별 Cleaning을 반복하지 않으므로 월간 ZIP에서
    불필요한 정렬·datetime 변환·파생열 계산을 크게 줄인다.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(file_bytes))
    except zipfile.BadZipFile as exc:
        raise ValueError("정상적인 ZIP 파일이 아닙니다.") from exc

    members = [
        info
        for info in archive.infolist()
        if (
            not info.is_dir()
            and info.filename.lower().endswith(".log")
            and "__macosx/" not in info.filename.lower()
        )
    ]

    if not members:
        raise ValueError("ZIP 안에서 .log 파일을 찾지 못했습니다.")

    if len(members) > MAX_ZIP_FILES:
        raise ValueError(
            f"ZIP에는 최대 {MAX_ZIP_FILES}개의 .log 파일만 넣을 수 있습니다."
        )

    total_uncompressed = sum(info.file_size for info in members)
    if total_uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
        raise ValueError("ZIP 압축 해제 후 전체 크기가 허용 범위를 초과합니다.")

    members = sorted(members, key=lambda x: x.filename)

    parsed_frames = []
    file_infos = []

    with tempfile.TemporaryDirectory(prefix="m2_zip_") as tmp:
        tmp_dir = Path(tmp)

        for index, info in enumerate(members, start=1):
            original_name = Path(info.filename).name
            safe_name = f"{index:03d}_{original_name}"
            path = tmp_dir / safe_name

            if progress_callback:
                progress_callback(
                    "parsing",
                    index - 1,
                    len(members),
                    original_name,
                )

            with archive.open(info) as source, open(path, "wb") as target:
                # read() 전체를 한 번 더 복사하지 않고 스트림으로 기록
                import shutil
                shutil.copyfileobj(source, target, length=1024 * 1024)

            parsed_df, parse_info = _parse_one_log_path(path)
            parse_info["filename"] = original_name

            parsed_frames.append(parsed_df)
            file_infos.append(parse_info)

            if progress_callback:
                progress_callback(
                    "parsing",
                    index,
                    len(members),
                    original_name,
                )

    if progress_callback:
        progress_callback("cleaning", 0, 1, "월간 데이터 통합/공통 전처리")

    parsed_all = pd.concat(
        parsed_frames,
        ignore_index=True,
        copy=False,
    )

    # 핵심: 31일을 합친 뒤 공통 Cleaning을 단 한 번 수행
    clean_df = prepare_clean_data(parsed_all)

    # 총 clean 행은 전체 기준이며, 각 파일의 parsed_rows 합과 동일해야 한다.
    if len(clean_df) != sum(x["parsed_rows"] for x in file_infos):
        raise RuntimeError(
            "ZIP 통합 Cleaning 전후 행 수가 일치하지 않습니다."
        )

    if progress_callback:
        progress_callback("cleaning", 1, 1, "월간 데이터 통합/공통 전처리")

    return clean_df, file_infos

def parse_uploaded_file(
    filename: str,
    file_bytes: bytes,
    progress_callback=None,
):
    suffix = Path(filename).suffix.lower()

    if suffix == ".log":
        return parse_uploaded_log(
            filename,
            file_bytes,
            progress_callback=progress_callback,
        )

    if suffix == ".zip":
        return parse_uploaded_zip(
            file_bytes,
            progress_callback=progress_callback,
        )

    raise ValueError(".log 또는 .zip 파일만 분석할 수 있습니다.")


# ---------------------------------------------------------------------
# Episode 생성 — 기존 regenerate.py와 같은 방식
# ---------------------------------------------------------------------

def episodes(sub: pd.DataFrame, gap: int = 60) -> list[pd.DataFrame]:
    out = []
    current = []
    last = None

    for _, row in sub.sort_values("ts").iterrows():
        if last is not None and (row.ts - last).total_seconds() > gap:
            out.append(pd.DataFrame(current))
            current = []

        current.append(row)
        last = row.ts

    if current:
        out.append(pd.DataFrame(current))

    return out


def make_alarm_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    alarm_rows = df[df["f_pt"] | df["f_ew"]]

    for e in episodes(alarm_rows):
        channel = "+".join(
            c
            for c, flag in [
                ("순간", bool(e["f_pt"].any())),
                ("드리프트", bool(e["f_ew"].any())),
            ]
            if flag
        )

        tool = int(e["tool"].mode().iat[0])
        band = int(e["band"].mode().iat[0])
        expected = float(e["ms"].median())
        max_z = float(e["zload"].abs().max())
        duration = float(
            (e["ts"].iloc[-1] - e["ts"].iloc[0]).total_seconds() + 5.5
        )
        direction = "높음" if float(e["zload"].mean()) > 0 else "낮음"

        rows.append(
            {
                "시작": e["ts"].iloc[0],
                "종료": e["ts"].iloc[-1],
                "지속초": round(duration),
                "채널": channel,
                "공구": f"T{tool:02d}",
                "RPM대역": f"{band}k",
                "기대부하": round(expected, 1),
                "실측": f"{e['sload'].min():.0f}~{e['sload'].max():.0f}%",
                "최대z": round(max_z, 1),
                "설명": (
                    f"T{tool:02d}·{band}k rpm: 기대 {expected:.0f}% 대비 "
                    f"부하 {direction} 상태 {duration:.0f}초 지속 "
                    f"(최대|z|={max_z:.1f})"
                ),
            }
        )

    columns = [
        "시작", "종료", "지속초", "채널", "공구",
        "RPM대역", "기대부하", "실측", "최대z", "설명",
    ]
    return pd.DataFrame(rows, columns=columns)


def make_novel_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for e in episodes(df[df["novel"]], gap=300):
        conditions = sorted(
            {
                f"T{int(t):02d}·{int(b)}k"
                for t, b in zip(e["tool"], e["band"])
            }
        )[:4]

        rows.append(
            {
                "시작": e["ts"].iloc[0],
                "종료": e["ts"].iloc[-1],
                "사이클수": len(e),
                "조건": ", ".join(conditions),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["시작", "종료", "사이클수", "조건"],
    )


def make_event_table(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    event_rows = df_all[
        (df_all["status"] == "STATUS_ERROR")
        | (df_all["emg"] != 0)
    ]

    for e in episodes(event_rows, gap=120):
        event_type = (
            "설비 알람(ERROR)"
            if (e["status"] == "STATUS_ERROR").any()
            else "비상신호(EMG)"
        )

        rows.append(
            {
                "시각": e["ts"].iloc[0],
                "유형": event_type,
                "지속초": round(
                    (e["ts"].iloc[-1] - e["ts"].iloc[0]).total_seconds() + 5.5
                ),
            }
        )

    return pd.DataFrame(rows, columns=["시각", "유형", "지속초"])


def make_daily_table(
    df_all: pd.DataFrame,
    alarms: pd.DataFrame,
    novel: pd.DataFrame,
) -> pd.DataFrame:
    work = df_all.copy()
    work["date"] = work["ts"].dt.date

    daily = (
        work.groupby("date")
        .agg(
            사이클=("ts", "size"),
            가동률=("status", lambda s: (s == "STATUS_RUN").mean() * 100),
        )
        .reset_index()
    )

    for name, table in [("이상알람", alarms), ("신규조건", novel)]:
        if len(table):
            count = (
                table.set_index(table["시작"].dt.date)
                .groupby(level=0)
                .size()
            )
        else:
            count = pd.Series(dtype=int)

        daily[name] = daily["date"].map(count).fillna(0).astype(int)

    return daily


# ---------------------------------------------------------------------
# M2* 저장 모델 추론
# ---------------------------------------------------------------------

def run_m2_inference_from_clean(
    clean_df: pd.DataFrame,
    file_infos: list[dict] | None = None,
    progress_callback=None,
) -> dict:
    """
    이미 저장된 M2* 정상 기준으로 새 clean 데이터만 추론한다.
    업로드 데이터로 stats/thr/IF를 재학습하지 않는다.
    """
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"M2* 모델 산출물을 찾지 못했습니다: {ARTIFACT_PATH}"
        )

    if progress_callback:
        progress_callback("inference", 0, 1, "M2* 이상 탐지")

    art = core.load_artifacts(ARTIFACT_PATH)
    stats = art["stats"]
    thr = art["thr"]
    if3 = art["if3"]
    trained_meta = art.get("meta", {})

    df_all = clean_to_m2_schema(clean_df)

    if df_all.empty:
        raise ValueError("정제 후 분석 가능한 데이터가 없습니다.")

    # 관제/운영용 전체 데이터의 state는 유지
    df_all = core.classify(df_all)

    # 지도학습과 동일한 '실제 절삭' 정의만 M2* 스코어링에 사용
    cutting_mask = (
        (df_all["status"] == "STATUS_RUN")
        & (df_all["rpm"] > 0)
        & (df_all["feed"] > 0)
    )
    df_model = df_all[cutting_mask].copy()

    # 결과 컬럼은 전체 데이터에 만든 뒤 모델 대상 행만 채운다.
    for c in ["zload", "ewma", "m3", "ms", "ss"]:
        df_all[c] = np.nan

    for c in ["novel", "f_pt", "f_ew", "f_m3"]:
        df_all[c] = False

    if not df_model.empty:
        sd = df_model[df_model["state"] == "steady"].copy()

        if not sd.empty:
            sd = core.score_steady(sd, stats)
            sd["m3"] = -if3.score_samples(sd[core.CF])
            sd = core.flag_channels(sd, thr)

            columns = [
                "zload", "ewma", "m3", "ms", "ss",
                "novel", "f_pt", "f_ew", "f_m3",
            ]
            df_all.loc[sd.index, columns] = sd[columns].values

    alarms = make_alarm_table(df_all)
    novel = make_novel_table(df_all)
    events = make_event_table(df_all)
    daily = make_daily_table(df_all, alarms, novel)

    meta = {
        "thr_z": float(thr["thr_z"]),
        "thr_ew": float(thr["thr_ew"]),
        "thr_m3": float(thr["thr_m3"]),
        "device": trained_meta.get("device", "MCT 3호"),
        "train_range": trained_meta.get("train_range", "-"),
        "note": "저장된 M2* 정상 기준으로 업로드 로그를 추론함",
        "upload_start": str(df_all["ts"].min()),
        "upload_end": str(df_all["ts"].max()),
    }

    if progress_callback:
        progress_callback("inference", 1, 1, "M2* 이상 탐지")

    return {
        "df": df_all,
        "alarms": alarms,
        "novel": novel,
        "events": events,
        "daily": daily,
        "meta": meta,
        "file_infos": file_infos or [],
        "summary": {
            "total_rows": int(len(df_all)),
            "cutting_rows": int(cutting_mask.sum()),
            "steady_cutting_rows": int(
                (cutting_mask & (df_all["state"] == "steady")).sum()
            ),
            "alarm_count": int(len(alarms)),
            "novel_count": int(len(novel)),
            "event_count": int(len(events)),
            "file_count": int(len(file_infos or [])),
            "start": df_all["ts"].min(),
            "end": df_all["ts"].max(),
        },
    }


def analyze_uploaded_file(
    filename: str,
    file_bytes: bytes,
    progress_callback=None,
) -> dict:
    clean_df, file_infos = parse_uploaded_file(
        filename,
        file_bytes,
        progress_callback=progress_callback,
    )
    return run_m2_inference_from_clean(
        clean_df,
        file_infos=file_infos,
        progress_callback=progress_callback,
    )
