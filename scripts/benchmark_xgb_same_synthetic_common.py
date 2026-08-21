# -*- coding: utf-8 -*-
"""
XGBoost vs M2* — 동일 합성 이상 시나리오 비교

팀원의 M2* benchmark.py가 사용한 것과 동일한:
- 시험 기간: 2026-05-22 ~ 2026-05-31
- seed: 42
- 유형별 60개 주입 위치
- 지속 과부하: Spindle Load +12, 11분
- 공구 파손형: Spindle Load 0~5로 강제, 3분
- 완만한 저하: 0 → +8 선형 증가, 2.3시간

를 그대로 재현하여 현재 XGBoost가 같은 주입 위치에서
얼마나 탐지하는지 확인한다.

중요:
- 실제 고장 데이터가 아니라 합성 이상 비교 실험이다.
- M2*는 cycle 단위, XGBoost는 1분 Window 단위이므로
  평균 탐지시간을 절대적인 우열로 해석하지 않는다.
- XGBoost가 분석할 수 없는 공구/Window는 '미탐'이 아니라
  '분석 불가'로 별도 집계한다.

실행:
python scripts/benchmark_xgb_same_synthetic_common.py --unsupervised "mct_poc_dashboard_common/dashboard"

필요 파일:
- 공통 전처리를 적용해 다시 생성한 M2* dashboard 폴더 또는 ZIP
- models/xgboost_model.json
- models/feature_columns.json
- models/train_normal_baseline.csv
- reports/model_comparison/comparison_summary.csv

출력:
reports/model_comparison/
├─ synthetic_same_scenario_detail.csv
├─ synthetic_same_scenario_comparison.csv
└─ synthetic_same_scenario_report.md
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = 42
N_INJ = 60
TRAIN_END = pd.Timestamp("2026-05-22")
KNOWN_MIN = 0.8
MIN_WINDOW_SAMPLES = 9
SEGMENT_GAP_SEC = 15
WINDOW_SEC = 60

INJ_TYPES = [
    ("지속과부하", 660, 60),
    ("공구파손", 180, 16),
    ("완만한저하", 8280, 750),
]

LABELS = {
    "지속과부하": "지속 과부하",
    "공구파손": "공구 파손형",
    "완만한저하": "완만한 저하",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost와 M2*의 동일 합성 이상 시나리오 비교"
    )
    parser.add_argument(
        "--unsupervised",
        required=True,
        help="공통 전처리 적용 M2* dashboard 폴더 또는 ZIP 경로",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="XGBoost 탐지 기준. 생략 시 comparison_summary.csv의 calibration P99 사용",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def read_source_member(source: Path, filename: str) -> bytes:
    """
    source가 dashboard 폴더여도 되고 ZIP이어도 된다.

    폴더 예:
      mct_poc_dashboard_common/dashboard
      └─ data/
         ├─ scored.csv.gz
         └─ bench.csv

    ZIP 예:
      기존 팀원 ZIP처럼 내부 어딘가에
      dashboard/data/scored.csv.gz, dashboard/data/bench.csv가 존재
    """
    if source.is_dir():
        direct = source / "data" / filename

        if direct.exists():
            return direct.read_bytes()

        matches = list(source.rglob(filename))

        if matches:
            return matches[0].read_bytes()

        raise FileNotFoundError(
            f"폴더 안에서 data/{filename}을 찾지 못했습니다: {source}"
        )

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as zf:
            matches = [
                name
                for name in zf.namelist()
                if name.replace("\\", "/").endswith(
                    f"/data/{filename}"
                )
                or name.replace("\\", "/") == f"data/{filename}"
            ]

            if not matches:
                raise FileNotFoundError(
                    f"ZIP 안에서 data/{filename}을 찾지 못했습니다."
                )

            return zf.read(matches[0])

    raise ValueError(
        "--unsupervised에는 dashboard 폴더 또는 .zip 파일을 지정해주세요."
    )


def load_teammate_data(source: Path):
    scored_raw = read_source_member(
        source,
        "scored.csv.gz",
    )

    bench_raw = read_source_member(
        source,
        "bench.csv",
    )

    scored = pd.read_csv(
        io.BytesIO(scored_raw),
        compression="gzip",
        parse_dates=["ts"],
        low_memory=False,
    )

    bench = pd.read_csv(
        io.BytesIO(bench_raw),
    )

    steady = (
        scored[scored["state"] == "steady"]
        .copy()
        .reset_index(drop=True)
    )

    return steady, bench


def load_xgb_artifacts():
    model_path = PROJECT_ROOT / "models" / "xgboost_model.json"
    feature_path = PROJECT_ROOT / "models" / "feature_columns.json"
    baseline_path = PROJECT_ROOT / "models" / "train_normal_baseline.csv"

    for path in [model_path, feature_path, baseline_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"필요 파일이 없습니다: {path}"
            )

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    meta = json.loads(
        feature_path.read_text(encoding="utf-8")
    )
    feature_columns = meta["feature_columns"]

    baseline = pd.read_csv(baseline_path)
    baseline["current_tool"] = pd.to_numeric(
        baseline["current_tool"],
        errors="coerce",
    )
    baseline = baseline.dropna(
        subset=["current_tool"]
    ).copy()
    baseline["current_tool"] = baseline["current_tool"].astype(int)

    return model, feature_columns, baseline


def load_xgb_threshold(user_threshold):
    if user_threshold is not None:
        return float(user_threshold), "사용자 지정"

    summary_path = (
        PROJECT_ROOT
        / "reports"
        / "model_comparison"
        / "comparison_summary.csv"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            "comparison_summary.csv가 없습니다.\n"
            "먼저 이전 단계의 지도/비지도 실제 로그 비교 스크립트를 실행하거나,\n"
            "--threshold 0.123943 처럼 직접 기준을 지정해주세요."
        )

    summary = pd.read_csv(summary_path)

    if "XGBoost 기준" not in summary.columns:
        raise ValueError(
            "comparison_summary.csv에서 'XGBoost 기준' 컬럼을 찾지 못했습니다."
        )

    threshold = float(summary.iloc[0]["XGBoost 기준"])
    return threshold, "5/1~5/21 실제 로그 calibration P99"


def sample_sites(sd: pd.DataFrame):
    """
    팀원 benchmark.py의 sample_sites()와 같은 seed, 같은 조건, 같은 RNG 호출 순서를 사용.
    """
    rng = np.random.default_rng(SEED)

    ts = sd["ts"].to_numpy()
    novel = sd["novel"].astype(bool).to_numpy()
    clean_alarm = (
        sd["f_pt"].astype(bool)
        | sd["f_ew"].astype(bool)
    ).to_numpy()

    test_pos = np.where(
        ts >= np.datetime64(TRAIN_END)
    )[0]

    t_max = ts[-1]
    sites = {}

    for key, dur, min_c in INJ_TYPES:
        got = []
        seen = set()

        while len(got) < N_INJ:
            j0 = int(rng.choice(test_pos))

            if j0 in seen:
                continue
            seen.add(j0)

            t0 = ts[j0]
            t1 = t0 + np.timedelta64(dur, "s")

            if t1 > t_max:
                continue

            j1 = int(
                np.searchsorted(
                    ts,
                    t1,
                    side="right",
                )
            )

            if j1 - j0 < min_c:
                continue

            if (
                novel[j0:j1].mean() > 1 - KNOWN_MIN
                or clean_alarm[j0:j1].any()
            ):
                continue

            elapsed = (
                ts[j0:j1] - t0
            ) / np.timedelta64(1, "s")

            base = (
                sd["sload"]
                .to_numpy()[j0:j1]
                .astype(float)
            )

            if key == "지속과부하":
                inj = base + 12.0

            elif key == "공구파손":
                inj = rng.uniform(
                    0.0,
                    5.0,
                    j1 - j0,
                )

            else:
                inj = (
                    base
                    + 8.0
                    * elapsed
                    / dur
                )

            got.append(
                {
                    "j0": j0,
                    "j1": j1,
                    "inj": np.asarray(inj, dtype=float),
                    "start": pd.Timestamp(ts[j0]),
                    "end": pd.Timestamp(ts[j1 - 1]),
                }
            )

        sites[key] = got

    return sites


def make_baseline_lookup(baseline: pd.DataFrame):
    required = [
        "current_tool",
        "load_center",
        "rpm_std_center",
        "rpmerr_center",
        "rpmerr_tol",
    ]

    missing = [
        c for c in required
        if c not in baseline.columns
    ]

    if missing:
        raise ValueError(
            "train_normal_baseline.csv 필수 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    lookup = {}

    for _, row in baseline.iterrows():
        tool = int(row["current_tool"])
        lookup[tool] = {
            "load_center": float(row["load_center"]),
            "rpm_std_center": float(row["rpm_std_center"]),
            "rpmerr_center": float(row["rpmerr_center"]),
            "rpmerr_tol": float(row["rpmerr_tol"]),
        }

    return lookup


def split_segments(df: pd.DataFrame):
    """
    XGBoost 서비스의 핵심 원칙과 맞춰:
    - 실제 절삭: RUN & RPM>0 & FEED>0
    - 공구 변경 또는 15초 초과 gap이면 새 Segment
    """
    if df.empty:
        return []

    d = df.copy()

    d = d[
        (d["status"] == "STATUS_RUN")
        & (pd.to_numeric(d["rpm"], errors="coerce") > 0)
        & (pd.to_numeric(d["feed"], errors="coerce") > 0)
    ].copy()

    if d.empty:
        return []

    d = d.sort_values("ts").reset_index(drop=True)

    gap = d["ts"].diff().dt.total_seconds()
    tool_changed = d["tool"].ne(d["tool"].shift())

    new_seg = (
        gap.isna()
        | (gap > SEGMENT_GAP_SEC)
        | tool_changed
    )

    d["_segment"] = new_seg.cumsum()

    return [
        g.drop(columns="_segment").copy()
        for _, g in d.groupby("_segment", sort=False)
    ]


def feature_row(window: pd.DataFrame, tool: int, baseline_lookup):
    if len(window) < MIN_WINDOW_SAMPLES:
        return None

    if tool not in baseline_lookup:
        return None

    rpm = pd.to_numeric(
        window["rpm"],
        errors="coerce",
    ).astype(float)

    srate = pd.to_numeric(
        window["srate"],
        errors="coerce",
    ).astype(float)

    load = pd.to_numeric(
        window["sload"],
        errors="coerce",
    ).astype(float)

    valid = (
        rpm.notna()
        & srate.notna()
        & load.notna()
    )

    rpm = rpm[valid]
    srate = srate[valid]
    load = load[valid]

    if len(rpm) < MIN_WINDOW_SAMPLES:
        return None

    rpm_error = srate - rpm

    rpm_mean = float(rpm.mean())
    rpm_std = float(rpm.std(ddof=1))
    rpm_min = float(rpm.min())
    rpm_max = float(rpm.max())
    rpm_range = rpm_max - rpm_min

    load_mean = float(load.mean())
    load_std = float(load.std(ddof=1))
    load_max = float(load.max())

    rpm_error_mean = float(rpm_error.mean())
    rpm_error_std = float(rpm_error.std(ddof=1))
    rpm_error_absmax = float(rpm_error.abs().max())

    b = baseline_lookup[tool]

    load_vs_normal = (
        load_mean
        / max(
            float(b["load_center"]),
            1.0,
        )
    )

    rpm_std_vs_normal = (
        rpm_std
        / max(
            float(b["rpm_std_center"]),
            1.0,
        )
    )

    rpm_error_vs_normal = (
        abs(
            rpm_error_mean
            - float(b["rpmerr_center"])
        )
        / max(
            float(b["rpmerr_tol"]),
            1.0,
        )
    )

    return {
        "rpm_mean": rpm_mean,
        "rpm_std": rpm_std,
        "rpm_min": rpm_min,
        "rpm_max": rpm_max,
        "rpm_range": rpm_range,
        "load_mean": load_mean,
        "load_std": load_std,
        "load_max": load_max,
        "rpm_error_mean": rpm_error_mean,
        "rpm_error_std": rpm_error_std,
        "rpm_error_absmax": rpm_error_absmax,
        "load_vs_normal": load_vs_normal,
        "rpm_std_vs_normal": rpm_std_vs_normal,
        "rpm_error_vs_normal": rpm_error_vs_normal,
    }


def build_windows_for_site(
    site_df,
    baseline_lookup,
):
    """
    주입 시작 시점부터 각 연속 절삭 Segment를 60초 단위로 나눔.
    """
    windows = []

    for seg in split_segments(site_df):
        if seg.empty:
            continue

        seg = seg.sort_values("ts").reset_index(drop=True)
        tool = int(seg.iloc[0]["tool"])

        seg_start = seg.iloc[0]["ts"]
        seg_end = seg.iloc[-1]["ts"]

        cursor = seg_start

        while cursor <= seg_end:
            end_limit = cursor + pd.Timedelta(seconds=WINDOW_SEC)

            w = seg[
                (seg["ts"] >= cursor)
                & (seg["ts"] < end_limit)
            ].copy()

            if not w.empty:
                feat = feature_row(
                    w,
                    tool,
                    baseline_lookup,
                )

                if feat is not None:
                    windows.append(
                        {
                            "window_start": w["ts"].iloc[0],
                            "window_end": w["ts"].iloc[-1],
                            "tool": tool,
                            "sample_count": len(w),
                            **feat,
                        }
                    )

            cursor = end_limit

    return pd.DataFrame(windows)


def score_windows(
    windows,
    model,
    feature_columns,
):
    if windows.empty:
        return windows

    X = windows[feature_columns].astype(float)

    windows = windows.copy()
    windows["anomaly_score"] = (
        model.predict_proba(X)[:, 1]
    )

    return windows


def evaluate_one_site(
    sd,
    site,
    model,
    feature_columns,
    baseline_lookup,
    threshold,
):
    j0 = site["j0"]
    j1 = site["j1"]

    d = (
        sd.iloc[j0:j1]
        .copy()
        .reset_index(drop=True)
    )

    d["sload"] = site["inj"]

    windows = build_windows_for_site(
        d,
        baseline_lookup,
    )

    if windows.empty:
        return {
            "eligible": False,
            "detected": False,
            "first_detection_sec": np.nan,
            "max_score": np.nan,
            "window_count": 0,
            "supported_tools": "",
        }

    windows = score_windows(
        windows,
        model,
        feature_columns,
    )

    detected = windows[
        windows["anomaly_score"] >= threshold
    ].copy()

    first_detection_sec = np.nan

    if not detected.empty:
        first = detected.sort_values(
            "window_end"
        ).iloc[0]

        # XGBoost의 판단은 1분 Window가 끝나야 확정된다고 봄
        first_detection_sec = float(
            (
                pd.Timestamp(first["window_end"])
                - site["start"]
            ).total_seconds()
        )

    return {
        "eligible": True,
        "detected": not detected.empty,
        "first_detection_sec": first_detection_sec,
        "max_score": float(
            windows["anomaly_score"].max()
        ),
        "window_count": int(len(windows)),
        "supported_tools": ",".join(
            str(int(x))
            for x in sorted(
                windows["tool"].dropna().unique()
            )
        ),
    }


def count_xgb_real_test_episodes(
    threshold,
):
    """
    이전 단계에서 저장한 실제 5/22~31 XGBoost Window 결과로
    고득점 Episode/일을 계산.
    실제 고장 라벨이 없으므로 '정상 시험구간 알람 후보/일'로 해석한다.
    """
    path = (
        PROJECT_ROOT
        / "reports"
        / "model_comparison"
        / "supervised_test_windows.csv"
    )

    if not path.exists():
        return np.nan, np.nan

    df = pd.read_csv(path)

    if df.empty:
        return 0.0, 0

    df["window_start"] = pd.to_datetime(
        df["window_start"],
        errors="coerce",
    )
    df["window_end"] = pd.to_datetime(
        df["window_end"],
        errors="coerce",
    )

    flagged = df[
        pd.to_numeric(
            df["anomaly_score"],
            errors="coerce",
        ) >= threshold
    ].copy()

    if flagged.empty:
        return 0.0, 0

    flagged = flagged.sort_values(
        [
            "current_tool",
            "window_start",
        ]
    )

    episode_count = 0
    prev_tool = None
    prev_end = None

    for _, row in flagged.iterrows():
        tool = int(row["current_tool"])
        start = row["window_start"]
        end = row["window_end"]

        if (
            prev_end is None
            or tool != prev_tool
            or (
                start - prev_end
            ).total_seconds() > 20
        ):
            episode_count += 1

        prev_tool = tool
        prev_end = end

    # 팀원 benchmark와 동일한 달력 약 10일 분모
    days = 10.0

    return episode_count / days, episode_count


def find_m2_row(bench):
    if "채택" in bench.columns:
        rows = bench[
            pd.to_numeric(
                bench["채택"],
                errors="coerce",
            ) == 1
        ]
        if not rows.empty:
            return rows.iloc[0]

    rows = bench[
        bench["방식"].astype(str).str.contains(
            "운전조건별 비교",
            na=False,
        )
    ]

    if rows.empty:
        raise ValueError(
            "bench.csv에서 M2* 채택 행을 찾지 못했습니다."
        )

    return rows.iloc[0]



def dataframe_to_markdown_simple(df: pd.DataFrame) -> str:
    """
    pandas.to_markdown()의 tabulate 추가 의존성을 피하기 위한 간단한 Markdown 표 생성.
    """
    if df.empty:
        return "(결과 없음)"

    def fmt(value):
        if pd.isna(value):
            return "-"
        if isinstance(value, float):
            return f"{value:.1f}"
        return str(value)

    columns = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]

    for _, row in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                fmt(row[c]).replace("|", r"\|")
                for c in df.columns
            )
            + " |"
        )

    return "\n".join(lines)


def main():
    args = parse_args()

    teammate_source = resolve_path(
        args.unsupervised
    )

    if not teammate_source.exists():
        raise FileNotFoundError(
            f"M2* 경로를 찾지 못했습니다: {teammate_source}"
        )

    print("공통 전처리 M2* 경로:", teammate_source)

    sd, bench = load_teammate_data(
        teammate_source
    )

    model, feature_columns, baseline = (
        load_xgb_artifacts()
    )

    baseline_lookup = make_baseline_lookup(
        baseline
    )

    threshold, threshold_source = (
        load_xgb_threshold(
            args.threshold
        )
    )

    print(
        f"XGBoost 탐지 기준: {threshold:.6f} "
        f"({threshold_source})"
    )

    print(
        "팀원 benchmark와 같은 seed=42, "
        "유형별 60개 주입 위치를 재현합니다."
    )

    sites = sample_sites(sd)

    detail_rows = []

    for scenario, scenario_sites in sites.items():
        print(
            f"\n[{LABELS[scenario]}] "
            f"{len(scenario_sites)}개 평가"
        )

        for i, site in enumerate(
            scenario_sites,
            start=1,
        ):
            result = evaluate_one_site(
                sd=sd,
                site=site,
                model=model,
                feature_columns=feature_columns,
                baseline_lookup=baseline_lookup,
                threshold=threshold,
            )

            detail_rows.append(
                {
                    "scenario": LABELS[scenario],
                    "site_no": i,
                    "start": site["start"],
                    "end": site["end"],
                    "eligible": result["eligible"],
                    "detected": result["detected"],
                    "first_detection_sec": result["first_detection_sec"],
                    "max_score": result["max_score"],
                    "xgb_window_count": result["window_count"],
                    "supported_tools": result["supported_tools"],
                }
            )

    detail = pd.DataFrame(detail_rows)

    m2 = find_m2_row(bench)

    comparison_rows = []

    for key, _, _ in INJ_TYPES:
        label = LABELS[key]

        part = detail[
            detail["scenario"] == label
        ].copy()

        eligible = part[
            part["eligible"]
        ].copy()

        detected = eligible[
            eligible["detected"]
        ].copy()

        xgb_rate = (
            100.0
            * len(detected)
            / len(eligible)
            if len(eligible) > 0
            else np.nan
        )

        xgb_latency = (
            float(
                detected[
                    "first_detection_sec"
                ].mean()
            )
            if not detected.empty
            else np.nan
        )

        comparison_rows.append(
            {
                "이상 유형": label,
                "M2*_전체주입수": N_INJ,
                "M2*_탐지n": int(
                    m2[f"{key}_탐지n"]
                ),
                "M2*_탐지율": float(
                    m2[f"{key}_탐지율"]
                ),
                "M2*_평균탐지초": float(
                    m2[f"{key}_평균탐지초"]
                )
                if pd.notna(
                    m2[f"{key}_평균탐지초"]
                )
                else np.nan,
                "XGBoost_전체주입수": N_INJ,
                "XGBoost_분석가능n": int(
                    len(eligible)
                ),
                "XGBoost_분석불가n": int(
                    N_INJ - len(eligible)
                ),
                "XGBoost_탐지n": int(
                    len(detected)
                ),
                "XGBoost_탐지율_분석가능기준": round(
                    xgb_rate,
                    1,
                )
                if pd.notna(xgb_rate)
                else np.nan,
                "XGBoost_평균탐지초": round(
                    xgb_latency,
                    1,
                )
                if pd.notna(xgb_latency)
                else np.nan,
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    xgb_alarm_day, xgb_episode_count = (
        count_xgb_real_test_episodes(
            threshold
        )
    )

    m2_alarm_day = float(
        m2["오탐_건일"]
    )

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "model_comparison"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_path = (
        output_dir
        / "synthetic_same_scenario_detail.csv"
    )
    comparison_path = (
        output_dir
        / "synthetic_same_scenario_comparison.csv"
    )
    report_path = (
        output_dir
        / "synthetic_same_scenario_report.md"
    )

    detail.to_csv(
        detail_path,
        index=False,
        encoding="utf-8-sig",
    )

    comparison.to_csv(
        comparison_path,
        index=False,
        encoding="utf-8-sig",
    )

    lines = [
        "# XGBoost vs M2* 동일 합성 이상 시나리오 비교 (공통 전처리 최종)",
        "",
        "## 실험 조건",
        "",
        "- 팀원 benchmark.py와 동일한 시험 구간(2026-05-22~05-31)",
        "- 동일 seed=42",
        "- 동일 주입 위치, 유형별 60회",
        "- 지속 과부하: Spindle Load +12, 11분",
        "- 공구 파손형: Spindle Load 0~5로 강제, 3분",
        "- 완만한 저하: 0→+8 선형 증가, 2.3시간",
        f"- XGBoost 탐지 기준: {threshold:.6f} ({threshold_source})",
        "- M2*: cycle 기반 지속성/EWMA 알람",
        "- XGBoost: 1분 Window 기반 이상 패턴 점수",
        "",
        "## 결과",
        "",
        dataframe_to_markdown_simple(comparison),
        "",
        "## 실제 시험 로그의 알람 후보 빈도",
        "",
        f"- M2*: {m2_alarm_day:.2f}건/일",
        (
            f"- XGBoost: {xgb_alarm_day:.2f} Episode/일 "
            f"(총 {xgb_episode_count} Episode)"
            if pd.notna(xgb_alarm_day)
            else "- XGBoost: 이전 실제 시험 Window 결과 파일이 없어 계산하지 못함"
        ),
        "",
        "## 해석 시 주의",
        "",
        "- 실제 고장 정답 라벨이 없는 합성 이상 실험이다.",
        "- XGBoost 분석 불가 구간은 미탐으로 세지 않고 별도 표시했다.",
        "- M2*와 XGBoost의 시간 단위가 다르므로 평균 탐지시간을 그대로 우열로 비교하면 안 된다.",
        "- 실제 시험 로그의 알람도 진짜 오탐이라고 확정할 수 없으므로 '알람 후보 빈도'로 보는 것이 안전하다.",
        "",
    ]

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("동일 합성 이상 시나리오 비교 완료")
    print("=" * 72)
    print(
        comparison.to_string(
            index=False
        )
    )

    print()
    print(
        f"실제 시험 로그 알람 후보 빈도: "
        f"M2* {m2_alarm_day:.2f}건/일"
    )

    if pd.notna(xgb_alarm_day):
        print(
            f"실제 시험 로그 알람 후보 빈도: "
            f"XGBoost {xgb_alarm_day:.2f} Episode/일 "
            f"(총 {xgb_episode_count} Episode)"
        )

    print()
    print("저장:")
    print(detail_path)
    print(comparison_path)
    print(report_path)


if __name__ == "__main__":
    main()
