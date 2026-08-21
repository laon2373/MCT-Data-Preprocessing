# -*- coding: utf-8 -*-
"""
지도학습 XGBoost vs 비지도 M2* 비교 스크립트

목적
----
같은 실제 기업 로그의 미래 구간(기본: 2026-05-22 ~ 2026-05-31)에 대해
1) 지도학습 XGBoost가 높은 이상 패턴 점수를 준 구간
2) 팀원 비지도 M2*가 알람으로 잡은 구간
을 시간/공구 기준으로 비교합니다.

중요
----
- 실제 고장 정답 라벨이 없으므로 "정확도 비교"가 아닙니다.
- 두 모델의 탐지 결과가 얼마나 겹치는지, 어느 모델만 잡는지 보는 비교입니다.
- M2*는 지속성 기반 알람이고, XGBoost는 1분 Window 기반이므로 단순 건수만으로 우열을 판단하면 안 됩니다.

사용 예
-------
프로젝트 루트에서:

python scripts/compare_supervised_unsupervised.py --unsupervised "mct_poc_dashboard1(1).zip"

팀원 ZIP이 다른 곳에 있으면 전체 경로를 넣어도 됩니다.

출력
----
reports/model_comparison/
├─ supervised_test_windows.csv
├─ supervised_episodes_calibrated_p99.csv
├─ supervised_episodes_score090.csv
├─ unsupervised_m2_alarms_test.csv
├─ comparison_calibrated_p99.csv
├─ comparison_score090.csv
├─ comparison_daily.csv
├─ comparison_summary.csv
└─ comparison_report.md
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# =========================================================
# 프로젝트 경로
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_pipeline import run_inference


# =========================================================
# 기본 설정
# =========================================================

DEFAULT_CAL_START = "2026-05-01"
DEFAULT_CAL_END = "2026-05-21 23:59:59.999999"

DEFAULT_TEST_START = "2026-05-22"
DEFAULT_TEST_END = "2026-05-31 23:59:59.999999"

# XGBoost Window와 M2* 알람은 시간 단위가 다르므로
# 같은 공구에서 ±90초 범위 안에 겹치면 "같은 현상을 본 후보"로 취급
DEFAULT_MATCH_TOLERANCE_SECONDS = 90

# 연속된 XGBoost 고득점 Window를 하나의 Episode로 묶는 최대 간격
DEFAULT_MERGE_GAP_SECONDS = 20

# XGBoost 고정 고득점 참고 기준
VERY_HIGH_SCORE = 0.90


# =========================================================
# Helper
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="지도학습 XGBoost와 비지도 M2* 탐지 결과 비교"
    )

    parser.add_argument(
        "--unsupervised",
        type=str,
        default=None,
        help="팀원 비지도 프로젝트 ZIP 또는 dashboard 폴더 경로",
    )

    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="원본 .log 폴더",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "model_comparison"),
        help="비교 결과 저장 폴더",
    )

    parser.add_argument(
        "--cal-start",
        type=str,
        default=DEFAULT_CAL_START,
    )

    parser.add_argument(
        "--cal-end",
        type=str,
        default=DEFAULT_CAL_END,
    )

    parser.add_argument(
        "--test-start",
        type=str,
        default=DEFAULT_TEST_START,
    )

    parser.add_argument(
        "--test-end",
        type=str,
        default=DEFAULT_TEST_END,
    )

    parser.add_argument(
        "--match-tolerance-sec",
        type=int,
        default=DEFAULT_MATCH_TOLERANCE_SECONDS,
    )

    parser.add_argument(
        "--merge-gap-sec",
        type=int,
        default=DEFAULT_MERGE_GAP_SECONDS,
    )

    return parser.parse_args()


def extract_date_from_name(filename: str):
    match = re.search(r"(20\d{6})", str(filename))
    if not match:
        return None

    dt = pd.to_datetime(
        match.group(1),
        format="%Y%m%d",
        errors="coerce",
    )

    if pd.isna(dt):
        return None

    return dt.normalize()


def normalize_tool(value):
    """
    T01, Tool 1, 1 등을 모두 정수 1로 변환
    """
    if pd.isna(value):
        return np.nan

    match = re.search(r"(\d+)", str(value))
    if not match:
        return np.nan

    return int(match.group(1))


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    return value


def find_unsupervised_source(user_path: str | None):
    """
    사용자가 경로를 안 주면 프로젝트 루트에서 팀원 ZIP/폴더를 자동 탐색
    """
    if user_path:
        path = Path(user_path).expanduser()

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(
                f"비지도 프로젝트 경로를 찾지 못했습니다: {path}"
            )

        return path

    candidates = list(PROJECT_ROOT.glob("mct_poc_dashboard*.zip"))

    if candidates:
        return sorted(candidates)[0]

    for candidate in PROJECT_ROOT.iterdir():
        if candidate.is_dir():
            alarms = list(candidate.rglob("alarms.csv"))
            if alarms:
                return candidate

    raise FileNotFoundError(
        "비지도 프로젝트를 찾지 못했습니다.\n"
        '--unsupervised "mct_poc_dashboard1(1).zip" 처럼 경로를 지정해주세요.'
    )


def read_unsupervised_file(source: Path, relative_suffix: str):
    """
    ZIP/폴더 모두 지원.
    예: relative_suffix="data/alarms.csv"
    """
    relative_suffix = relative_suffix.replace("\\", "/")

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source, "r") as archive:
            matches = [
                name
                for name in archive.namelist()
                if name.replace("\\", "/").endswith(relative_suffix)
            ]

            if not matches:
                raise FileNotFoundError(
                    f"팀원 ZIP 안에서 {relative_suffix}를 찾지 못했습니다."
                )

            with archive.open(matches[0]) as f:
                return f.read()

    if source.is_dir():
        matches = list(source.rglob(Path(relative_suffix).name))

        matches = [
            path
            for path in matches
            if str(path).replace("\\", "/").endswith(relative_suffix)
        ]

        if not matches:
            raise FileNotFoundError(
                f"팀원 폴더 안에서 {relative_suffix}를 찾지 못했습니다."
            )

        return matches[0].read_bytes()

    raise ValueError(
        "비지도 프로젝트는 .zip 파일 또는 폴더여야 합니다."
    )


def load_unsupervised_alarms(source: Path):
    raw = read_unsupervised_file(
        source,
        "data/alarms.csv",
    )

    alarms = pd.read_csv(
        io.BytesIO(raw)
    )

    required = [
        "시작",
        "종료",
        "공구",
    ]

    missing = [
        column
        for column in required
        if column not in alarms.columns
    ]

    if missing:
        raise ValueError(
            "alarms.csv 필수 컬럼이 없습니다: "
            + ", ".join(missing)
        )

    alarms = alarms.copy()

    alarms["start"] = pd.to_datetime(
        alarms["시작"],
        errors="coerce",
    )

    alarms["end"] = pd.to_datetime(
        alarms["종료"],
        errors="coerce",
    )

    alarms["tool"] = alarms[
        "공구"
    ].map(
        normalize_tool
    )

    alarms = alarms.dropna(
        subset=[
            "start",
            "end",
            "tool",
        ]
    ).copy()

    alarms["tool"] = alarms[
        "tool"
    ].astype(int)

    alarms = alarms.sort_values(
        "start"
    ).reset_index(drop=True)

    alarms["m2_alarm_id"] = [
        f"M2_{index + 1:03d}"
        for index in range(len(alarms))
    ]

    return alarms


def load_unsupervised_meta(source: Path):
    try:
        raw = read_unsupervised_file(
            source,
            "data/meta.json",
        )

        return json.loads(
            raw.decode("utf-8")
        )

    except Exception:
        return {}


def find_raw_logs(
    raw_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    logs = []

    for path in sorted(
        raw_dir.glob("*.log")
    ):
        date = extract_date_from_name(
            path.name
        )

        if date is None:
            continue

        if start.normalize() <= date <= end.normalize():
            logs.append(
                (
                    date,
                    path,
                )
            )

    return logs


def run_supervised_logs(logs):
    """
    기존 검증 완료된 run_inference()를 그대로 사용.
    """
    frames = []

    total = len(logs)

    for index, (date, path) in enumerate(
        logs,
        start=1,
    ):
        print(
            f"[XGBoost {index}/{total}] {path.name}"
        )

        output = run_inference(
            path
        )

        result_df = output[
            "results"
        ].copy()

        if result_df.empty:
            continue

        if "supported" in result_df.columns:
            result_df = result_df[
                result_df["supported"]
            ].copy()

        if result_df.empty:
            continue

        result_df[
            "window_start"
        ] = pd.to_datetime(
            result_df[
                "window_start"
            ],
            errors="coerce",
        )

        result_df[
            "window_end"
        ] = pd.to_datetime(
            result_df[
                "window_end"
            ],
            errors="coerce",
        )

        result_df[
            "current_tool"
        ] = pd.to_numeric(
            result_df[
                "current_tool"
            ],
            errors="coerce",
        )

        result_df[
            "anomaly_score"
        ] = pd.to_numeric(
            result_df[
                "anomaly_score"
            ],
            errors="coerce",
        )

        result_df = result_df.dropna(
            subset=[
                "window_start",
                "window_end",
                "current_tool",
                "anomaly_score",
            ]
        ).copy()

        result_df[
            "current_tool"
        ] = result_df[
            "current_tool"
        ].astype(int)

        result_df[
            "source_file"
        ] = path.name

        result_df[
            "log_date"
        ] = date

        frames.append(
            result_df
        )

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )
        .sort_values(
            "window_start"
        )
        .reset_index(drop=True)
    )


def make_supervised_episodes(
    windows: pd.DataFrame,
    threshold: float,
    merge_gap_seconds: int,
    prefix: str,
):
    """
    threshold 이상인 연속 1분 Window를 같은 공구 기준으로 Episode로 묶음.
    """
    if windows.empty:
        return pd.DataFrame(
            columns=[
                "xgb_episode_id",
                "start",
                "end",
                "tool",
                "max_score",
                "mean_score",
                "window_count",
                "source_files",
            ]
        )

    selected = windows[
        windows[
            "anomaly_score"
        ] >= threshold
    ].copy()

    if selected.empty:
        return pd.DataFrame(
            columns=[
                "xgb_episode_id",
                "start",
                "end",
                "tool",
                "max_score",
                "mean_score",
                "window_count",
                "source_files",
            ]
        )

    selected = selected.sort_values(
        [
            "current_tool",
            "window_start",
        ]
    ).reset_index(drop=True)

    episodes = []

    current = None

    for _, row in selected.iterrows():
        start = row[
            "window_start"
        ]

        end = row[
            "window_end"
        ]

        tool = int(
            row[
                "current_tool"
            ]
        )

        score = float(
            row[
                "anomaly_score"
            ]
        )

        source_file = row.get(
            "source_file",
            "",
        )

        if current is None:
            current = {
                "start": start,
                "end": end,
                "tool": tool,
                "scores": [score],
                "source_files": {
                    source_file
                },
            }
            continue

        gap = (
            start
            - current[
                "end"
            ]
        ).total_seconds()

        same_episode = (
            tool
            == current[
                "tool"
            ]
            and gap
            <= merge_gap_seconds
        )

        if same_episode:
            current[
                "end"
            ] = max(
                current[
                    "end"
                ],
                end,
            )

            current[
                "scores"
            ].append(
                score
            )

            current[
                "source_files"
            ].add(
                source_file
            )

        else:
            episodes.append(
                current
            )

            current = {
                "start": start,
                "end": end,
                "tool": tool,
                "scores": [score],
                "source_files": {
                    source_file
                },
            }

    if current is not None:
        episodes.append(
            current
        )

    rows = []

    for index, episode in enumerate(
        episodes,
        start=1,
    ):
        rows.append(
            {
                "xgb_episode_id":
                    f"{prefix}_{index:04d}",
                "start":
                    episode[
                        "start"
                    ],
                "end":
                    episode[
                        "end"
                    ],
                "tool":
                    episode[
                        "tool"
                    ],
                "max_score":
                    max(
                        episode[
                            "scores"
                        ]
                    ),
                "mean_score":
                    float(
                        np.mean(
                            episode[
                                "scores"
                            ]
                        )
                    ),
                "window_count":
                    len(
                        episode[
                            "scores"
                        ]
                    ),
                "source_files":
                    ", ".join(
                        sorted(
                            episode[
                                "source_files"
                            ]
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def interval_matches(
    x_start,
    x_end,
    m_start,
    m_end,
    tolerance_seconds,
):
    tolerance = pd.Timedelta(
        seconds=tolerance_seconds
    )

    return (
        x_start
        <= (
            m_end
            + tolerance
        )
        and x_end
        >= (
            m_start
            - tolerance
        )
    )


def compare_episodes(
    xgb_episodes: pd.DataFrame,
    m2_alarms: pd.DataFrame,
    tolerance_seconds: int,
):
    """
    같은 공구 + 시간 겹침(± tolerance)을 기준으로 결과 비교.
    """
    rows = []

    matched_xgb = set()
    matched_m2 = set()

    for _, xgb in xgb_episodes.iterrows():
        candidates = m2_alarms[
            m2_alarms[
                "tool"
            ] == int(
                xgb[
                    "tool"
                ]
            )
        ]

        for _, m2 in candidates.iterrows():
            if interval_matches(
                xgb[
                    "start"
                ],
                xgb[
                    "end"
                ],
                m2[
                    "start"
                ],
                m2[
                    "end"
                ],
                tolerance_seconds,
            ):
                matched_xgb.add(
                    xgb[
                        "xgb_episode_id"
                    ]
                )

                matched_m2.add(
                    m2[
                        "m2_alarm_id"
                    ]
                )

                rows.append(
                    {
                        "comparison":
                            "둘 다 탐지",
                        "tool":
                            int(
                                xgb[
                                    "tool"
                                ]
                            ),
                        "xgb_episode_id":
                            xgb[
                                "xgb_episode_id"
                            ],
                        "xgb_start":
                            xgb[
                                "start"
                            ],
                        "xgb_end":
                            xgb[
                                "end"
                            ],
                        "xgb_max_score":
                            xgb[
                                "max_score"
                            ],
                        "m2_alarm_id":
                            m2[
                                "m2_alarm_id"
                            ],
                        "m2_start":
                            m2[
                                "start"
                            ],
                        "m2_end":
                            m2[
                                "end"
                            ],
                        "m2_channel":
                            m2.get(
                                "채널",
                                "",
                            ),
                        "m2_description":
                            m2.get(
                                "설명",
                                "",
                            ),
                    }
                )

    for _, xgb in xgb_episodes.iterrows():
        if (
            xgb[
                "xgb_episode_id"
            ]
            not in matched_xgb
        ):
            rows.append(
                {
                    "comparison":
                        "XGBoost만 탐지",
                    "tool":
                        int(
                            xgb[
                                "tool"
                            ]
                        ),
                    "xgb_episode_id":
                        xgb[
                            "xgb_episode_id"
                        ],
                    "xgb_start":
                        xgb[
                            "start"
                        ],
                    "xgb_end":
                        xgb[
                            "end"
                        ],
                    "xgb_max_score":
                        xgb[
                            "max_score"
                        ],
                    "m2_alarm_id":
                        "",
                    "m2_start":
                        pd.NaT,
                    "m2_end":
                        pd.NaT,
                    "m2_channel":
                        "",
                    "m2_description":
                        "",
                }
            )

    for _, m2 in m2_alarms.iterrows():
        if (
            m2[
                "m2_alarm_id"
            ]
            not in matched_m2
        ):
            rows.append(
                {
                    "comparison":
                        "M2*만 탐지",
                    "tool":
                        int(
                            m2[
                                "tool"
                            ]
                        ),
                    "xgb_episode_id":
                        "",
                    "xgb_start":
                        pd.NaT,
                    "xgb_end":
                        pd.NaT,
                    "xgb_max_score":
                        np.nan,
                    "m2_alarm_id":
                        m2[
                            "m2_alarm_id"
                        ],
                    "m2_start":
                        m2[
                            "start"
                        ],
                    "m2_end":
                        m2[
                            "end"
                        ],
                    "m2_channel":
                        m2.get(
                            "채널",
                            "",
                        ),
                    "m2_description":
                        m2.get(
                            "설명",
                            "",
                        ),
                }
            )

    comparison = pd.DataFrame(
        rows
    )

    return (
        comparison,
        matched_xgb,
        matched_m2,
    )


def make_daily_comparison(
    test_windows,
    xgb_episodes,
    m2_alarms,
):
    dates = pd.date_range(
        DEFAULT_TEST_START,
        "2026-05-31",
        freq="D",
    )

    rows = []

    for date in dates:
        next_date = (
            date
            + pd.Timedelta(
                days=1
            )
        )

        day_windows = test_windows[
            (
                test_windows[
                    "window_start"
                ] >= date
            )
            & (
                test_windows[
                    "window_start"
                ] < next_date
            )
        ]

        day_xgb = xgb_episodes[
            (
                xgb_episodes[
                    "start"
                ] >= date
            )
            & (
                xgb_episodes[
                    "start"
                ] < next_date
            )
        ]

        day_m2 = m2_alarms[
            (
                m2_alarms[
                    "start"
                ] >= date
            )
            & (
                m2_alarms[
                    "start"
                ] < next_date
            )
        ]

        rows.append(
            {
                "date":
                    date.date(),
                "xgb_window_count":
                    len(
                        day_windows
                    ),
                "xgb_daily_max_score":
                    (
                        float(
                            day_windows[
                                "anomaly_score"
                            ].max()
                        )
                        if not day_windows.empty
                        else np.nan
                    ),
                "xgb_episode_count":
                    len(
                        day_xgb
                    ),
                "m2_alarm_count":
                    len(
                        day_m2
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def save_csv(
    df,
    path,
):
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def make_report(
    output_dir,
    meta,
    cal_threshold,
    test_windows,
    xgb_cal_episodes,
    xgb_090_episodes,
    m2_test,
    compare_cal,
    compare_090,
    matched_xgb_cal,
    matched_m2_cal,
    matched_xgb_090,
    matched_m2_090,
):
    def percent(
        numerator,
        denominator,
    ):
        if denominator == 0:
            return 0.0

        return (
            numerator
            / denominator
            * 100
        )

    train_range = meta.get(
        "train_range",
        "5/1~5/21",
    )

    lines = [
        "# 지도학습 XGBoost vs 비지도 M2* 비교",
        "",
        "## 비교 목적",
        "",
        "- 실제 고장 정답 라벨이 없기 때문에 이 비교는 **정확도 대결이 아닙니다.**",
        "- 같은 미래 실제 로그(5/22~5/31)에 두 방식이 어떤 구간을 탐지하는지 비교합니다.",
        "- M2*는 실제 정상 데이터로 정상 기준을 학습하고, XGBoost는 정상+가상 이상 라벨로 학습합니다.",
        "",
        "## 비교 조건",
        "",
        f"- M2* 정상 기준 학습 기간: `{train_range}`",
        "- 비교 시험 기간: `2026-05-22 ~ 2026-05-31`",
        f"- XGBoost 비교용 P99 기준: `{cal_threshold:.6f}`",
        "  - 5/1~5/21 실제 로그 XGBoost Score의 99분위에서 별도로 계산",
        f"- XGBoost 고정 고득점 기준: `{VERY_HIGH_SCORE:.2f}`",
        f"- 시간 매칭 허용 범위: `±{DEFAULT_MATCH_TOLERANCE_SECONDS}초`",
        "- 같은 공구이면서 시간 범위가 겹치면 '둘 다 탐지' 후보로 분류",
        "",
        "## 실제 시험기간 결과",
        "",
        f"- XGBoost 분석 가능 1분 Window: **{len(test_windows):,}개**",
        f"- M2* 실제 시험기간 알람 Episode: **{len(m2_test):,}건**",
        "",
        "### 1) XGBoost 5/1~21 P99 보정 기준",
        "",
        f"- XGBoost Episode: **{len(xgb_cal_episodes):,}건**",
        f"- XGBoost Episode 중 M2*와 겹침: **{len(matched_xgb_cal):,}건 ({percent(len(matched_xgb_cal), len(xgb_cal_episodes)):.1f}%)**",
        f"- M2* 알람 중 XGBoost와 겹침: **{len(matched_m2_cal):,}건 ({percent(len(matched_m2_cal), len(m2_test)):.1f}%)**",
        "",
        "### 2) XGBoost Score >= 0.9 기준",
        "",
        f"- XGBoost Episode: **{len(xgb_090_episodes):,}건**",
        f"- XGBoost Episode 중 M2*와 겹침: **{len(matched_xgb_090):,}건 ({percent(len(matched_xgb_090), len(xgb_090_episodes)):.1f}%)**",
        f"- M2* 알람 중 XGBoost와 겹침: **{len(matched_m2_090):,}건 ({percent(len(matched_m2_090), len(m2_test)):.1f}%)**",
        "",
        "## 이 결과를 어떻게 해석해야 하나?",
        "",
        "1. **둘 다 탐지**",
        "   - 서로 학습 방식이 다른데도 같은 시간대를 잡았다면 우선 검토 가치가 높은 후보입니다.",
        "2. **XGBoost만 탐지**",
        "   - XGBoost가 학습한 LOAD 증가 / RPM 불안정 / RPM 오차 증가 같은 가상 이상 패턴과 유사했을 가능성이 있습니다.",
        "   - 공구 교체·공정 전환 같은 정상 변화에 민감했는지도 확인해야 합니다.",
        "3. **M2*만 탐지**",
        "   - 실제 정상 운전 조건에서 지속적으로 벗어난 패턴이지만 XGBoost의 가상 이상 정의와는 달랐을 가능성이 있습니다.",
        "",
        "## 중요한 한계",
        "",
        "- 실제 고장 라벨이 없으므로 '둘 다 탐지 = 진짜 고장'이 아닙니다.",
        "- M2*는 지속성(EWMA/연속 초과)을 반영한 Episode이고, XGBoost는 1분 Window 분류이므로 탐지 건수를 1:1 성능처럼 비교하면 안 됩니다.",
        "- 현재 XGBoost 자체의 학습용 가상 이상 Seed가 5/22 이후 실제 정상 데이터에서 만들어졌는지 여부까지 완전히 동일 조건으로 통제한 비교는 아닙니다.",
        "- 따라서 이번 결과는 **실제 로그에서 두 방식의 탐지 성향을 비교하는 실험**으로 사용하는 것이 정확합니다.",
        "",
        "## 다음 분석",
        "",
        "- `comparison_calibrated_p99.csv`에서 **둘 다 탐지 / XGBoost만 / M2*만** 대표 구간을 3~5개씩 선택",
        "- 원본 로그에서 공구 교체·공정 전환·절삭 종료 여부 확인",
        "- 두 모델 중 어느 쪽이 정상적인 작업 변화를 더 많이 이상으로 판단하는지 정리",
        "- 그 결과를 바탕으로 기업 발표에서 최종 권장 방식을 결정",
        "",
    ]

    report_path = (
        output_dir
        / "comparison_report.md"
    )

    report_path.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )

    return report_path


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()

    cal_start = pd.Timestamp(
        args.cal_start
    )

    cal_end = pd.Timestamp(
        args.cal_end
    )

    test_start = pd.Timestamp(
        args.test_start
    )

    test_end = pd.Timestamp(
        args.test_end
    )

    raw_dir = Path(
        args.raw_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    unsup_source = (
        find_unsupervised_source(
            args.unsupervised
        )
    )

    print(
        "비지도 프로젝트:",
        unsup_source,
    )

    print(
        "원본 로그 폴더:",
        raw_dir,
    )

    # -----------------------------------------------------
    # 1. M2* 알람
    # -----------------------------------------------------

    m2_all = (
        load_unsupervised_alarms(
            unsup_source
        )
    )

    meta = (
        load_unsupervised_meta(
            unsup_source
        )
    )

    m2_test = m2_all[
        (
            m2_all[
                "start"
            ] >= test_start
        )
        & (
            m2_all[
                "start"
            ] <= test_end
        )
    ].copy()

    print(
        "M2* 시험기간 알람:",
        len(
            m2_test
        ),
    )

    # -----------------------------------------------------
    # 2. XGBoost calibration: 5/1~21 실제 로그
    # -----------------------------------------------------

    cal_logs = find_raw_logs(
        raw_dir,
        cal_start,
        cal_end,
    )

    if not cal_logs:
        raise FileNotFoundError(
            "5/1~5/21 calibration 로그를 찾지 못했습니다."
        )

    print(
        "XGBoost calibration 로그:",
        len(
            cal_logs
        ),
    )

    cal_windows = (
        run_supervised_logs(
            cal_logs
        )
    )

    if cal_windows.empty:
        raise RuntimeError(
            "Calibration 구간에서 XGBoost Score를 계산하지 못했습니다."
        )

    calibrated_p99 = float(
        cal_windows[
            "anomaly_score"
        ].quantile(
            0.99
        )
    )

    print(
        f"XGBoost 5/1~21 P99 = {calibrated_p99:.6f}"
    )

    # -----------------------------------------------------
    # 3. XGBoost test: 5/22~31 실제 로그
    # -----------------------------------------------------

    test_logs = find_raw_logs(
        raw_dir,
        test_start,
        test_end,
    )

    if not test_logs:
        raise FileNotFoundError(
            "5/22~5/31 시험 로그를 찾지 못했습니다."
        )

    print(
        "XGBoost 시험 로그:",
        len(
            test_logs
        ),
    )

    test_windows = (
        run_supervised_logs(
            test_logs
        )
    )

    if test_windows.empty:
        raise RuntimeError(
            "시험기간 XGBoost 분석 결과가 없습니다."
        )

    # -----------------------------------------------------
    # 4. XGBoost Episode 생성
    # -----------------------------------------------------

    xgb_cal_episodes = (
        make_supervised_episodes(
            test_windows,
            threshold=
                calibrated_p99,
            merge_gap_seconds=
                args.merge_gap_sec,
            prefix=
                "XGB_P99",
        )
    )

    xgb_090_episodes = (
        make_supervised_episodes(
            test_windows,
            threshold=
                VERY_HIGH_SCORE,
            merge_gap_seconds=
                args.merge_gap_sec,
            prefix=
                "XGB_090",
        )
    )

    # -----------------------------------------------------
    # 5. 같은 시간/공구 비교
    # -----------------------------------------------------

    (
        compare_cal,
        matched_xgb_cal,
        matched_m2_cal,
    ) = compare_episodes(
        xgb_cal_episodes,
        m2_test,
        tolerance_seconds=
            args.match_tolerance_sec,
    )

    (
        compare_090,
        matched_xgb_090,
        matched_m2_090,
    ) = compare_episodes(
        xgb_090_episodes,
        m2_test,
        tolerance_seconds=
            args.match_tolerance_sec,
    )

    daily = (
        make_daily_comparison(
            test_windows,
            xgb_cal_episodes,
            m2_test,
        )
    )

    # -----------------------------------------------------
    # 6. Summary CSV
    # -----------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "비교 기준":
                    "XGBoost calibration P99",
                "XGBoost 기준":
                    calibrated_p99,
                "XGBoost Episode":
                    len(
                        xgb_cal_episodes
                    ),
                "M2* Episode":
                    len(
                        m2_test
                    ),
                "XGBoost 중 겹침":
                    len(
                        matched_xgb_cal
                    ),
                "M2* 중 겹침":
                    len(
                        matched_m2_cal
                    ),
            },
            {
                "비교 기준":
                    "XGBoost Score >= 0.9",
                "XGBoost 기준":
                    VERY_HIGH_SCORE,
                "XGBoost Episode":
                    len(
                        xgb_090_episodes
                    ),
                "M2* Episode":
                    len(
                        m2_test
                    ),
                "XGBoost 중 겹침":
                    len(
                        matched_xgb_090
                    ),
                "M2* 중 겹침":
                    len(
                        matched_m2_090
                    ),
            },
        ]
    )

    # -----------------------------------------------------
    # 7. 저장
    # -----------------------------------------------------

    save_csv(
        test_windows,
        output_dir
        / "supervised_test_windows.csv",
    )

    save_csv(
        xgb_cal_episodes,
        output_dir
        / "supervised_episodes_calibrated_p99.csv",
    )

    save_csv(
        xgb_090_episodes,
        output_dir
        / "supervised_episodes_score090.csv",
    )

    save_csv(
        m2_test,
        output_dir
        / "unsupervised_m2_alarms_test.csv",
    )

    save_csv(
        compare_cal,
        output_dir
        / "comparison_calibrated_p99.csv",
    )

    save_csv(
        compare_090,
        output_dir
        / "comparison_score090.csv",
    )

    save_csv(
        daily,
        output_dir
        / "comparison_daily.csv",
    )

    save_csv(
        summary,
        output_dir
        / "comparison_summary.csv",
    )

    report_path = make_report(
        output_dir=
            output_dir,
        meta=
            meta,
        cal_threshold=
            calibrated_p99,
        test_windows=
            test_windows,
        xgb_cal_episodes=
            xgb_cal_episodes,
        xgb_090_episodes=
            xgb_090_episodes,
        m2_test=
            m2_test,
        compare_cal=
            compare_cal,
        compare_090=
            compare_090,
        matched_xgb_cal=
            matched_xgb_cal,
        matched_m2_cal=
            matched_m2_cal,
        matched_xgb_090=
            matched_xgb_090,
        matched_m2_090=
            matched_m2_090,
    )

    print()
    print("=" * 60)
    print("비교 완료")
    print("=" * 60)
    print(
        f"시험 XGBoost Window: {len(test_windows):,}"
    )
    print(
        f"M2* 시험기간 알람: {len(m2_test):,}"
    )
    print(
        f"XGBoost calibration P99: {calibrated_p99:.6f}"
    )
    print(
        f"P99 XGBoost Episode: {len(xgb_cal_episodes):,}"
    )
    print(
        f"P99 기준 XGBoost/M2* 겹침: "
        f"{len(matched_xgb_cal):,} / {len(matched_m2_cal):,}"
    )
    print(
        f"Score>=0.9 XGBoost Episode: {len(xgb_090_episodes):,}"
    )
    print(
        f"0.9 기준 XGBoost/M2* 겹침: "
        f"{len(matched_xgb_090):,} / {len(matched_m2_090):,}"
    )
    print()
    print(
        "결과 폴더:",
        output_dir,
    )
    print(
        "요약 보고서:",
        report_path,
    )
    print()
    print(
        "다음에는 comparison_calibrated_p99.csv의 "
        "'둘 다 탐지 / XGBoost만 탐지 / M2*만 탐지'를 확인하면 됩니다."
    )


if __name__ == "__main__":
    main()
