# -*- coding: utf-8 -*-
"""
MCT 3호 지도학습 이상 패턴 분석 대시보드

지원:
- 단일 .log 분석
- 여러 .log가 들어 있는 .zip 일괄 분석
- XGBoost Anomaly Score 추이
- 관찰 후보 구간 / 상세 해석
- 업로드 데이터 요약
- PDF 요약 보고서 다운로드

주의:
- Anomaly Score는 실제 고장 확률이 아니다.
- P95/P99는 실제 2026년 5월 로그 분포의 참고값이지 고장 판정 임계값이 아니다.
"""

from __future__ import annotations

import hashlib
import html
import io
import math
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def render_xgb_view():
    """통합 대시보드 안에서 기존 XGBoost 화면을 렌더링한다."""
    st.markdown("##### 지도학습 XGBoost 분석")
    st.caption(
        "단일 `.log` 또는 여러 날짜의 `.log`가 들어 있는 `.zip`을 업로드하면 "
        "저장된 XGBoost 모델로 1분 절삭 Window의 이상 패턴을 분석합니다."
    )

    # =========================================================
    # 기본 설정
    # =========================================================


    API_URL = "http://127.0.0.1:8000/predict"

    # 실제 2026년 5월 31일 로그에서 얻은 Anomaly Score 참고 분포
    # 고장 임계값이 아님.
    REFERENCE_P95 = 0.018857
    REFERENCE_P99 = 0.136186

    MAX_ZIP_FILES = 100
    MAX_ZIP_UNCOMPRESSED_BYTES = 1_500_000_000  # 약 1.5GB


    # =========================================================
    # 디자인 토큰
    # =========================================================

    # 통합 대시보드의 M2* 화면과 같은 팔레트 사용
    T = {
        "blue": "#3987e5",
        "aqua": "#199e70",
        "orange": "#d95926",
        "red": "#d03b3b",
        "green": "#0ca30c",
        "yellow": "#fab219",
        "ink": "#ffffff",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#232322",
        "axis": "#383835",
        "surface": "#1a1a19",
        "surface2": "#1a1a19",
        "border": "rgba(255,255,255,.08)",
    }

    FONT = '"Malgun Gothic", "Segoe UI", system-ui, sans-serif'
    PLOT_CONFIG = {
        "displayModeBar": False,
        "responsive": True,
    }


    # =========================================================
    # CSS
    # =========================================================

    st.markdown(
        f"""
        <style>
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1240px;
        }}

        h1, h2, h3, h4 {{
            letter-spacing: -0.02em;
        }}

        .hero-title {{
            font-size: 1.85rem;
            font-weight: 800;
            color: {T["ink"]};
            margin-bottom: .35rem;
        }}

        .hero-sub {{
            color: {T["ink2"]};
            font-size: .92rem;
            line-height: 1.65;
            margin-bottom: .65rem;
        }}

        .chips {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin: 4px 0 12px;
        }}

        .chip {{
            font-size: .79rem;
            color: {T["ink2"]};
            background: {T["surface"]};
            border: 1px solid {T["border"]};
            border-radius: 999px;
            padding: 4px 11px;
        }}

        .section-kicker {{
            color: {T["blue"]};
            font-weight: 800;
            font-size: .70rem;
            letter-spacing: .07em;
            margin-top: 1.2rem;
            margin-bottom: .2rem;
        }}

        .section-title {{
            color: {T["ink"]};
            font-size: 1.25rem;
            font-weight: 800;
            margin-bottom: .25rem;
        }}

        .section-sub {{
            color: {T["muted"]};
            font-size: .84rem;
            line-height: 1.55;
            margin-bottom: .75rem;
        }}

        .kpi-row {{
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin: 5px 0 8px;
        }}

        .kpi {{
            flex: 1 1 0;
            min-width: 118px;
            background: {T["surface"]};
            border: 1px solid {T["border"]};
            border-radius: 12px;
            padding: 12px 14px 10px;
        }}

        .kpi-label {{
            font-size: .79rem;
            color: {T["ink2"]};
            margin-bottom: 5px;
        }}

        .kpi-value {{
            font-size: 1.62rem;
            font-weight: 600;
            color: {T["ink"]};
            line-height: 1.2;
        }}

        .kpi-sub {{
            font-size: .72rem;
            color: {T["muted"]};
            margin-top: 5px;
            line-height: 1.45;
        }}

        .dot {{
            display: inline-block;
            width: 10px;
            height: 10px;
            margin-right: 7px;
            border-radius: 50%;
            vertical-align: 1px;
        }}

        .panel {{
            background: {T["surface"]};
            border: 1px solid {T["border"]};
            border-radius: 12px;
            padding: 14px 16px;
            line-height: 1.7;
            margin-bottom: 10px;
        }}

        .panel-title {{
            font-size: .92rem;
            font-weight: 600;
            color: {T["ink"]};
            margin-bottom: 6px;
        }}

        .panel-text {{
            color: {T["ink2"]};
            font-size: .84rem;
        }}

        .warning-panel {{
            background: rgba(217,119,6,.08);
            border: 1px solid rgba(217,119,6,.3);
            border-radius: 12px;
            padding: 14px 16px;
            line-height: 1.7;
            margin-bottom: 10px;
        }}

        .good-panel {{
            background: rgba(39,179,106,.07);
            border: 1px solid rgba(39,179,106,.28);
            border-radius: 12px;
            padding: 14px 16px;
            line-height: 1.7;
            margin-bottom: 10px;
        }}

        .small-note {{
            color: {T["muted"]};
            font-size: .74rem;
            line-height: 1.5;
        }}

        .pdf-note {{
            background: rgba(57,135,229,.07);
            border: 1px solid rgba(57,135,229,.25);
            border-radius: 10px;
            padding: 10px 12px;
            color: {T["ink2"]};
            font-size: .78rem;
            line-height: 1.55;
        }}

        [data-testid="stDataFrame"] * {{
            font-variant-numeric: tabular-nums;
        }}

        div[data-testid="stExpander"] {{
            border: 1px solid {T["border"]};
            border-radius: 10px;
        }}

        [data-testid="stFileUploader"] section {{
            border-radius: 12px;
        }}

        </style>
        """,
        unsafe_allow_html=True,
    )


    # =========================================================
    # 공통 Helper
    # =========================================================

    FEATURE_NAME_MAP = {
        "rpm_mean": "RPM 평균",
        "rpm_std": "RPM 변동성",
        "rpm_min": "RPM 최솟값",
        "rpm_max": "RPM 최댓값",
        "rpm_range": "RPM 변화 폭",
        "load_mean": "Spindle Load 평균",
        "load_std": "Spindle Load 변동성",
        "load_max": "Spindle Load 최댓값",
        "rpm_error_mean": "RPM 오차 평균",
        "rpm_error_std": "RPM 오차 변동성",
        "rpm_error_absmax": "RPM 오차 최대 절댓값",
        "load_vs_normal": "정상 대비 Spindle Load",
        "rpm_std_vs_normal": "정상 대비 RPM 변동",
        "rpm_error_vs_normal": "정상 대비 RPM 오차",
    }


    def safe_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(value):
            return None

        return value


    def format_dt(value, with_date=True):
        dt = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(dt):
            return "-"

        return (
            dt.strftime("%Y-%m-%d %H:%M:%S")
            if with_date
            else dt.strftime("%H:%M:%S")
        )


    def extract_log_date(filename):
        """
        Data.log_20260531.log처럼 파일명에 YYYYMMDD가 있으면 날짜를 추출한다.
        절삭 데이터가 0건인 날도 '업로드 데이터 기간'에는 포함하기 위함이다.
        """
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

        return dt


    def format_date_only(value):
        dt = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(dt):
            return "-"

        return dt.strftime("%Y-%m-%d")


    def analysis_period_text(result):
        """
        실제 절삭/AI Window가 존재한 기간.
        """
        start = result.get("file_start")
        end = result.get("file_end")

        if not start or not end:
            return "분석 가능한 절삭 구간 없음"

        return (
            f"{format_dt(start)} - "
            f"{format_dt(end)}"
        )


    def uploaded_period_text(result):
        """
        사용자가 업로드한 로그 파일 자체의 기간.
        """
        start = result.get("upload_start_date")
        end = result.get("upload_end_date")

        if not start and not end:
            return analysis_period_text(result)

        if start == end:
            return format_date_only(start)

        return (
            f"{format_date_only(start)} - "
            f"{format_date_only(end)}"
        )


    def score_level(score):
        if score is None:
            return "결과 없음", T["muted"]

        if score >= REFERENCE_P99:
            return "높은 이상 유사도", T["red"]

        if score >= REFERENCE_P95:
            return "관찰 필요", T["yellow"]

        return "낮은 이상 유사도", T["green"]


    def status_label(score):
        """사용자 화면용 쉬운 AI 분석 상태."""
        score = safe_float(score)

        if score is None:
            return "분석 결과 없음"
        if score >= REFERENCE_P99:
            return "우선 확인"
        if score >= REFERENCE_P95:
            return "관찰"
        return "일반 범위"


    def percent_change_text(ratio):
        ratio = safe_float(ratio)
        if ratio is None:
            return "-"

        change = (ratio - 1.0) * 100
        sign = "+" if change >= 0 else ""
        return f"{sign}{change:.1f}%"


    def multiple_text(value):
        value = safe_float(value)
        if value is None:
            return "-"
        return f"{value:.1f}배"


    def selected_window_explanation(row):
        """선택한 구간을 현장 사용자가 읽기 쉬운 문장으로 설명."""
        lines = []

        load_ratio = safe_float(row.get("load_vs_normal"))
        rpm_std = safe_float(row.get("rpm_std_vs_normal"))
        rpm_error = safe_float(row.get("rpm_error_vs_normal"))

        if load_ratio is not None:
            change = (load_ratio - 1) * 100
            if change >= 10:
                lines.append(f"Spindle 부하가 이 공구의 평소 기준보다 약 {change:.1f}% 높았습니다.")
            elif change >= 2:
                lines.append(f"Spindle 부하가 평소보다 약 {change:.1f}% 조금 높았습니다.")
            elif change <= -10:
                lines.append(f"Spindle 부하가 평소보다 약 {abs(change):.1f}% 낮았습니다.")
            else:
                lines.append("Spindle 부하는 이 공구의 평소 수준과 큰 차이가 없었습니다.")

        if rpm_std is not None:
            if rpm_std >= 2:
                lines.append(f"RPM 흔들림이 평소보다 약 {rpm_std:.1f}배 크게 나타났습니다.")
            elif rpm_std >= 1:
                lines.append("RPM 흔들림이 평소보다 다소 크게 나타났습니다.")
            else:
                lines.append("RPM 흔들림은 평소보다 크게 나타나지 않았습니다.")

        if rpm_error is not None:
            if rpm_error >= 1:
                lines.append(f"명령 RPM과 실제 RPM의 차이가 평소 기준보다 약 {rpm_error:.1f}배 크게 나타났습니다.")
            else:
                lines.append("명령 RPM과 실제 RPM의 차이는 공구별 평소 기준 안쪽이었습니다.")

        return lines


    def kpi_row(items):
        tiles = []

        for item in items:
            dot = (
                f'<span class="dot" style="background:{item["dot"]}"></span>'
                if item.get("dot")
                else ""
            )

            sub = (
                f'<div class="kpi-sub">{html.escape(str(item["sub"]))}</div>'
                if item.get("sub")
                else ""
            )

            tiles.append(
                '<div class="kpi">'
                f'<div class="kpi-label">{html.escape(str(item["label"]))}</div>'
                f'<div class="kpi-value">{dot}{html.escape(str(item["value"]))}</div>'
                f'{sub}'
                '</div>'
            )

        st.markdown(
            f'<div class="kpi-row">{"".join(tiles)}</div>',
            unsafe_allow_html=True,
        )


    def section(kicker, title, subtitle=""):
        st.markdown(
            f"""
            <div class="section-kicker">{html.escape(kicker)}</div>
            <div class="section-title">{html.escape(title)}</div>
            <div class="section-sub">{html.escape(subtitle)}</div>
            """,
            unsafe_allow_html=True,
        )


    def style_figure(fig, height=420, legend=True):
        fig.update_layout(
            height=height,
            margin=dict(l=10, r=12, t=48, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family=FONT,
                size=12,
                color=T["ink2"],
            ),
            hovermode="x unified",
            showlegend=legend,
            legend=dict(
                orientation="h",
                x=1,
                xanchor="right",
                y=1.02,
                yanchor="bottom",
                bgcolor="rgba(0,0,0,0)",
            ),
        )

        fig.update_xaxes(
            gridcolor=T["grid"],
            linecolor=T["axis"],
            zeroline=False,
        )

        fig.update_yaxes(
            gridcolor=T["grid"],
            linecolor=T["axis"],
            zeroline=False,
        )

        return fig


    def normalize_single_result(result):
        """
        단일 API 응답을 ZIP 집계와 동일한 구조로 맞춘다.
        """
        result = dict(result)

        filename = result.get(
            "filename",
            "uploaded.log",
        )

        result["analysis_mode"] = "single"
        result["file_count"] = 1
        result["filenames"] = [filename]

        uploaded_date = extract_log_date(filename)

        result["upload_start_date"] = (
            str(uploaded_date.date())
            if uploaded_date is not None
            else None
        )
        result["upload_end_date"] = (
            str(uploaded_date.date())
            if uploaded_date is not None
            else None
        )
        result["daily_results"] = [
            {
                "filename": filename,
                "score_max": result.get("score_max"),
                "score_mean": result.get("score_mean"),
                "highest_window": result.get("highest_window"),
                "top_contributions": result.get("top_contributions", []),
                "clean_rows": result.get("clean_rows", 0),
                "cutting_rows": result.get("cutting_rows", 0),
                "supported_windows": result.get("supported_windows", 0),
                "tool_summary": result.get("tool_summary", []),
                "cutting_segments": result.get("cutting_segments", []),
            }
        ]

        # 모든 Score 포인트에 파일명 부여
        score_series = []

        for item in result.get(
            "score_series",
            [],
        ):
            copied = dict(item)
            copied["source_file"] = filename
            score_series.append(copied)

        result["score_series"] = score_series

        top_windows = []

        for item in result.get(
            "top_windows",
            [],
        ):
            copied = dict(item)
            copied["source_file"] = filename
            top_windows.append(copied)

        result["top_windows"] = top_windows

        window_details = []

        for item in result.get(
            "window_details",
            [],
        ):
            copied = dict(item)
            copied["source_file"] = filename
            window_details.append(copied)

        result["window_details"] = window_details

        return result


    def aggregate_zip_results(
        results,
    ):
        if not results:
            raise ValueError(
                "집계할 분석 결과가 없습니다."
            )

        all_scores = []
        all_top_windows = []
        all_window_details = []
        all_segments = []
        daily_results = []

        used_tools_set = set()
        supported_tools_set = set()
        unsupported_tools_set = set()
        tool_analysis_rows = defaultdict(
            lambda: {
                "analyzed_windows": 0,
                "baseline_missing_windows": 0,
                "insufficient_windows": 0,
            }
        )
        tool_rows = defaultdict(
            lambda: {
                "cutting_rows": 0,
                "rpm_weighted": 0.0,
                "rpm_weight": 0.0,
                "load_weighted": 0.0,
                "load_weight": 0.0,
            }
        )

        total_clean = 0
        total_cutting = 0
        total_supported = 0
        total_unsupported = 0
        total_insufficient = 0
        weighted_score_sum = 0.0
        weighted_score_n = 0

        interval_weighted = 0.0
        interval_weight = 0.0

        file_starts = []
        file_ends = []
        uploaded_dates = []

        global_highest = None
        global_highest_score = -1.0
        global_contributions = []

        for result in results:
            filename = result.get(
                "filename",
                "-"
            )

            uploaded_date = extract_log_date(filename)
            if uploaded_date is not None:
                uploaded_dates.append(uploaded_date)

            clean_rows = int(
                result.get(
                    "clean_rows",
                    0,
                )
            )

            cutting_rows = int(
                result.get(
                    "cutting_rows",
                    0,
                )
            )

            supported = int(
                result.get(
                    "supported_windows",
                    0,
                )
            )

            total_clean += clean_rows
            total_cutting += cutting_rows
            total_supported += supported
            total_unsupported += int(
                result.get(
                    "unsupported_windows",
                    0,
                )
            )
            total_insufficient += int(
                result.get(
                    "insufficient_windows",
                    0,
                )
            )

            score_mean = safe_float(
                result.get(
                    "score_mean"
                )
            )

            if (
                score_mean is not None
                and supported > 0
            ):
                weighted_score_sum += (
                    score_mean * supported
                )
                weighted_score_n += supported

            interval_mean = safe_float(
                result.get(
                    "sampling_interval_mean"
                )
            )

            if (
                interval_mean is not None
                and cutting_rows > 0
            ):
                interval_weighted += (
                    interval_mean
                    * cutting_rows
                )
                interval_weight += cutting_rows

            if result.get("file_start"):
                file_starts.append(
                    pd.to_datetime(
                        result["file_start"],
                        errors="coerce",
                    )
                )

            if result.get("file_end"):
                file_ends.append(
                    pd.to_datetime(
                        result["file_end"],
                        errors="coerce",
                    )
                )

            daily_results.append(
                {
                    "filename": filename,
                    "score_max": result.get(
                        "score_max"
                    ),
                    "score_mean": result.get(
                        "score_mean"
                    ),
                    "highest_window": result.get(
                        "highest_window"
                    ),
                    "top_contributions": result.get(
                        "top_contributions",
                        [],
                    ),
                    "clean_rows": clean_rows,
                    "cutting_rows": cutting_rows,
                    "supported_windows": supported,
                    "tool_summary": result.get(
                        "tool_summary",
                        [],
                    ),
                    "cutting_segments": result.get(
                        "cutting_segments",
                        [],
                    ),
                }
            )

            # 전체 Score 시계열
            for item in result.get(
                "score_series",
                [],
            ):
                copied = dict(item)
                copied[
                    "source_file"
                ] = filename
                all_scores.append(
                    copied
                )

            # 모든 AI 분석 완료 Window를 월간 상세 조회용으로 합친다.
            for item in result.get(
                "window_details",
                [],
            ):
                copied = dict(item)
                copied["source_file"] = filename
                all_window_details.append(copied)

            used_tools_set.update(
                int(tool)
                for tool in result.get(
                    "used_tools",
                    [],
                )
            )

            supported_tools_set.update(
                int(tool)
                for tool in result.get(
                    "supported_tools",
                    [],
                )
            )

            unsupported_tools_set.update(
                int(tool)
                for tool in result.get(
                    "unsupported_tools",
                    [],
                )
            )

            for item in result.get(
                "tool_analysis_summary",
                [],
            ):
                tool = int(item["tool"])
                agg_tool = tool_analysis_rows[tool]
                agg_tool["analyzed_windows"] += int(item.get("analyzed_windows", 0))
                agg_tool["baseline_missing_windows"] += int(item.get("baseline_missing_windows", 0))
                agg_tool["insufficient_windows"] += int(item.get("insufficient_windows", 0))

            # 일별 Top10을 모으면 전체 Top10을 얻기에 충분하다.
            for item in result.get(
                "top_windows",
                [],
            ):
                copied = dict(item)
                copied[
                    "source_file"
                ] = filename
                all_top_windows.append(
                    copied
                )

            # Segment
            for item in result.get(
                "cutting_segments",
                [],
            ):
                copied = dict(item)
                copied[
                    "source_file"
                ] = filename
                all_segments.append(
                    copied
                )

            # 공구 요약
            for item in result.get(
                "tool_summary",
                [],
            ):
                tool = int(
                    item["tool"]
                )

                count = int(
                    item.get(
                        "cutting_rows",
                        0,
                    )
                )

                agg = tool_rows[tool]
                agg[
                    "cutting_rows"
                ] += count

                rpm = safe_float(
                    item.get(
                        "rpm_median"
                    )
                )

                load = safe_float(
                    item.get(
                        "load_median"
                    )
                )

                if (
                    rpm is not None
                    and count > 0
                ):
                    agg[
                        "rpm_weighted"
                    ] += rpm * count
                    agg[
                        "rpm_weight"
                    ] += count

                if (
                    load is not None
                    and count > 0
                ):
                    agg[
                        "load_weighted"
                    ] += load * count
                    agg[
                        "load_weight"
                    ] += count

            # 전체 최고점
            score_max = safe_float(
                result.get(
                    "score_max"
                )
            )

            if (
                score_max is not None
                and score_max
                > global_highest_score
            ):
                global_highest_score = (
                    score_max
                )

                global_highest = dict(
                    result.get(
                        "highest_window"
                    )
                    or {}
                )

                global_highest[
                    "source_file"
                ] = filename

                global_contributions = (
                    result.get(
                        "top_contributions",
                        [],
                    )
                )

        all_top_windows.sort(
            key=lambda x:
                safe_float(
                    x.get(
                        "anomaly_score"
                    )
                )
                or -1,
            reverse=True,
        )

        tool_summary = []

        for tool, agg in sorted(
            tool_rows.items()
        ):
            cutting_rows = agg[
                "cutting_rows"
            ]

            tool_summary.append(
                {
                    "tool": tool,
                    "cutting_rows":
                        cutting_rows,
                    "cutting_ratio": (
                        cutting_rows
                        / total_cutting
                        if total_cutting > 0
                        else 0.0
                    ),
                    "rpm_median": (
                        agg[
                            "rpm_weighted"
                        ]
                        / agg[
                            "rpm_weight"
                        ]
                        if agg[
                            "rpm_weight"
                        ] > 0
                        else None
                    ),
                    "load_median": (
                        agg[
                            "load_weighted"
                        ]
                        / agg[
                            "load_weight"
                        ]
                        if agg[
                            "load_weight"
                        ] > 0
                        else None
                    ),
                }
            )

        valid_starts = [
            value
            for value in file_starts
            if not pd.isna(value)
        ]

        valid_ends = [
            value
            for value in file_ends
            if not pd.isna(value)
        ]

        return {
            "analysis_mode": "zip",
            "filename": "ZIP 일괄 분석",
            "file_count": len(results),
            "filenames": [
                r.get(
                    "filename",
                    "-"
                )
                for r in results
            ],
            "parsed_rows": sum(
                int(
                    r.get(
                        "parsed_rows",
                        0,
                    )
                )
                for r in results
            ),
            "clean_rows": total_clean,
            "cutting_rows": total_cutting,
            "cutting_ratio": (
                total_cutting
                / total_clean
                if total_clean > 0
                else 0.0
            ),
            "result_windows": sum(
                int(
                    r.get(
                        "result_windows",
                        0,
                    )
                )
                for r in results
            ),
            "supported_windows":
                total_supported,
            "unsupported_windows":
                total_unsupported,
            "insufficient_windows":
                total_insufficient,
            "score_mean": (
                weighted_score_sum
                / weighted_score_n
                if weighted_score_n > 0
                else None
            ),
            "score_max": (
                global_highest_score
                if global_highest_score
                >= 0
                else None
            ),
            "highest_window":
                global_highest,
            "top_windows":
                all_top_windows[:50],
            "top_contributions":
                global_contributions,
            "score_series":
                all_scores,
            "window_details":
                all_window_details,
            "used_tools":
                sorted(used_tools_set),
            "supported_tools":
                sorted(supported_tools_set),
            "unsupported_tools":
                sorted(unsupported_tools_set),
            "tool_analysis_summary": [
                {
                    "tool": tool,
                    "analyzed_windows": values["analyzed_windows"],
                    "baseline_missing_windows": values["baseline_missing_windows"],
                    "insufficient_windows": values["insufficient_windows"],
                    "analysis_supported": values["analyzed_windows"] > 0,
                }
                for tool, values in sorted(tool_analysis_rows.items())
            ],
            "used_tool_count":
                len(tool_summary),
            "segment_count":
                len(all_segments),
            "sampling_interval_mean": (
                interval_weighted
                / interval_weight
                if interval_weight > 0
                else None
            ),
            "file_start": (
                str(min(valid_starts))
                if valid_starts
                else None
            ),
            "file_end": (
                str(max(valid_ends))
                if valid_ends
                else None
            ),
            "upload_start_date": (
                str(min(uploaded_dates).date())
                if uploaded_dates
                else None
            ),
            "upload_end_date": (
                str(max(uploaded_dates).date())
                if uploaded_dates
                else None
            ),
            "tool_summary":
                tool_summary,
            "cutting_segments":
                all_segments,
            "daily_results":
                daily_results,
        }


    def api_predict_log(
        filename,
        file_bytes,
    ):
        response = requests.post(
            API_URL,
            files={
                "file": (
                    filename,
                    file_bytes,
                    "text/plain",
                )
            },
            timeout=300,
        )

        if response.status_code != 200:
            detail = response.text

            try:
                detail = (
                    response.json()
                    .get(
                        "detail",
                        detail,
                    )
                )
            except Exception:
                pass

            raise RuntimeError(
                f"{filename}: "
                f"HTTP {response.status_code} - "
                f"{detail}"
            )

        return response.json()


    def analyze_zip(
        uploaded_bytes,
    ):
        try:
            archive = zipfile.ZipFile(
                io.BytesIO(
                    uploaded_bytes
                )
            )
        except zipfile.BadZipFile as exc:
            raise ValueError(
                "정상적인 ZIP 파일이 아닙니다."
            ) from exc

        members = [
            info
            for info in archive.infolist()
            if (
                not info.is_dir()
                and info.filename
                .lower()
                .endswith(".log")
                and "__macosx/"
                not in info.filename.lower()
            )
        ]

        if not members:
            raise ValueError(
                "ZIP 안에서 .log 파일을 찾지 못했습니다."
            )

        if len(members) > MAX_ZIP_FILES:
            raise ValueError(
                f"ZIP에는 최대 {MAX_ZIP_FILES}개의 "
                ".log 파일만 넣을 수 있습니다."
            )

        total_uncompressed = sum(
            info.file_size
            for info in members
        )

        if (
            total_uncompressed
            > MAX_ZIP_UNCOMPRESSED_BYTES
        ):
            raise ValueError(
                "ZIP 압축 해제 후 전체 크기가 "
                "허용 범위를 초과합니다."
            )

        members = sorted(
            members,
            key=lambda x:
                x.filename,
        )

        progress = st.progress(
            0,
            text=(
                f"{len(members)}개 로그 파일 분석 준비 중..."
            ),
        )

        status_box = st.empty()

        results = []

        for index, info in enumerate(
            members,
            start=1,
        ):
            filename = Path(
                info.filename
            ).name

            status_box.info(
                f"[{index}/{len(members)}] "
                f"{filename} 분석 중"
            )

            with archive.open(info) as f:
                file_bytes = f.read()

            result = api_predict_log(
                filename,
                file_bytes,
            )

            results.append(
                result
            )

            progress.progress(
                index / len(members),
                text=(
                    f"{index}/{len(members)} "
                    "로그 분석 완료"
                ),
            )

        status_box.success(
            f"{len(members)}개 로그 파일 분석이 완료되었습니다."
        )

        progress.empty()

        return aggregate_zip_results(
            results
        )


    def build_reason_lines(
        highest,
        contributions,
    ):
        if not highest:
            return []

        lines = []

        load_ratio = safe_float(
            highest.get(
                "load_vs_normal"
            )
        )

        if load_ratio is not None:
            if load_ratio >= 1.10:
                lines.append(
                    "Spindle Load가 공구별 정상 기준보다 "
                    f"약 {(load_ratio - 1) * 100:.1f}% 높았습니다."
                )
            elif load_ratio >= 1.02:
                lines.append(
                    "Spindle Load가 공구별 정상 기준보다 "
                    f"약 {(load_ratio - 1) * 100:.1f}% 조금 높았습니다."
                )
            else:
                lines.append(
                    "Spindle Load는 공구별 정상 기준과 "
                    "큰 차이가 없었습니다."
                )

        rpm_std = safe_float(
            highest.get(
                "rpm_std_vs_normal"
            )
        )

        if rpm_std is not None:
            if rpm_std >= 2:
                lines.append(
                    "RPM 변동성이 공구별 정상 기준보다 "
                    f"약 {rpm_std:.1f}배 크게 나타났습니다."
                )
            elif rpm_std >= 1:
                lines.append(
                    "RPM 변동성이 평소보다 다소 크게 나타났습니다."
                )
            else:
                lines.append(
                    "RPM 변동성은 평소보다 크게 나타나지 않았습니다."
                )

        rpm_error = safe_float(
            highest.get(
                "rpm_error_vs_normal"
            )
        )

        if rpm_error is not None:
            if rpm_error >= 1:
                lines.append(
                    "명령 RPM과 실제 RPM의 차이가 "
                    "공구별 정상 허용 기준보다 크게 나타났습니다."
                )
            else:
                lines.append(
                    "명령 RPM과 실제 RPM의 차이는 "
                    "공구별 정상 허용 기준 안쪽이었습니다."
                )

        if contributions:
            names = [
                FEATURE_NAME_MAP.get(
                    item.get("feature"),
                    item.get("feature"),
                )
                for item in contributions
            ]

            names = [
                name
                for name in names
                if name
            ]

            if names:
                lines.append(
                    "XGBoost 내부 판단에서는 "
                    + ", ".join(names)
                    + " 항목의 영향이 상대적으로 컸습니다."
                )

        return lines


    # =========================================================
    # PDF
    # =========================================================

    def find_korean_font():
        candidates = [
            Path(
                r"C:\Windows\Fonts\malgun.ttf"
            ),
            Path(
                r"C:\Windows\Fonts\malgunbd.ttf"
            ),
            Path(
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
            ),
        ]

        for path in candidates:
            if path.exists():
                return path

        return None


    def make_pdf_report(result):
        try:
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import (
                ParagraphStyle,
                getSampleStyleSheet,
            )
            from reportlab.lib.units import mm
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.platypus import (
                Paragraph,
                SimpleDocTemplate,
                Spacer,
                Table,
                TableStyle,
            )
        except ImportError as exc:
            raise RuntimeError(
                "PDF 기능을 사용하려면 "
                "`python -m pip install reportlab`을 실행해주세요."
            ) from exc

        font_path = find_korean_font()

        if font_path is None:
            raise RuntimeError(
                "한글 PDF 생성을 위한 시스템 한글 폰트를 찾지 못했습니다."
            )

        font_name = "MCTKorean"
        pdfmetrics.registerFont(
            TTFont(
                font_name,
                str(font_path),
            )
        )

        buffer = io.BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=16 * mm,
            leftMargin=16 * mm,
            topMargin=16 * mm,
            bottomMargin=16 * mm,
            title="MCT AI 지도학습 분석 보고서",
        )

        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            "title_kr",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=18,
            leading=23,
            alignment=TA_CENTER,
            textColor=colors.HexColor(
                "#17365D"
            ),
        )

        heading_style = ParagraphStyle(
            "heading_kr",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=12,
            leading=16,
            spaceBefore=8,
            spaceAfter=6,
        )

        body_style = ParagraphStyle(
            "body_kr",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9,
            leading=14,
        )

        small_style = ParagraphStyle(
            "small_kr",
            parent=body_style,
            fontSize=7.5,
            leading=11,
            textColor=colors.HexColor(
                "#555555"
            ),
        )

        elements = []

        elements.append(
            Paragraph(
                "MCT 3호 지도학습 이상 패턴 분석 보고서",
                title_style,
            )
        )

        elements.append(
            Spacer(
                1,
                5 * mm,
            )
        )

        mode_text = (
            "ZIP 다중 로그 분석"
            if result.get(
                "analysis_mode"
            ) == "zip"
            else "단일 로그 분석"
        )

        score = safe_float(
            result.get(
                "score_max"
            )
        )

        level, _ = score_level(
            score
        )

        highest = result.get(
            "highest_window"
        ) or {}

        summary_rows = [
            [
                "분석 방식",
                mode_text,
                "분석 파일",
                f"{result.get('file_count', 1)}개",
            ],
            [
                "업로드 데이터 기간",
                uploaded_period_text(result),
                "실제 절삭 데이터",
                f"{int(result.get('cutting_rows', 0)):,}건",
            ],
            [
                "AI 분석 기간",
                analysis_period_text(result),
                "사용 공구",
                f"{int(result.get('used_tool_count', 0))}종",
            ],
            [
                "AI 분석 Window",
                f"{int(result.get('supported_windows', 0)):,}개",
                "분석 가능 공구",
                f"{len(result.get('supported_tools', []))}종",
            ],
            [
                "최대 Anomaly Score",
                (
                    f"{score:.6f}"
                    if score is not None
                    else "-"
                ),
                "참고 수준",
                level,
            ],
        ]

        table = Table(
            summary_rows,
            colWidths=[
                34 * mm,
                62 * mm,
                32 * mm,
                36 * mm,
            ],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, -1),
                        font_name,
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        8,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (0, -1),
                        colors.HexColor(
                            "#EAF2F8"
                        ),
                    ),
                    (
                        "BACKGROUND",
                        (2, 0),
                        (2, -1),
                        colors.HexColor(
                            "#EAF2F8"
                        ),
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor(
                            "#CCCCCC"
                        ),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                ]
            )
        )

        elements.append(
            table
        )

        elements.append(
            Spacer(
                1,
                5 * mm,
            )
        )

        elements.append(
            Paragraph(
                "1. AI 분석 요약",
                heading_style,
            )
        )

        if highest:
            highest_text = (
                f"파일 내 최고점은 Tool {highest.get('current_tool', '-')}의 "
                f"{format_dt(highest.get('window_start'))} - "
                f"{format_dt(highest.get('window_end'))} 구간에서 확인되었습니다. "
                f"최대 Anomaly Score는 "
                f"{score:.6f}입니다."
                if score is not None
                else "최고점 정보를 확인할 수 없습니다."
            )
        else:
            highest_text = (
                "분석 가능한 최고점 Window가 없습니다."
            )

        elements.append(
            Paragraph(
                highest_text,
                body_style,
            )
        )

        elements.append(
            Spacer(
                1,
                3 * mm,
            )
        )

        elements.append(
            Paragraph(
                "2. 최고점 구간 해석",
                heading_style,
            )
        )

        reasons = build_reason_lines(
            highest,
            result.get(
                "top_contributions",
                [],
            ),
        )

        if reasons:
            for reason in reasons:
                elements.append(
                    Paragraph(
                        "• " + reason,
                        body_style,
                    )
                )
        else:
            elements.append(
                Paragraph(
                    "상세 해석 정보가 없습니다.",
                    body_style,
                )
            )

        elements.append(
            Spacer(
                1,
                3 * mm,
            )
        )

        elements.append(
            Paragraph(
                "3. 최고 점수 구간",
                heading_style,
            )
        )

        top_windows = result.get(
            "top_windows",
            [],
        )[:10]

        rows = [
            [
                "순위",
                "시작",
                "종료",
                "공구",
                "Score",
            ]
        ]

        for index, item in enumerate(
            top_windows,
            start=1,
        ):
            rows.append(
                [
                    str(index),
                    format_dt(
                        item.get(
                            "window_start"
                        )
                    ),
                    format_dt(
                        item.get(
                            "window_end"
                        )
                    ),
                    f"Tool {item.get('current_tool', '-')}",
                    (
                        f"{safe_float(item.get('anomaly_score')):.6f}"
                        if safe_float(
                            item.get(
                                "anomaly_score"
                            )
                        )
                        is not None
                        else "-"
                    ),
                ]
            )

        top_table = Table(
            rows,
            repeatRows=1,
            colWidths=[
                12 * mm,
                48 * mm,
                48 * mm,
                24 * mm,
                28 * mm,
            ],
        )

        top_table.setStyle(
            TableStyle(
                [
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, -1),
                        font_name,
                    ),
                    (
                        "FONTSIZE",
                        (0, 0),
                        (-1, -1),
                        7,
                    ),
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#17365D"
                        ),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.35,
                        colors.HexColor(
                            "#CCCCCC"
                        ),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "MIDDLE",
                    ),
                ]
            )
        )

        elements.append(
            top_table
        )

        elements.append(
            Spacer(
                1,
                4 * mm,
            )
        )

        elements.append(
            Paragraph(
                "4. 현장 확인 권장 사항",
                heading_style,
            )
        )

        checks = [
            "해당 시간대의 공구 교체 또는 공정 전환 여부를 확인합니다.",
            "해당 공구의 마모·파손·체결 상태를 확인합니다.",
            "Spindle Load와 명령/실제 RPM 차이가 일시적으로 변했는지 원본 로그와 함께 확인합니다.",
        ]

        for check in checks:
            elements.append(
                Paragraph(
                    "• " + check,
                    body_style,
                )
            )

        elements.append(
            Spacer(
                1,
                4 * mm,
            )
        )

        elements.append(
            Paragraph(
                "주의사항",
                heading_style,
            )
        )

        elements.append(
            Paragraph(
                "Anomaly Score는 실제 고장 확률이 아니라 모델이 학습한 가상 이상 패턴과의 유사도입니다. "
                "P95/P99는 실제 2026년 5월 로그 분포의 참고값이며 고장 판정 임계값이 아닙니다. "
                "공구 교체나 공정 전환에서도 높은 점수가 나타날 수 있으므로 현장 정보와 함께 판단해야 합니다.",
                small_style,
            )
        )

        doc.build(
            elements
        )

        buffer.seek(
            0
        )

        return buffer.getvalue()


    # =========================================================
    # 파일 업로드 및 분석
    # =========================================================

    with st.container(
        border=True
    ):
        st.markdown(
            "##### 지도학습 XGBoost 분석 데이터"
        )

        st.caption(
            "단일 `.log` 또는 여러 날짜의 `.log`가 들어 있는 `.zip`을 업로드하면 저장된 XGBoost 모델로 분석합니다. 업로드 데이터로 모델을 다시 학습하지 않습니다."
        )

        uploaded_file = (
            st.file_uploader(
                "MCT 로그",
                type=[
                    "log",
                    "zip",
                ],
                label_visibility="collapsed",
            )
        )

        current_hash = None

        if uploaded_file is not None:
            file_bytes = (
                uploaded_file
                .getvalue()
            )

            current_hash = (
                hashlib
                .sha1(
                    file_bytes
                )
                .hexdigest()
            )

            file_size = (
                uploaded_file.size
                / 1024
                / 1024
            )

            suffix = (
                Path(
                    uploaded_file.name
                )
                .suffix
                .lower()
            )

            col1, col2, col3 = (
                st.columns(
                    [2.5, 1, 1]
                )
            )

            with col1:
                st.markdown(
                    f"**파일**　`{uploaded_file.name}`"
                )

            with col2:
                st.markdown(
                    f"**크기**　{file_size:.2f} MB"
                )

            with col3:
                st.markdown(
                    "**분석 방식**　"
                    + (
                        "다중 로그"
                        if suffix == ".zip"
                        else "단일 로그"
                    )
                )

            if st.button(
                "분석 시작",
                type="primary",
                width="stretch",
            ):
                try:
                    if suffix == ".log":
                        with st.spinner(
                            "로그 정제 → 절삭 구간 추출 → XGBoost 추론 중..."
                        ):
                            raw_result = (
                                api_predict_log(
                                    uploaded_file.name,
                                    file_bytes,
                                )
                            )

                            result = (
                                normalize_single_result(
                                    raw_result
                                )
                            )

                    else:
                        result = (
                            analyze_zip(
                                file_bytes
                            )
                        )

                    st.session_state[
                        "supervised_result"
                    ] = result

                    st.session_state[
                        "supervised_file_hash"
                    ] = current_hash

                    st.success(
                        "분석이 완료되었습니다."
                    )

                except requests.exceptions.ConnectionError:
                    st.error(
                        "FastAPI 서버에 연결할 수 없습니다. "
                        "`python -m uvicorn api.main:app --reload` 실행 여부를 확인해주세요."
                    )

                except requests.exceptions.Timeout:
                    st.error(
                        "로그 분석 요청 시간이 초과되었습니다."
                    )

                except Exception as exc:
                    st.error(
                        f"분석 중 오류: {exc}"
                    )


    has_result = (
        uploaded_file is not None
        and
        "supervised_result"
        in st.session_state
        and
        st.session_state.get(
            "supervised_file_hash"
        )
        == current_hash
    )


    # =========================================================
    # 결과 Dashboard
    # =========================================================

    if has_result:

        result = st.session_state[
            "supervised_result"
        ]

        score_max = safe_float(
            result.get(
                "score_max"
            )
        )

        score_mean = safe_float(
            result.get(
                "score_mean"
            )
        )

        level, level_color = (
            score_level(
                score_max
            )
        )

        highest = (
            result.get(
                "highest_window"
            )
            or {}
        )

        mode_text = (
            "ZIP 다중 로그"
            if result.get(
                "analysis_mode"
            ) == "zip"
            else "단일 로그"
        )

        upload_period = uploaded_period_text(result)
        ai_period = analysis_period_text(result)

        st.markdown(
            '<div class="chips">'
            f'<span class="chip">{html.escape(mode_text)}</span>'
            f'<span class="chip">분석 파일 {int(result.get("file_count", 1))}개</span>'
            f'<span class="chip">업로드 데이터 기간 {html.escape(upload_period)}</span>'
            f'<span class="chip">AI 분석 기간 {html.escape(ai_period)}</span>'
            f'<span class="chip">AI 분석 구간 {int(result.get("supported_windows", 0)):,}개</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # PDF
        pdf_col1, pdf_col2 = (
            st.columns(
                [4, 1]
            )
        )

        with pdf_col1:
            st.caption(
                "분석 결과를 확인한 뒤 필요한 경우 PDF 요약 보고서로 저장할 수 있습니다."
            )

        with pdf_col2:
            try:
                pdf_bytes = (
                    make_pdf_report(
                        result
                    )
                )

                st.download_button(
                    "PDF 보고서",
                    data=pdf_bytes,
                    file_name=(
                        "mct_supervised_report.pdf"
                    ),
                    mime="application/pdf",
                    width="stretch",
                )

            except Exception as exc:
                st.button(
                    "PDF 보고서",
                    disabled=True,
                    width="stretch",
                )

                st.caption(
                    str(exc)
                )

        tab1, tab2, tab3, tab4 = (
            st.tabs(
                [
                    "① 전체 이상 패턴 흐름",
                    "② 확인이 필요한 시간대",
                    "③ 가공 데이터 현황",
                    "④ AI 분석 기준",
                ]
            )
        )

        # =====================================================
        # ① 이상 점수 추이
        # =====================================================

        with tab1:

            overview_title = (
                "한 달 전체 이상 패턴 흐름"
                if result.get("analysis_mode") == "zip"
                else "단일 로그 이상 패턴 흐름"
            )

            overview_subtitle = (
                "월간 로그에서 AI가 학습한 이상 패턴과 유사한 변화가 언제 나타났는지 확인합니다."
                if result.get("analysis_mode") == "zip"
                else "이 로그에서 AI가 학습한 이상 패턴과 유사한 변화가 나타났는지 확인합니다."
            )

            section(
                "OVERVIEW",
                overview_title,
                overview_subtitle,
            )

            highest_tool = (
                highest.get(
                    "current_tool",
                    "-"
                )
            )

            highest_time = (
                format_dt(
                    highest.get(
                        "window_start"
                    ),
                    with_date=(
                        result.get(
                            "analysis_mode"
                        )
                        == "zip"
                    ),
                )
            )

            kpi_row(
                [
                    {
                        "label":
                            "AI 분석 결과",
                        "value":
                            level,
                        "dot":
                            level_color,
                        "sub":
                            "P95/P99 실제 로그 분포 기준",
                    },
                    {
                        "label":
                            "가장 높은 이상 패턴 점수",
                        "value":
                            (
                                f"{score_max:.4f}"
                                if score_max
                                is not None
                                else "-"
                            ),
                        "sub":
                            (
                                f"평균 {score_mean:.4f}"
                                if score_mean
                                is not None
                                else "평균 없음"
                            ),
                    },
                    {
                        "label":
                            "가장 높은 점수의 공구",
                        "value":
                            f"T{int(highest_tool):02d}"
                            if str(
                                highest_tool
                            ).isdigit()
                            else "-",
                        "sub":
                            highest_time,
                    },
                    {
                        "label":
                            "실제 절삭 데이터",
                        "value":
                            f"{int(result.get('cutting_rows', 0)):,}",
                        "sub":
                            "RUN & RPM>0 & FEED>0",
                    },
                    {
                        "label":
                            "사용 공구 / AI 분석 가능",
                        "value":
                            f"{int(result.get('used_tool_count', 0))}종 / {len(result.get('supported_tools', []))}종",
                        "sub":
                            "정상 기준이 없는 공구는 AI 상세 분석에서 제외",
                    },
                ]
            )

            detail_for_summary = pd.DataFrame(
                result.get(
                    "window_details",
                    [],
                )
            )

            if not detail_for_summary.empty:
                score_values = pd.to_numeric(
                    detail_for_summary["anomaly_score"],
                    errors="coerce",
                ).dropna()

                total_ai = int(len(score_values))
                p95_count = int((score_values >= REFERENCE_P95).sum())
                p99_count = int((score_values >= REFERENCE_P99).sum())
                score_09_count = int((score_values >= 0.9).sum())

                p99_ratio = (
                    p99_count / total_ai * 100
                    if total_ai > 0
                    else 0.0
                )
                score_09_ratio = (
                    score_09_count / total_ai * 100
                    if total_ai > 0
                    else 0.0
                )

                highest_desc = (
                    f"{format_dt(highest.get('window_start'))} · "
                    f"Tool {highest.get('current_tool', '-')}"
                    if highest
                    else "-"
                )

                st.markdown(
                    f"""
                    <div class="panel">
                    <div class="panel-title">그래서 무엇을 확인하면 되나요?</div>
                    <div class="panel-text">
                    전체 <b>{total_ai:,}개 1분 분석 구간</b> 중
                    <b>{p99_count:,}개({p99_ratio:.2f}%)</b>가 실제 5월 로그의 상위 1% 참고선(P99) 이상이었고,
                    그중 <b>{score_09_count:,}개({score_09_ratio:.2f}%)</b>는 이상 패턴 점수가 0.9 이상이었습니다.<br>
                    먼저 <b>{highest_desc}</b>와 같은 높은 점수 시간대의
                    <b>공구 교체·공정 전환 여부</b>를 확인한 뒤, 부하·RPM 변화를 함께 보는 것이 좋습니다.<br>
                    <span class="small-note">높은 점수는 고장 확정이 아닙니다. 공구 교체·절삭 종료 같은 정상 공정 변화에서도 높아질 수 있습니다.</span>
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            score_df = pd.DataFrame(
                result.get(
                    "score_series",
                    [],
                )
            )

            if (
                not score_df.empty
                and "window_start"
                in score_df.columns
            ):
                score_df[
                    "window_start"
                ] = pd.to_datetime(
                    score_df[
                        "window_start"
                    ],
                    errors="coerce",
                )

                score_df = (
                    score_df
                    .dropna(
                        subset=[
                            "window_start",
                            "anomaly_score",
                        ]
                    )
                    .sort_values(
                        "window_start"
                    )
                )

            if go is None:
                st.warning(
                    "추이 그래프를 표시하려면 plotly가 필요합니다."
                )

            elif score_df.empty:
                st.info(
                    "이 파일에는 실제 절삭 데이터가 없어 AI 이상 패턴 점수를 계산할 1분 구간이 없습니다."
                )

            else:
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=score_df[
                            "window_start"
                        ],
                        y=score_df[
                            "anomaly_score"
                        ],
                        mode="lines",
                        name="이상 패턴 점수",
                        line=dict(
                            color=T[
                                "blue"
                            ],
                            width=1.6,
                        ),
                        hovertemplate=(
                            "%{x|%Y-%m-%d %H:%M:%S}"
                            "<br>Score %{y:.6f}"
                            "<extra></extra>"
                        ),
                    )
                )

                fig.add_hline(
                    y=REFERENCE_P95,
                    line_dash="dash",
                    line_color=T[
                        "yellow"
                    ],
                    annotation_text=(
                        f"상위 5% 참고선 {REFERENCE_P95:.4f}"
                    ),
                    annotation_position="top left",
                )

                fig.add_hline(
                    y=REFERENCE_P99,
                    line_dash="dot",
                    line_color=T[
                        "red"
                    ],
                    annotation_text=(
                        f"상위 1% 참고선 {REFERENCE_P99:.4f}"
                    ),
                    annotation_position="top right",
                )

                # 31일처럼 P99가 너무 높으면 그래프가 눌리지 않도록
                # 실제 데이터와 P95 중심으로 보되 P99는 필요 시 자동 포함.
                data_max = float(
                    score_df[
                        "anomaly_score"
                    ].max()
                )

                y_top = max(
                    data_max * 1.12,
                    REFERENCE_P95 * 1.20,
                )

                if data_max >= (
                    REFERENCE_P99
                    * 0.65
                ):
                    y_top = max(
                        y_top,
                        REFERENCE_P99
                        * 1.08,
                    )

                fig.update_yaxes(
                    range=[
                        0,
                        y_top,
                    ]
                )

                style_figure(
                    fig,
                    height=430,
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                    config=PLOT_CONFIG,
                )

            st.caption(
                "P95/P99는 2026년 5월 실제 로그 Score 분포의 참고값입니다. "
                "고장 판정 임계값이 아니며 공구 교체·공정 전환에서도 높은 점수가 나타날 수 있습니다."
            )

            # ZIP의 경우 일별 최고점
            if (
                result.get(
                    "analysis_mode"
                )
                == "zip"
            ):
                section(
                    "DAILY",
                    "일별 최고 Anomaly Score",
                    "월간 분석에서 어느 날짜에 상대적으로 높은 Score가 나타났는지 확인합니다.",
                )

                daily_rows = []

                for item in result.get(
                    "daily_results",
                    [],
                ):
                    high = (
                        item.get(
                            "highest_window"
                        )
                        or {}
                    )

                    daily_rows.append(
                        {
                            "날짜":
                                format_dt(
                                    high.get(
                                        "window_start"
                                    )
                                )[:10],
                            "파일":
                                item.get(
                                    "filename",
                                    "-"
                                ),
                            "최대 Score":
                                safe_float(
                                    item.get(
                                        "score_max"
                                    )
                                ),
                            "평균 Score":
                                safe_float(
                                    item.get(
                                        "score_mean"
                                    )
                                ),
                            "공구":
                                (
                                    f"Tool {high.get('current_tool')}"
                                    if high
                                    else "-"
                                ),
                        }
                    )

                daily_df = pd.DataFrame(
                    daily_rows
                ).dropna(
                    subset=[
                        "최대 Score"
                    ]
                )

                if (
                    go is not None
                    and not daily_df.empty
                ):
                    daily_df[
                        "날짜"
                    ] = pd.to_datetime(
                        daily_df[
                            "날짜"
                        ],
                        errors="coerce",
                    )

                    daily_fig = go.Figure()

                    daily_fig.add_trace(
                        go.Bar(
                            x=daily_df[
                                "날짜"
                            ],
                            y=daily_df[
                                "최대 Score"
                            ],
                            name="일별 최대 Score",
                            marker_color=T[
                                "blue"
                            ],
                            hovertemplate=(
                                "%{x|%m/%d}"
                                "<br>최대 %{y:.6f}"
                                "<extra></extra>"
                            ),
                        )
                    )

                    daily_fig.add_hline(
                        y=REFERENCE_P95,
                        line_dash="dash",
                        line_color=T[
                            "yellow"
                        ],
                    )

                    daily_fig.add_hline(
                        y=REFERENCE_P99,
                        line_dash="dot",
                        line_color=T[
                            "red"
                        ],
                    )

                    style_figure(
                        daily_fig,
                        height=350,
                    )

                    st.plotly_chart(
                        daily_fig,
                        width="stretch",
                        config=PLOT_CONFIG,
                    )

                st.dataframe(
                    daily_df.sort_values(
                        "최대 Score",
                        ascending=False,
                    ),
                    width="stretch",
                    hide_index=True,
                )

        # =====================================================
        # ② 확인이 필요한 시간대
        # =====================================================

        with tab2:

            section(
                "WINDOW REVIEW",
                "확인이 필요한 시간대",
                "AI 분석이 완료된 모든 공구와 모든 1분 구간을 조회할 수 있습니다. 높은 점수부터 확인하되 실제 작업 상황을 함께 보세요.",
            )

            detail_df = pd.DataFrame(
                result.get(
                    "window_details",
                    [],
                )
            )

            all_used_tools = sorted(
                int(tool)
                for tool in result.get(
                    "used_tools",
                    [],
                )
            )

            supported_tools = set(
                int(tool)
                for tool in result.get(
                    "supported_tools",
                    [],
                )
            )

            unsupported_tools = set(
                int(tool)
                for tool in result.get(
                    "unsupported_tools",
                    [],
                )
            )

            if detail_df.empty:
                st.info("AI 분석이 완료된 1분 구간이 없습니다.")

            else:
                detail_df["window_start_dt"] = pd.to_datetime(
                    detail_df["window_start"],
                    errors="coerce",
                )
                detail_df["window_end_dt"] = pd.to_datetime(
                    detail_df["window_end"],
                    errors="coerce",
                )
                detail_df["current_tool"] = pd.to_numeric(
                    detail_df["current_tool"],
                    errors="coerce",
                ).astype("Int64")
                detail_df["anomaly_score"] = pd.to_numeric(
                    detail_df["anomaly_score"],
                    errors="coerce",
                )
                detail_df = detail_df.dropna(
                    subset=["window_start_dt", "current_tool", "anomaly_score"]
                ).copy()

                total_ai = len(detail_df)
                p95_count = int((detail_df["anomaly_score"] >= REFERENCE_P95).sum())
                p99_count = int((detail_df["anomaly_score"] >= REFERENCE_P99).sum())
                score_09_count = int((detail_df["anomaly_score"] >= 0.9).sum())

                kpi_row(
                    [
                        {
                            "label": "AI 분석 완료 구간",
                            "value": f"{total_ai:,}개",
                            "sub": "1분 단위 절삭 구간",
                        },
                        {
                            "label": "평소보다 높은 구간",
                            "value": f"{p95_count:,}개",
                            "dot": T["yellow"],
                            "sub": "상위 5% 참고선 이상",
                        },
                        {
                            "label": "우선 확인 구간",
                            "value": f"{p99_count:,}개",
                            "dot": T["red"],
                            "sub": "상위 1% 참고선 이상",
                        },
                        {
                            "label": "점수 0.9 이상",
                            "value": f"{score_09_count:,}개",
                            "sub": "매우 높은 모델 유사도",
                        },
                        {
                            "label": "사용 공구 / AI 분석 가능",
                            "value": f"{len(all_used_tools)}종 / {len(supported_tools)}종",
                            "sub": "전체 사용 공구를 필터에서 확인 가능",
                        },
                    ]
                )

                st.markdown(
                    """
                    <div class="panel">
                    <div class="panel-title">화면을 이렇게 보면 됩니다</div>
                    <div class="panel-text">
                    <b>우선 확인</b>은 실제 5월 로그에서 상대적으로 매우 드문 높은 점수 구간입니다.
                    먼저 해당 시각에 공구 교체·공정 전환·절삭 종료가 있었는지 확인하고,
                    그런 정상 변화가 아니라면 평소 대비 부하·RPM 흔들림·RPM 오차를 살펴보세요.
                    </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---------------- Filters ----------------
                date_values = sorted(
                    detail_df["window_start_dt"].dt.date.unique().tolist()
                )

                col_date, col_tool, col_state = st.columns([1, 1, 1.35])

                with col_date:
                    selected_date = st.selectbox(
                        "날짜",
                        options=["전체"] + date_values,
                        format_func=lambda value: (
                            "전체 날짜"
                            if value == "전체"
                            else value.strftime("%Y-%m-%d")
                        ),
                    )

                with col_tool:
                    selected_tool = st.selectbox(
                        "공구",
                        options=["전체"] + all_used_tools,
                        format_func=lambda value: (
                            "전체 공구"
                            if value == "전체"
                            else (
                                f"Tool {value} · AI 분석 가능"
                                if value in supported_tools
                                else f"Tool {value} · 정상 기준 없음"
                            )
                        ),
                    )

                with col_state:
                    selected_state = st.selectbox(
                        "AI 분석 상태",
                        options=[
                            "우선 확인 (상위 1% 참고선 이상)",
                            "관찰 포함 (상위 5% 참고선 이상)",
                            "전체 분석 구간",
                        ],
                    )

                filtered = detail_df.copy()

                if selected_date != "전체":
                    filtered = filtered[
                        filtered["window_start_dt"].dt.date == selected_date
                    ]

                if selected_tool != "전체":
                    if selected_tool in unsupported_tools:
                        st.warning(
                            f"Tool {selected_tool}은 실제 절삭 데이터에는 존재하지만 "
                            "학습용 정상 기준(Baseline)이 없어 XGBoost 상세 분석에서 제외되었습니다. "
                            "공구 사용 현황은 ③ 가공 데이터 현황에서 확인할 수 있습니다."
                        )
                        filtered = filtered.iloc[0:0]
                    else:
                        filtered = filtered[
                            filtered["current_tool"].astype(int) == int(selected_tool)
                        ]

                if selected_state.startswith("우선 확인"):
                    filtered = filtered[
                        filtered["anomaly_score"] >= REFERENCE_P99
                    ]
                elif selected_state.startswith("관찰 포함"):
                    filtered = filtered[
                        filtered["anomaly_score"] >= REFERENCE_P95
                    ]

                filtered = filtered.sort_values(
                    "anomaly_score",
                    ascending=False,
                )

                st.caption(
                    f"현재 조건에 해당하는 구간: {len(filtered):,}개 · 점수가 높은 순으로 최대 300개를 표에 표시합니다."
                )

                if filtered.empty:
                    if not (
                        selected_tool != "전체"
                        and selected_tool in unsupported_tools
                    ):
                        st.info("선택한 조건에 해당하는 분석 구간이 없습니다.")
                else:
                    display_source = filtered.head(300).copy()

                    display = pd.DataFrame(
                        {
                            "시작 시간": display_source["window_start_dt"].dt.strftime("%Y-%m-%d %H:%M:%S"),
                            "종료 시간": display_source["window_end_dt"].dt.strftime("%H:%M:%S"),
                            "공구": "Tool " + display_source["current_tool"].astype(int).astype(str),
                            "이상 패턴 점수": display_source["anomaly_score"].round(6),
                            "평소 대비 부하": display_source["load_vs_normal"].map(percent_change_text),
                            "RPM 흔들림": display_source["rpm_std_vs_normal"].map(multiple_text),
                            "RPM 오차": display_source["rpm_error_vs_normal"].map(multiple_text),
                            "AI 분석 상태": display_source["anomaly_score"].map(status_label),
                        }
                    )

                    st.dataframe(
                        display,
                        width="stretch",
                        hide_index=True,
                    )

                    st.markdown("##### 선택 구간 쉽게 보기")

                    labels = []

                    for idx, row in filtered.head(300).iterrows():
                        labels.append(
                            (
                                idx,
                                f"{format_dt(row['window_start'])} · "
                                f"Tool {int(row['current_tool'])} · "
                                f"이상 패턴 점수 {row['anomaly_score']:.6f}",
                            )
                        )

                    selected_label = st.selectbox(
                        "상세 구간",
                        options=[label for _, label in labels],
                        label_visibility="collapsed",
                    )

                    selected_index = next(
                        idx
                        for idx, label in labels
                        if label == selected_label
                    )

                    row = filtered.loc[selected_index]
                    selected_score = safe_float(row["anomaly_score"])
                    selected_status = status_label(selected_score)
                    _, selected_color = score_level(selected_score)

                    kpi_row(
                        [
                            {
                                "label": "AI 분석 상태",
                                "value": selected_status,
                                "dot": selected_color,
                            },
                            {
                                "label": "공구",
                                "value": f"T{int(row['current_tool']):02d}",
                            },
                            {
                                "label": "이상 패턴 점수",
                                "value": f"{selected_score:.6f}",
                                "sub": "실제 고장 확률이 아님",
                            },
                            {
                                "label": "평소 대비 부하",
                                "value": percent_change_text(row.get("load_vs_normal")),
                                "sub": "해당 공구의 평소 기준 대비",
                            },
                            {
                                "label": "RPM 흔들림",
                                "value": multiple_text(row.get("rpm_std_vs_normal")),
                                "sub": "평소 변동성 대비",
                            },
                            {
                                "label": "RPM 오차",
                                "value": multiple_text(row.get("rpm_error_vs_normal")),
                                "sub": "평소 허용 기준 대비",
                            },
                        ]
                    )

                    explanation_lines = selected_window_explanation(row)
                    explanation_html = "".join(
                        f"<li>{html.escape(line)}</li>"
                        for line in explanation_lines
                    )

                    if selected_status == "우선 확인":
                        intro = (
                            "<b>이 시간대는 우선 확인을 권장합니다.</b> "
                            "다만 공구 교체나 공정 전환이었다면 정상적인 작업 변화일 수 있습니다."
                        )
                    elif selected_status == "관찰":
                        intro = (
                            "<b>평소보다 높은 패턴이 관찰되었습니다.</b> "
                            "전후 작업 상황과 함께 확인하는 것이 좋습니다."
                        )
                    else:
                        intro = (
                            "<b>실제 로그 분포상 일반 범위에 가까운 구간입니다.</b> "
                            "파일 내부 비교를 위해 상세 값을 제공합니다."
                        )

                    st.markdown(
                        f"""
                        <div class="warning-panel">
                        <div class="panel-title">그래서 이 시간대는 어떻게 보면 되나요?</div>
                        <div class="panel-text">{intro}</div>
                        <ul>{explanation_html}</ul>
                        <div class="small-note">
                        이상 패턴 점수는 XGBoost가 학습한 가상 이상 패턴과의 유사도이며 실제 고장 확률이 아닙니다.
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # 전체 최고점은 XGBoost Feature Contribution까지 계산되어 있으므로 별도 표시
            if highest:
                section(
                    "MODEL EXPLANATION",
                    "전체 데이터에서 가장 높은 점수의 판단 근거",
                    "가장 높은 점수 구간에 대해서는 XGBoost 내부 기여 Feature까지 함께 보여줍니다.",
                )

                reasons = build_reason_lines(
                    highest,
                    result.get(
                        "top_contributions",
                        [],
                    ),
                )

                left, right = st.columns([1.15, 1])

                with left:
                    reason_html = "".join(
                        f"<li>{html.escape(line)}</li>"
                        for line in reasons
                    )

                    st.markdown(
                        f"""
                        <div class="warning-panel">
                        <div class="panel-title">
                        Tool {highest.get("current_tool", "-")} · {format_dt(highest.get("window_start"))}
                        </div>
                        <ul>{reason_html}</ul>
                        <div class="small-note">
                        모델 내부의 판단 근거이며 실제 고장 원인을 확정한 결과는 아닙니다.
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with right:
                    contributions = result.get(
                        "top_contributions",
                        [],
                    )

                    if contributions:
                        contribution_df = pd.DataFrame(contributions)
                        contribution_df["AI가 본 주요 변화"] = contribution_df["feature"].map(
                            lambda x: FEATURE_NAME_MAP.get(x, x)
                        )
                        contribution_df["모델 기여도"] = contribution_df["contribution"].round(4)

                        st.dataframe(
                            contribution_df[["AI가 본 주요 변화", "모델 기여도"]],
                            width="stretch",
                            hide_index=True,
                        )

                    st.markdown(
                        """
                        <div class="good-panel">
                        <div class="panel-title">현장에서는 이것부터 확인하세요</div>
                        <div class="panel-text">
                        ① 해당 시각의 공구 교체·공정 전환 여부<br>
                        ② 해당 공구의 마모·파손·체결 상태<br>
                        ③ Spindle 부하와 명령/실제 RPM의 순간 변화
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # =====================================================
        # ③ 업로드 데이터 분석
        # =====================================================

        with tab3:

            section(
                "DATA PROFILE",
                "가공 데이터 현황",
                "AI 결과와 별개로 업로드한 로그에 어떤 공구와 절삭 작업이 있었는지 확인합니다.",
            )

            interval_mean = safe_float(
                result.get(
                    "sampling_interval_mean"
                )
            )

            kpi_row(
                [
                    {
                        "label":
                            "분석 파일",
                        "value":
                            f"{int(result.get('file_count', 1))}개",
                        "sub":
                            mode_text,
                    },
                    {
                        "label":
                            "전체 측정 데이터",
                        "value":
                            f"{int(result.get('clean_rows', 0)):,}",
                        "sub":
                            "파싱·정제 후 행",
                    },
                    {
                        "label":
                            "실제 절삭 데이터",
                        "value":
                            f"{int(result.get('cutting_rows', 0)):,}",
                        "sub":
                            f"전체의 {safe_float(result.get('cutting_ratio')) * 100:.1f}%"
                            if safe_float(
                                result.get(
                                    "cutting_ratio"
                                )
                            )
                            is not None
                            else "-",
                    },
                    {
                        "label":
                            "사용 공구",
                        "value":
                            f"{int(result.get('used_tool_count', 0))}종",
                        "sub":
                            f"AI 분석 가능 {len(result.get('supported_tools', []))}종",
                    },
                    {
                        "label":
                            "연속 절삭 구간",
                        "value":
                            f"{int(result.get('segment_count', 0))}개",
                    },
                    {
                        "label":
                            "추정 수집 간격",
                        "value":
                            f"{interval_mean:.2f}초"
                            if interval_mean
                            is not None
                            else "-",
                        "sub":
                            "절삭 Window 기반 추정",
                    },
                ]
            )

            tool_summary = pd.DataFrame(
                result.get(
                    "tool_summary",
                    [],
                )
            )

            if not tool_summary.empty:
                section(
                    "TOOL USAGE",
                    "공구별 사용 현황",
                    "공구별 절삭 데이터량과 대표 운전 수준을 비교합니다.",
                )

                tool_summary[
                    "공구"
                ] = (
                    "Tool "
                    + tool_summary[
                        "tool"
                    ].astype(int).astype(str)
                )

                tool_summary[
                    "절삭 비율(%)"
                ] = (
                    tool_summary[
                        "cutting_ratio"
                    ] * 100
                ).round(1)

                tool_summary[
                    "대표 RPM"
                ] = pd.to_numeric(
                    tool_summary[
                        "rpm_median"
                    ],
                    errors="coerce",
                ).round(0)

                tool_summary[
                    "대표 LOAD"
                ] = pd.to_numeric(
                    tool_summary[
                        "load_median"
                    ],
                    errors="coerce",
                ).round(1)

                left, right = st.columns(
                    [1.2, 1]
                )

                with left:
                    st.dataframe(
                        tool_summary[
                            [
                                "공구",
                                "cutting_rows",
                                "절삭 비율(%)",
                                "대표 RPM",
                                "대표 LOAD",
                            ]
                        ].rename(
                            columns={
                                "cutting_rows":
                                    "절삭 데이터 수",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

                    if (
                        result.get(
                            "analysis_mode"
                        )
                        == "zip"
                    ):
                        st.caption(
                            "ZIP의 대표 RPM/LOAD는 일별 대표값을 절삭 데이터 수로 가중한 참고값입니다."
                        )

                with right:
                    if go is not None:
                        fig = go.Figure(
                            go.Bar(
                                x=tool_summary[
                                    "공구"
                                ],
                                y=tool_summary[
                                    "cutting_rows"
                                ],
                                marker_color=T[
                                    "blue"
                                ],
                                hovertemplate=(
                                    "%{x}<br>"
                                    "절삭 데이터 %{y:,}"
                                    "<extra></extra>"
                                ),
                            )
                        )

                        style_figure(
                            fig,
                            height=340,
                            legend=False,
                        )

                        fig.update_layout(
                            xaxis_title="공구",
                            yaxis_title="절삭 데이터 수",
                        )

                        st.plotly_chart(
                            fig,
                            width="stretch",
                            config=PLOT_CONFIG,
                        )

            # ZIP이면 날짜별 공정 흐름 선택
            segments = pd.DataFrame(
                result.get(
                    "cutting_segments",
                    [],
                )
            )

            if not segments.empty:
                section(
                    "PROCESS TIMELINE",
                    "절삭 작업 흐름",
                    "시간대별로 어떤 공구가 절삭 작업에 사용되었는지 확인합니다.",
                )

                segments[
                    "start_dt"
                ] = pd.to_datetime(
                    segments[
                        "start"
                    ],
                    errors="coerce",
                )

                segments[
                    "end_dt"
                ] = pd.to_datetime(
                    segments[
                        "end"
                    ],
                    errors="coerce",
                )

                segments = segments.dropna(
                    subset=[
                        "start_dt",
                        "end_dt",
                    ]
                )

                timeline_df = segments.copy()

                if (
                    result.get(
                        "analysis_mode"
                    )
                    == "zip"
                    and not segments.empty
                ):
                    available_dates = sorted(
                        segments[
                            "start_dt"
                        ].dt.date.unique()
                    )

                    selected_date = (
                        st.selectbox(
                            "Timeline 날짜",
                            options=available_dates,
                            format_func=lambda d:
                                d.strftime(
                                    "%Y-%m-%d"
                                ),
                        )
                    )

                    timeline_df = (
                        segments[
                            segments[
                                "start_dt"
                            ].dt.date
                            == selected_date
                        ]
                        .copy()
                    )

                if (
                    go is not None
                    and not timeline_df.empty
                ):
                    fig = go.Figure()

                    for _, row in timeline_df.iterrows():
                        tool = int(
                            row[
                                "tool"
                            ]
                        )

                        duration_ms = (
                            row[
                                "end_dt"
                            ]
                            - row[
                                "start_dt"
                            ]
                        ).total_seconds() * 1000

                        fig.add_trace(
                            go.Bar(
                                x=[
                                    duration_ms
                                ],
                                y=[
                                    f"Tool {tool}"
                                ],
                                base=[
                                    row[
                                        "start_dt"
                                    ]
                                ],
                                orientation="h",
                                name=f"Tool {tool}",
                                hovertemplate=(
                                    f"Tool {tool}<br>"
                                    f"{row['start_dt']:%H:%M:%S}"
                                    " - "
                                    f"{row['end_dt']:%H:%M:%S}"
                                    "<extra></extra>"
                                ),
                                showlegend=False,
                                marker_color=T[
                                    "blue"
                                ],
                            )
                        )

                    fig.update_xaxes(
                        type="date",
                        tickformat="%H:%M",
                    )

                    style_figure(
                        fig,
                        height=max(
                            230,
                            70
                            * timeline_df[
                                "tool"
                            ].nunique(),
                        ),
                        legend=False,
                    )

                    fig.update_layout(
                        barmode="overlay",
                        xaxis_title="시간",
                        yaxis_title="공구",
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch",
                        config=PLOT_CONFIG,
                    )

                with st.expander(
                    "절삭 구간 목록 보기"
                ):
                    segment_display = (
                        timeline_df[
                            [
                                "start",
                                "end",
                                "tool",
                                "duration_minutes",
                                "cutting_rows",
                            ]
                        ]
                        .copy()
                    )

                    segment_display[
                        "tool"
                    ] = (
                        "Tool "
                        + segment_display[
                            "tool"
                        ].astype(int).astype(str)
                    )

                    st.dataframe(
                        segment_display.rename(
                            columns={
                                "start":
                                    "시작",
                                "end":
                                    "종료",
                                "tool":
                                    "공구",
                                "duration_minutes":
                                    "지속 시간(분)",
                                "cutting_rows":
                                    "절삭 데이터 수",
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            section(
                "DATA COVERAGE",
                "AI 분석 가능 범위",
                "실제 모델 분석에 사용된 Window와 제외된 Window를 구분합니다.",
            )

            kpi_row(
                [
                    {
                        "label":
                            "AI 분석 가능",
                        "value":
                            f"{int(result.get('supported_windows', 0)):,}",
                        "dot":
                            T[
                                "green"
                            ],
                        "sub":
                            "정상 Baseline + 충분한 표본",
                    },
                    {
                        "label":
                            "정상 기준 없음",
                        "value":
                            f"{int(result.get('unsupported_windows', 0)):,}",
                        "dot":
                            T[
                                "yellow"
                            ],
                        "sub":
                            "Train 정상 Baseline 없음",
                    },
                    {
                        "label":
                            "측정값 부족",
                        "value":
                            f"{int(result.get('insufficient_windows', 0)):,}",
                        "dot":
                            T[
                                "yellow"
                            ],
                        "sub":
                            "1분 Window 표본 수 부족",
                    },
                ]
            )

            st.caption(
                "실제 절삭 데이터 = STATUS_RUN & Spindle RPM > 0 & Feed Rate > 0. "
                "수집 간격은 절삭 Window에서 추정한 참고값입니다."
            )

        # =====================================================
        # ④ 분석 방법과 신뢰성
        # =====================================================

        with tab4:

            section(
                "METHOD",
                "이 AI는 어떤 기준으로 분석하나요?",
                "실제 고장 라벨 부족을 보완하기 위해 물리적으로 설명 가능한 가상 이상 시나리오를 만든 뒤 지도학습 모델을 비교했습니다.",
            )

            st.markdown(
                """
                <div class="panel">
                <div class="panel-title">학습 구성</div>
                <div class="panel-text">
                정상 절삭 데이터에 <b>LOAD 증가</b>, <b>RPM 불안정</b>,
                <b>명령-실제 RPM 오차 증가</b>의 세 가지 가상 이상을 주입했습니다.
                정상 Window는 0, 가상 이상 Window는 1로 라벨링했습니다.
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            model_table = pd.DataFrame(
                [
                    {
                        "모델":
                            "Logistic Regression",
                        "Validation Accuracy":
                            0.986486,
                        "F1":
                            0.990291,
                        "PR-AUC":
                            0.999838,
                        "Log Loss":
                            0.053698,
                        "Brier":
                            0.012469,
                    },
                    {
                        "모델":
                            "Random Forest",
                        "Validation Accuracy":
                            1.0,
                        "F1":
                            1.0,
                        "PR-AUC":
                            1.0,
                        "Log Loss":
                            0.006400,
                        "Brier":
                            0.000513,
                    },
                    {
                        "모델":
                            "XGBoost",
                        "Validation Accuracy":
                            1.0,
                        "F1":
                            1.0,
                        "PR-AUC":
                            1.0,
                        "Log Loss":
                            0.002670,
                        "Brier":
                            0.000066,
                    },
                ]
            )

            st.dataframe(
                model_table,
                width="stretch",
                hide_index=True,
            )

            kpi_row(
                [
                    {
                        "label":
                            "최종 모델",
                        "value":
                            "XGBoost",
                        "sub":
                            "RF와 주요 지표 동률, Log Loss/Brier 우수",
                    },
                    {
                        "label":
                            "Test Accuracy",
                        "value":
                            "99.64%",
                        "sub":
                            "가상 이상 시나리오 기반 Test",
                    },
                    {
                        "label":
                            "Test F1",
                        "value":
                            "99.74%",
                        "sub":
                            "실제 고장 정확도가 아님",
                    },
                    {
                        "label":
                            "입력 Feature",
                        "value":
                            "14개",
                        "sub":
                            "1분 Window 정형 Feature",
                    },
                ]
            )

            st.markdown(
                """
                <div class="warning-panel">
                <div class="panel-title">해석 시 주의</div>
                <div class="panel-text">
                XGBoost의 높은 Test 성능은 <b>가상 이상 시나리오를 구분한 결과</b>입니다.
                실제 현장 고장을 99.64% 맞힌다는 의미가 아닙니다.
                실제 31일 로그 검증에서는 공구 교체·공정 전환 직전에도 높은 Score가
                나타날 수 있음을 확인했습니다.
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="good-panel">
                <div class="panel-title">비지도학습과의 비교 계획</div>
                <div class="panel-text">
                실제 고장 라벨이 부족한 데이터 특성을 고려해 비지도 이상 탐지 방식도 별도로 구현합니다.
                지도학습은 “정의한 가상 이상과 얼마나 비슷한가”, 비지도학습은
                “평소 정상 분포에서 얼마나 벗어났는가”를 본다는 차이가 있습니다.
                두 결과를 실제 기업 로그에서 비교한 뒤 현장 적용에 더 적합한 방식을 제안할 예정입니다.
                </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="small-note" style="margin-top:28px;padding-top:14px;border-top:1px solid rgba(255,255,255,.08)">
            본 화면은 설비 상태 판단을 보조하기 위한 지도학습 기반 PoC입니다.
            실제 고장 여부와 정비 시점은 공정 정보, 정비 이력, 현장 점검 결과와 함께 판단해야 합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )


    elif uploaded_file is not None:
        st.info(
            "파일을 선택했습니다. '분석 시작' 버튼을 눌러주세요."
        )
