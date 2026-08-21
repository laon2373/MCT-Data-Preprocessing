from pathlib import Path
import io
import json
import math
import os
import shutil
import sys
import tempfile
import zipfile

import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask


# =========================================================
# 프로젝트 경로 설정
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DASHBOARD_DIR = PROJECT_ROOT / "mct_poc_dashboard_common" / "dashboard"

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))


from src.inference_pipeline import (
    load_model_artifacts,
    run_inference,
)

from m2_upload_pipeline import analyze_uploaded_file as analyze_m2_uploaded


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="MCT 예지보전 통합 AI API",
    version="1.0.0",
)


# =========================================================
# Helper
# =========================================================

def _to_float(value):
    """NaN/inf를 제외한 JSON-safe float."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _window_detail_summary(
    result_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    cutting_rows: int,
):
    """
    추론 결과 Window를 이용해 UI용 데이터 요약을 만든다.
    모델 입력/예측 로직은 변경하지 않는다.
    """

    frames = []

    if result_df is not None and not result_df.empty:
        frames.append(result_df.copy())

    if skipped_df is not None and not skipped_df.empty:
        frames.append(skipped_df.copy())

    if not frames:
        return {
            "used_tool_count": 0,
            "segment_count": 0,
            "sampling_interval_mean": None,
            "file_start": None,
            "file_end": None,
            "tool_summary": [],
            "cutting_segments": [],
        }

    all_windows = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    for column in ["window_start", "window_end"]:
        if column in all_windows.columns:
            all_windows[column] = pd.to_datetime(
                all_windows[column],
                errors="coerce",
            )

    if "current_tool" in all_windows.columns:
        all_windows["current_tool"] = pd.to_numeric(
            all_windows["current_tool"],
            errors="coerce",
        )

    if "sample_count" in all_windows.columns:
        all_windows["sample_count"] = pd.to_numeric(
            all_windows["sample_count"],
            errors="coerce",
        ).fillna(0)
    else:
        all_windows["sample_count"] = 0

    all_windows = all_windows.dropna(
        subset=["window_start", "window_end", "current_tool"]
    ).copy()

    if all_windows.empty:
        return {
            "used_tool_count": 0,
            "segment_count": 0,
            "sampling_interval_mean": None,
            "file_start": None,
            "file_end": None,
            "tool_summary": [],
            "cutting_segments": [],
        }

    all_windows = all_windows.sort_values(
        ["window_start", "window_end"]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # 기본 정보
    # -----------------------------------------------------

    used_tool_count = int(
        all_windows["current_tool"].nunique()
    )

    file_start = str(
        all_windows["window_start"].min()
    )

    file_end = str(
        all_windows["window_end"].max()
    )

    # -----------------------------------------------------
    # 절삭 Window 안에서의 평균 수집 간격 추정
    # 각 Window의 (끝-시작)/(표본수-1)을 전체 관측 간격으로 가중
    # -----------------------------------------------------

    valid_interval = all_windows[
        all_windows["sample_count"] > 1
    ].copy()

    sampling_interval_mean = None

    if not valid_interval.empty:

        valid_interval["duration_sec"] = (
            valid_interval["window_end"]
            - valid_interval["window_start"]
        ).dt.total_seconds()

        valid_interval["interval_n"] = (
            valid_interval["sample_count"]
            - 1
        )

        valid_interval = valid_interval[
            (valid_interval["duration_sec"] >= 0)
            & (valid_interval["interval_n"] > 0)
        ]

        if not valid_interval.empty:

            total_interval_n = (
                valid_interval["interval_n"].sum()
            )

            if total_interval_n > 0:
                sampling_interval_mean = _to_float(
                    valid_interval["duration_sec"].sum()
                    / total_interval_n
                )

    # -----------------------------------------------------
    # 공구별 요약
    # -----------------------------------------------------

    tool_rows = (
        all_windows
        .groupby("current_tool", as_index=False)["sample_count"]
        .sum()
        .rename(columns={"sample_count": "cutting_rows"})
    )

    # 대표 RPM/LOAD는 Feature가 존재하는 result Window에서 계산
    if (
        result_df is not None
        and not result_df.empty
        and "current_tool" in result_df.columns
    ):
        feature_source = result_df.copy()

        feature_source["current_tool"] = pd.to_numeric(
            feature_source["current_tool"],
            errors="coerce",
        )

        agg_map = {}

        if "rpm_mean" in feature_source.columns:
            agg_map["rpm_mean"] = "median"

        if "load_mean" in feature_source.columns:
            agg_map["load_mean"] = "median"

        if agg_map:
            feature_summary = (
                feature_source
                .dropna(subset=["current_tool"])
                .groupby("current_tool", as_index=False)
                .agg(agg_map)
            )

            tool_rows = tool_rows.merge(
                feature_summary,
                on="current_tool",
                how="left",
            )

    tool_summary = []

    denominator = (
        cutting_rows
        if cutting_rows > 0
        else max(int(tool_rows["cutting_rows"].sum()), 1)
    )

    for _, row in tool_rows.sort_values(
        "cutting_rows",
        ascending=False,
    ).iterrows():

        item = {
            "tool": int(row["current_tool"]),
            "cutting_rows": int(row["cutting_rows"]),
            "cutting_ratio": _to_float(
                row["cutting_rows"] / denominator
            ),
            "rpm_median": _to_float(
                row.get("rpm_mean")
            ),
            "load_median": _to_float(
                row.get("load_mean")
            ),
        }

        tool_summary.append(item)

    # -----------------------------------------------------
    # 연속 절삭 Segment 요약
    #
    # 1순위: inference 결과에 segment id가 있으면 그대로 사용
    # 2순위: window_id reset + 공구 변경 + 15초 초과 gap으로 재구성
    # -----------------------------------------------------

    segment_column = None

    for candidate in [
        "segment_id",
        "cutting_segment_id",
        "source_segment_id",
        "segment_no",
    ]:
        if candidate in all_windows.columns:
            segment_column = candidate
            break

    if segment_column is not None:

        all_windows["_segment_key"] = (
            all_windows[segment_column]
            .astype(str)
        )

    else:

        previous_end = all_windows["window_end"].shift(1)
        previous_tool = all_windows["current_tool"].shift(1)

        gap_seconds = (
            all_windows["window_start"]
            - previous_end
        ).dt.total_seconds()

        new_segment = (
            previous_end.isna()
            | (
                all_windows["current_tool"]
                != previous_tool
            )
            | (gap_seconds > 15)
            | (
                all_windows["window_start"].dt.date
                != previous_end.dt.date
            )
        )

        # window_id가 segment마다 0부터 다시 시작한다면
        # reset도 segment 경계로 사용
        if "window_id" in all_windows.columns:

            window_ids = pd.to_numeric(
                all_windows["window_id"],
                errors="coerce",
            )

            previous_window_id = window_ids.shift(1)

            reset = (
                window_ids.notna()
                & previous_window_id.notna()
                & (window_ids <= previous_window_id)
            )

            new_segment = new_segment | reset

        all_windows["_segment_key"] = (
            new_segment.cumsum().astype(str)
        )

    segment_rows = []

    for _, group in all_windows.groupby(
        "_segment_key",
        sort=False,
    ):

        start = group["window_start"].min()
        end = group["window_end"].max()
        tool = int(group["current_tool"].mode().iloc[0])

        duration_minutes = (
            end - start
        ).total_seconds() / 60

        segment_rows.append(
            {
                "tool": tool,
                "start": str(start),
                "end": str(end),
                "duration_minutes": _to_float(
                    duration_minutes
                ),
                "cutting_rows": int(
                    group["sample_count"].sum()
                ),
            }
        )

    return {
        "used_tool_count": used_tool_count,
        "segment_count": len(segment_rows),
        "sampling_interval_mean": sampling_interval_mean,
        "file_start": file_start,
        "file_end": file_end,
        "tool_summary": tool_summary,
        "cutting_segments": segment_rows,
    }


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "MCT Predictive Maintenance Unified API is running",
        "models": {
            "m2": "ready",
            "xgboost": "ready",
        },
        "endpoints": {
            "m2": "/predict/m2",
            "xgboost": "/predict",
        },
    }



# =========================================================
# M2* Batch 추론
# =========================================================

M2_RESPONSE_COLUMNS = [
    "ts", "status", "state", "tool", "feed",
    "srate", "sload", "rpm", "band",
    "ms", "ss", "zload", "ewma",
    "novel", "f_pt", "f_ew", "f_m3",
]


def _zip_write_dataframe(
    archive: zipfile.ZipFile,
    filename: str,
    df: pd.DataFrame,
):
    """큰 DataFrame을 ZIP 안의 CSV로 스트리밍 저장."""
    with archive.open(filename, "w") as raw:
        with io.TextIOWrapper(
            raw,
            encoding="utf-8",
            newline="",
        ) as text:
            df.to_csv(text, index=False)


def _build_m2_response_zip(result: dict) -> Path:
    fd, temp_name = tempfile.mkstemp(suffix=".m2result.zip")
    os.close(fd)
    temp_path = Path(temp_name)

    df = result["df"].copy()

    keep = [
        column
        for column in M2_RESPONSE_COLUMNS
        if column in df.columns
    ]

    with zipfile.ZipFile(
        temp_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        _zip_write_dataframe(
            archive,
            "df.csv",
            df[keep],
        )
        _zip_write_dataframe(
            archive,
            "alarms.csv",
            result["alarms"],
        )
        _zip_write_dataframe(
            archive,
            "novel.csv",
            result["novel"],
        )
        _zip_write_dataframe(
            archive,
            "events.csv",
            result["events"],
        )
        _zip_write_dataframe(
            archive,
            "daily.csv",
            result["daily"],
        )

        archive.writestr(
            "meta.json",
            json.dumps(
                result.get("meta", {}),
                ensure_ascii=False,
                default=str,
            ),
        )
        archive.writestr(
            "summary.json",
            json.dumps(
                result.get("summary", {}),
                ensure_ascii=False,
                default=str,
            ),
        )
        archive.writestr(
            "file_infos.json",
            json.dumps(
                result.get("file_infos", []),
                ensure_ascii=False,
                default=str,
            ),
        )

    return temp_path


@app.post("/predict/m2")
def predict_m2(
    file: UploadFile = File(...)
):
    """
    .log / .zip을 받아 M2* batch 분석을 수행한다.

    Streamlit → FastAPI → 공통 전처리 → M2* → 결과 ZIP → Streamlit
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="파일명이 없습니다.",
        )

    suffix = Path(file.filename).suffix.lower()

    if suffix not in {".log", ".zip"}:
        raise HTTPException(
            status_code=400,
            detail="M2* 분석은 .log 또는 .zip만 지원합니다.",
        )

    try:
        file_bytes = file.file.read()

        if not file_bytes:
            raise HTTPException(
                status_code=400,
                detail="업로드 파일이 비어 있습니다.",
            )

        result = analyze_m2_uploaded(
            file.filename,
            file_bytes,
        )

        result_zip = _build_m2_response_zip(result)

        return FileResponse(
            result_zip,
            media_type="application/zip",
            filename="m2_analysis_result.zip",
            background=BackgroundTask(
                lambda: result_zip.unlink(missing_ok=True)
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"M2* 분석 중 오류가 발생했습니다: {exc}",
        )

    finally:
        file.file.close()


# =========================================================
# MCT LOG 추론
# =========================================================

@app.post("/predict")
def predict_log(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # 1. 파일 확인
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="파일명이 없습니다.",
        )

    suffix = Path(
        file.filename
    ).suffix.lower()

    if suffix != ".log":
        raise HTTPException(
            status_code=400,
            detail=".log 파일만 업로드할 수 있습니다.",
        )

    temp_path = None

    try:

        # -------------------------------------------------
        # 2. 업로드 파일 임시 저장
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".log",
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = Path(
                temp_file.name
            )

        # -------------------------------------------------
        # 3. 기존 추론 파이프라인 실행
        # -------------------------------------------------

        output = run_inference(
            temp_path
        )

        result_df = output[
            "results"
        ].copy()

        skipped_df = output[
            "skipped_windows"
        ].copy()

        # -------------------------------------------------
        # 4. 지원 가능한 Window
        # -------------------------------------------------

        if result_df.empty:

            supported_df = result_df

        else:

            supported_df = result_df[
                result_df["supported"]
            ].copy()

        # -------------------------------------------------
        # 5. Anomaly Score 통계
        # -------------------------------------------------

        if (
            not supported_df.empty
            and
            "anomaly_score"
            in supported_df.columns
        ):

            scores = (
                supported_df[
                    "anomaly_score"
                ]
                .dropna()
            )

        else:

            scores = []

        if len(scores) > 0:

            score_mean = float(
                scores.mean()
            )

            score_max = float(
                scores.max()
            )

            max_index = (
                supported_df[
                    "anomaly_score"
                ]
                .idxmax()
            )

            max_row = supported_df.loc[
                max_index
            ]

            highest_window = {
                "window_start": str(
                    max_row[
                        "window_start"
                    ]
                ),
                "window_end": str(
                    max_row[
                        "window_end"
                    ]
                ),
                "current_tool": int(
                    max_row[
                        "current_tool"
                    ]
                ),
                "sample_count": int(
                    max_row[
                        "sample_count"
                    ]
                ),
                "anomaly_score": float(
                    max_row[
                        "anomaly_score"
                    ]
                ),
                "load_vs_normal": float(
                    max_row[
                        "load_vs_normal"
                    ]
                ),
                "rpm_std_vs_normal": float(
                    max_row[
                        "rpm_std_vs_normal"
                    ]
                ),
                "rpm_error_vs_normal": float(
                    max_row[
                        "rpm_error_vs_normal"
                    ]
                ),
            }

            # ---------------------------------------------
            # 최고점 Window의 XGBoost Feature 기여도
            # ---------------------------------------------

            explain_model, feature_columns, _ = (
                load_model_artifacts()
            )

            explain_X = (
                max_row[
                    feature_columns
                ]
                .astype(float)
                .to_frame()
                .T
            )

            dmatrix = xgb.DMatrix(
                explain_X,
                feature_names=feature_columns,
            )

            contributions = (
                explain_model
                .get_booster()
                .predict(
                    dmatrix,
                    pred_contribs=True,
                    iteration_range=(
                        0,
                        explain_model.best_iteration + 1,
                    ),
                )[0]
            )

            feature_contributions = []

            for feature_name, contribution in zip(
                feature_columns,
                contributions[:-1],
            ):

                feature_contributions.append(
                    {
                        "feature": feature_name,
                        "contribution": float(
                            contribution
                        ),
                    }
                )

            positive_contributions = [
                item
                for item in feature_contributions
                if item["contribution"] > 0
            ]

            positive_contributions = sorted(
                positive_contributions,
                key=lambda x: x["contribution"],
                reverse=True,
            )

            top_contributions = (
                positive_contributions[:3]
            )

            # ---------------------------------------------
            # Anomaly Score 상위 10개 Window
            # ---------------------------------------------

            top_df = (
                supported_df
                .sort_values(
                    "anomaly_score",
                    ascending=False,
                )
                .head(10)
            )

            top_windows = []

            for _, row in top_df.iterrows():

                top_windows.append(
                    {
                        "window_start": str(
                            row["window_start"]
                        ),
                        "window_end": str(
                            row["window_end"]
                        ),
                        "current_tool": int(
                            row["current_tool"]
                        ),
                        "sample_count": int(
                            row["sample_count"]
                        ),
                        "anomaly_score": float(
                            row["anomaly_score"]
                        ),
                        "load_vs_normal": float(
                            row["load_vs_normal"]
                        ),
                        "rpm_std_vs_normal": float(
                            row["rpm_std_vs_normal"]
                        ),
                        "rpm_error_vs_normal": float(
                            row["rpm_error_vs_normal"]
                        ),
                    }
                )

            # ---------------------------------------------
            # 시간 흐름 그래프용 Score Series
            # ---------------------------------------------

            score_df = (
                supported_df[
                    [
                        "window_start",
                        "window_end",
                        "current_tool",
                        "anomaly_score",
                    ]
                ]
                .dropna(subset=["anomaly_score"])
                .sort_values("window_start")
                .copy()
            )

            # 매우 큰 파일에서도 API 응답이 과도하게 커지지 않도록
            # 차트용 포인트는 최대 5,000개로 제한
            if len(score_df) > 5000:
                step = math.ceil(
                    len(score_df) / 5000
                )
                score_df = score_df.iloc[
                    ::step
                ].copy()

            score_series = []

            for _, row in score_df.iterrows():
                score_series.append(
                    {
                        "window_start": str(
                            row["window_start"]
                        ),
                        "window_end": str(
                            row["window_end"]
                        ),
                        "current_tool": int(
                            row["current_tool"]
                        ),
                        "anomaly_score": float(
                            row["anomaly_score"]
                        ),
                    }
                )

        else:

            score_mean = None
            score_max = None
            highest_window = None
            top_windows = []
            top_contributions = []
            score_series = []

        # -------------------------------------------------
        # 6. 업로드 데이터 분석용 요약
        # -------------------------------------------------

        clean_rows = int(
            output[
                "clean_rows"
            ]
        )

        cutting_rows = int(
            output[
                "cutting_rows"
            ]
        )

        cutting_ratio = (
            cutting_rows / clean_rows
            if clean_rows > 0
            else 0.0
        )

        detail = _window_detail_summary(
            result_df=result_df,
            skipped_df=skipped_df,
            cutting_rows=cutting_rows,
        )

        # -------------------------------------------------
        # 7. Streamlit 상세 조회용 Window / 공구 정보
        # -------------------------------------------------
        #
        # 기존 API는 score_series, 최고점, Top10만 반환하고
        # 모든 지원 Window(window_details)는 반환하지 않았다.
        # 그 때문에 실제로 XGBoost Window가 존재해도
        # Streamlit의 "확인이 필요한 시간대" 탭은 빈 목록으로 보였다.
        # 모델 계산은 변경하지 않고 API 응답만 보완한다.

        window_details = []

        if not supported_df.empty:
            detail_columns = [
                "window_start",
                "window_end",
                "current_tool",
                "sample_count",
                "anomaly_score",
                "load_vs_normal",
                "rpm_std_vs_normal",
                "rpm_error_vs_normal",
            ]

            available_detail_columns = [
                column
                for column in detail_columns
                if column in supported_df.columns
            ]

            detail_df = (
                supported_df[
                    available_detail_columns
                ]
                .dropna(subset=["anomaly_score"])
                .sort_values("window_start")
                .copy()
            )

            for _, row in detail_df.iterrows():
                window_details.append(
                    {
                        "window_start": str(
                            row["window_start"]
                        ),
                        "window_end": str(
                            row["window_end"]
                        ),
                        "current_tool": int(
                            row["current_tool"]
                        ),
                        "sample_count": int(
                            row["sample_count"]
                        ),
                        "anomaly_score": _to_float(
                            row["anomaly_score"]
                        ),
                        "load_vs_normal": _to_float(
                            row.get(
                                "load_vs_normal"
                            )
                        ),
                        "rpm_std_vs_normal": _to_float(
                            row.get(
                                "rpm_std_vs_normal"
                            )
                        ),
                        "rpm_error_vs_normal": _to_float(
                            row.get(
                                "rpm_error_vs_normal"
                            )
                        ),
                    }
                )

        # 전체 사용 공구
        used_tool_frames = []

        if result_df is not None and not result_df.empty:
            used_tool_frames.append(
                result_df[["current_tool"]].copy()
            )

        if skipped_df is not None and not skipped_df.empty:
            if "current_tool" in skipped_df.columns:
                used_tool_frames.append(
                    skipped_df[["current_tool"]].copy()
                )

        if used_tool_frames:
            used_tool_series = pd.concat(
                used_tool_frames,
                ignore_index=True,
            )["current_tool"]

            used_tools = sorted(
                int(tool)
                for tool in pd.to_numeric(
                    used_tool_series,
                    errors="coerce",
                )
                .dropna()
                .unique()
            )
        else:
            used_tools = []

        # 정상 Baseline이 있어 실제 XGBoost 점수를 계산한 공구
        if (
            not supported_df.empty
            and "current_tool" in supported_df.columns
        ):
            supported_tools = sorted(
                int(tool)
                for tool in pd.to_numeric(
                    supported_df["current_tool"],
                    errors="coerce",
                )
                .dropna()
                .unique()
            )
        else:
            supported_tools = []

        # 실제 절삭 Window는 있으나 Train Normal Baseline이 없는 공구
        unsupported_tools = sorted(
            set(used_tools)
            - set(supported_tools)
        )

        # 공구별 분석 범위 요약
        tool_analysis_summary = []

        for tool in used_tools:
            analyzed_windows = 0
            baseline_missing_windows = 0
            insufficient_windows = 0

            if (
                not result_df.empty
                and "current_tool" in result_df.columns
            ):
                tool_result = result_df[
                    pd.to_numeric(
                        result_df["current_tool"],
                        errors="coerce",
                    )
                    == tool
                ]

                if "supported" in tool_result.columns:
                    analyzed_windows = int(
                        tool_result["supported"]
                        .fillna(False)
                        .astype(bool)
                        .sum()
                    )
                    baseline_missing_windows = int(
                        (
                            ~tool_result["supported"]
                            .fillna(False)
                            .astype(bool)
                        ).sum()
                    )

            if (
                not skipped_df.empty
                and "current_tool" in skipped_df.columns
            ):
                insufficient_windows = int(
                    (
                        pd.to_numeric(
                            skipped_df["current_tool"],
                            errors="coerce",
                        )
                        == tool
                    ).sum()
                )

            tool_analysis_summary.append(
                {
                    "tool": int(tool),
                    "analyzed_windows": analyzed_windows,
                    "baseline_missing_windows": baseline_missing_windows,
                    "insufficient_windows": insufficient_windows,
                    "analysis_supported": analyzed_windows > 0,
                }
            )

        # -------------------------------------------------
        # 8. API 결과 반환
        # -------------------------------------------------

        return {
            "filename": file.filename,

            "parsed_rows": int(
                output[
                    "written_rows"
                ]
            ),

            "clean_rows": clean_rows,

            "cutting_rows": cutting_rows,

            "cutting_ratio": float(
                cutting_ratio
            ),

            "result_windows": int(
                len(
                    result_df
                )
            ),

            "supported_windows": int(
                len(
                    supported_df
                )
            ),

            "unsupported_windows": int(
                len(
                    result_df
                )
                -
                len(
                    supported_df
                )
            ),

            "insufficient_windows": int(
                len(
                    skipped_df
                )
            ),

            "score_mean": score_mean,

            "score_max": score_max,

            "highest_window": highest_window,

            "top_windows": top_windows,

            "top_contributions": top_contributions,

            "score_series": score_series,

            # Streamlit의 전체 1분 Window 상세 조회용
            "window_details": window_details,

            # 사용/지원 공구 범위
            "used_tools": used_tools,
            "supported_tools": supported_tools,
            "unsupported_tools": unsupported_tools,
            "tool_analysis_summary": tool_analysis_summary,

            "used_tool_count": detail[
                "used_tool_count"
            ],

            "segment_count": detail[
                "segment_count"
            ],

            "sampling_interval_mean": detail[
                "sampling_interval_mean"
            ],

            "file_start": detail[
                "file_start"
            ],

            "file_end": detail[
                "file_end"
            ],

            "tool_summary": detail[
                "tool_summary"
            ],

            "cutting_segments": detail[
                "cutting_segments"
            ],
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "분석 중 오류가 발생했습니다: "
                f"{str(e)}"
            ),
        )

    finally:

        if (
            temp_path is not None
            and
            temp_path.exists()
        ):
            temp_path.unlink()

        file.file.close()
