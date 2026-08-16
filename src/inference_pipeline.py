from pathlib import Path
from io import StringIO
import csv
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


# 직접 실행할 때와 다른 파일에서 import할 때
# 둘 다 동작하도록 처리
try:
    from .parse_mct_log import FIELDNAMES, parse_log_file
    from .build_clean_dataset import INTERVAL_THRESHOLD_SECONDS
except ImportError:
    from parse_mct_log import FIELDNAMES, parse_log_file
    from build_clean_dataset import INTERVAL_THRESHOLD_SECONDS


# =========================================================
# 프로젝트 경로
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "xgboost_model.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
BASELINE_PATH = MODELS_DIR / "train_normal_baseline.csv"


# =========================================================
# 추론 규칙
# =========================================================

# 학습 데이터의 1분 Window 최소 표본 수
MIN_WINDOW_SAMPLES = 9

# 상대 Feature 계산 시 0으로 나누는 것을 방지하는 안전 하한
LOAD_EPS = 1.0
RPM_STD_EPS = 1.0
RPM_ERROR_EPS = 1.0


# =========================================================
# 1. 모델 산출물 로드
# =========================================================

def load_model_artifacts():
    """
    추론에 필요한 산출물을 불러온다.

    1. 학습된 XGBoost 모델
    2. Feature 14개의 이름과 순서
    3. 공구별 Train Normal Baseline
    """

    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    with open(
        FEATURE_COLUMNS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        feature_info = json.load(f)

    feature_columns = feature_info["feature_columns"]

    baseline = pd.read_csv(
        BASELINE_PATH,
        encoding="utf-8-sig",
    )

    return model, feature_columns, baseline


# =========================================================
# 2. 원본 LOG Parsing
# =========================================================

def parse_raw_log(log_path):
    """
    기업 원본 .log 파일 하나를
    기존 parse_mct_log.py와 동일한 규칙으로 Parsing한다.
    """

    log_path = Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(
            f"원본 로그 파일을 찾을 수 없습니다: {log_path}"
        )

    buffer = StringIO()

    writer = csv.DictWriter(
        buffer,
        fieldnames=FIELDNAMES,
        extrasaction="ignore",
    )

    writer.writeheader()

    written_rows, skipped_lines = parse_log_file(
        log_path,
        writer,
    )

    buffer.seek(0)

    df = pd.read_csv(buffer)

    return df, written_rows, skipped_lines


# =========================================================
# 3. Cleaning
# =========================================================

def prepare_clean_data(df):
    """
    기존 build_clean_dataset.py와 동일한 규칙으로
    Parsing 데이터를 분석용 Clean 데이터로 변환한다.
    """

    clean_df = df.copy()

    original_rows = len(clean_df)

    clean_df["timestamp"] = pd.to_datetime(
        clean_df["timestamp"],
        format="%Y-%m-%d %H:%M:%S,%f",
        errors="raise",
    )

    clean_df = clean_df.sort_values(
        ["machine", "timestamp"]
    ).reset_index(drop=True)

    # 직전 측정값과의 수집 간격
    clean_df["interval_seconds"] = (
        clean_df.groupby("machine")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # 좌표 mm 변환
    clean_df["x_axis_mm"] = (
        clean_df["x_axis"] / 10000
    )
    clean_df["y_axis_mm"] = (
        clean_df["y_axis"] / 10000
    )
    clean_df["z_axis_mm"] = (
        clean_df["z_axis"] / 10000
    )

    # 상태 Flag
    clean_df["is_alarm"] = (
        clean_df["alarm"].ne(0)
    )

    clean_df["is_emergency"] = (
        clean_df["emergency"].eq(1)
    )

    clean_df["is_reset"] = (
        clean_df["emergency"].eq(2)
    )

    clean_df["is_spindle_overload"] = (
        clean_df["spindle_load"].gt(100)
    )

    clean_df["is_abnormal_interval"] = (
        clean_df["interval_seconds"]
        .gt(INTERVAL_THRESHOLD_SECONDS)
    )

    clean_df["has_event"] = clean_df[
        [
            "is_alarm",
            "is_emergency",
            "is_reset",
            "is_spindle_overload",
            "is_abnormal_interval",
        ]
    ].any(axis=1)

    # 기존 mct_clean.csv와 같은 timestamp 형식
    clean_df["timestamp"] = (
        clean_df["timestamp"]
        .dt.strftime("%Y-%m-%d %H:%M:%S,%f")
        .str[:-3]
    )

    if len(clean_df) != original_rows:
        raise RuntimeError(
            "Cleaning 과정에서 행 수가 변경되었습니다."
        )

    return clean_df


# =========================================================
# 4. 실제 절삭 Segment 생성
# =========================================================

def extract_cutting_segments(clean_df):
    """
    실제 절삭 중인 데이터를 추출한 뒤
    연속 절삭 Segment로 분리한다.

    새 Segment 조건:
    1. 공구 변경
    2. 이전 절삭 행과 시간 차이 > 15초
    3. 중간에 비절삭 행 존재
    4. 날짜 변경
    """

    df = clean_df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S,%f",
        errors="raise",
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    # 실제 절삭 조건
    df["is_cutting"] = (
        df["status"].eq("STATUS_RUN")
        & df["spindle_rpm"].gt(0)
        & df["feed_rate"].gt(0)
    )

    cutting_df = df[
        df["is_cutting"]
    ].copy()

    if cutting_df.empty:
        return cutting_df

    cutting_df["date"] = (
        cutting_df["timestamp"].dt.date
    )

    gap_seconds = (
        cutting_df["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    tool_change = (
        cutting_df["current_tool"]
        .ne(
            cutting_df["current_tool"].shift()
        )
    )

    date_change = (
        cutting_df["date"]
        .ne(
            cutting_df["date"].shift()
        )
    )

    # 원래 행 번호가 연속되지 않으면
    # 중간에 비절삭 행이 존재했다는 뜻
    index_break = (
        cutting_df.index
        .to_series()
        .diff()
        .ne(1)
    )

    new_segment = (
        gap_seconds.isna()
        | gap_seconds.gt(
            INTERVAL_THRESHOLD_SECONDS
        )
        | tool_change
        | date_change
        | index_break
    )

    cutting_df["source_segment_id"] = (
        new_segment.cumsum()
    )

    return cutting_df


# =========================================================
# 5. 1분 Window
# =========================================================

def build_one_minute_windows(segmented_df):
    """
    각 연속 절삭 Segment를
    Segment 시작 기준 60초 Window로 나눈다.
    """

    df = segmented_df.copy()

    segment_start = (
        df.groupby(
            "source_segment_id"
        )["timestamp"]
        .transform("min")
    )

    df["elapsed_s"] = (
        df["timestamp"]
        - segment_start
    ).dt.total_seconds()

    df["window_id"] = np.floor(
        df["elapsed_s"] / 60
    ).astype(int)

    window_table = (
        df.groupby(
            [
                "source_segment_id",
                "window_id",
            ]
        )
        .agg(
            window_start=(
                "timestamp",
                "min",
            ),
            window_end=(
                "timestamp",
                "max",
            ),
            sample_count=(
                "timestamp",
                "size",
            ),
            current_tool=(
                "current_tool",
                "first",
            ),
        )
        .reset_index()
    )

    return df, window_table


# =========================================================
# 6. 표본 부족 Window 분리
# =========================================================

def filter_valid_windows(
    rows_with_window,
    window_table,
    min_samples=MIN_WINDOW_SAMPLES,
):
    """
    학습 때의 최소 표본 수인 9개 이상인
    Window만 모델 Feature 계산 대상으로 사용한다.
    """

    valid_window_table = window_table[
        window_table["sample_count"]
        >= min_samples
    ].copy()

    skipped_window_table = window_table[
        window_table["sample_count"]
        < min_samples
    ].copy()

    skipped_window_table[
        "supported"
    ] = False

    skipped_window_table[
        "reason"
    ] = "insufficient_samples"

    valid_keys = valid_window_table[
        [
            "source_segment_id",
            "window_id",
        ]
    ]

    valid_rows = rows_with_window.merge(
        valid_keys,
        on=[
            "source_segment_id",
            "window_id",
        ],
        how="inner",
    )

    return (
        valid_rows,
        valid_window_table.reset_index(
            drop=True
        ),
        skipped_window_table.reset_index(
            drop=True
        ),
    )


# =========================================================
# 7. 기본 Feature 11종
# =========================================================

def calculate_base_features(
    valid_rows,
    valid_window_table,
):
    """
    1분 Window마다
    학습 때와 동일한 기본 Feature 11종을 계산한다.
    """

    rows = valid_rows.copy()

    rows["rpm_error"] = (
        rows["spindle_rate"]
        - rows["spindle_rpm"]
    )

    grouped = rows.groupby(
        [
            "source_segment_id",
            "window_id",
        ]
    )

    features = grouped.agg(

        # RPM 5종 중 4종
        rpm_mean=(
            "spindle_rpm",
            "mean",
        ),
        rpm_std=(
            "spindle_rpm",
            "std",
        ),
        rpm_min=(
            "spindle_rpm",
            "min",
        ),
        rpm_max=(
            "spindle_rpm",
            "max",
        ),

        # LOAD 3종
        load_mean=(
            "spindle_load",
            "mean",
        ),
        load_std=(
            "spindle_load",
            "std",
        ),
        load_max=(
            "spindle_load",
            "max",
        ),

        # RPM_ERROR 3종
        rpm_error_mean=(
            "rpm_error",
            "mean",
        ),
        rpm_error_std=(
            "rpm_error",
            "std",
        ),
        rpm_error_absmax=(
            "rpm_error",
            lambda s: s.abs().max(),
        ),

    ).reset_index()

    # RPM 5번째 Feature
    features["rpm_range"] = (
        features["rpm_max"]
        - features["rpm_min"]
    )

    std_columns = [
        "rpm_std",
        "load_std",
        "rpm_error_std",
    ]

    features[std_columns] = (
        features[std_columns]
        .fillna(0.0)
    )

    feature_df = (
        valid_window_table.merge(
            features,
            on=[
                "source_segment_id",
                "window_id",
            ],
            how="inner",
        )
    )

    return feature_df


# =========================================================
# 8. 공구별 정상 Baseline 적용
# =========================================================

def apply_tool_baseline(
    feature_df,
    baseline,
):
    """
    Train Normal Baseline을 적용해
    상대 Feature 3종을 생성한다.

    Baseline이 없는 공구에는
    다른 공구나 전역 평균으로 fallback하지 않는다.
    """

    df = feature_df.copy()
    baseline_df = baseline.copy()

    df["current_tool"] = pd.to_numeric(
        df["current_tool"],
        errors="raise",
    ).astype(int)

    baseline_df[
        "current_tool"
    ] = pd.to_numeric(
        baseline_df["current_tool"],
        errors="raise",
    ).astype(int)

    df = df.merge(
        baseline_df,
        on="current_tool",
        how="left",
    )

    baseline_columns = [
        "load_center",
        "rpm_std_center",
        "rpmerr_center",
        "rpmerr_tol",
    ]

    df["supported"] = (
        df[baseline_columns]
        .notna()
        .all(axis=1)
    )

    df["reason"] = ""

    df.loc[
        ~df["supported"],
        "reason",
    ] = "train_normal_baseline 없음"

    supported_mask = df["supported"]

    # LOAD 정상 대비 비율
    df.loc[
        supported_mask,
        "load_vs_normal",
    ] = (
        df.loc[
            supported_mask,
            "load_mean",
        ]
        / df.loc[
            supported_mask,
            "load_center",
        ].clip(
            lower=LOAD_EPS
        )
    )

    # RPM 변동성 정상 대비 비율
    df.loc[
        supported_mask,
        "rpm_std_vs_normal",
    ] = (
        df.loc[
            supported_mask,
            "rpm_std",
        ]
        / df.loc[
            supported_mask,
            "rpm_std_center",
        ].clip(
            lower=RPM_STD_EPS
        )
    )

    # RPM_ERROR 정상 중심 대비 이탈
    df.loc[
        supported_mask,
        "rpm_error_vs_normal",
    ] = (
        (
            df.loc[
                supported_mask,
                "rpm_error_mean",
            ]
            - df.loc[
                supported_mask,
                "rpmerr_center",
            ]
        ).abs()
        / df.loc[
            supported_mask,
            "rpmerr_tol",
        ].clip(
            lower=RPM_ERROR_EPS
        )
    )

    return df


# =========================================================
# 9. XGBoost 예측
# =========================================================

def predict_windows(
    feature_df,
    model,
    feature_columns,
):
    """
    지원 가능한 Window에 대해
    XGBoost Anomaly Score를 계산한다.

    Anomaly Score는 실제 고장 확률이 아니다.
    """

    result_df = feature_df.copy()

    result_df[
        "anomaly_score"
    ] = np.nan

    supported_mask = (
        result_df["supported"]
    )

    if supported_mask.sum() == 0:
        return result_df

    # 저장 당시 Feature 순서 그대로 사용
    X = result_df.loc[
        supported_mask,
        feature_columns,
    ]

    scores = model.predict_proba(
        X
    )[:, 1]

    result_df.loc[
        supported_mask,
        "anomaly_score",
    ] = scores

    return result_df


# =========================================================
# 10. 전체 추론 Pipeline
# =========================================================

def run_inference(log_path):
    """
    기업 원본 로그 하나를 입력받아
    Parsing부터 XGBoost Anomaly Score까지 수행한다.

    흐름:
    Raw Log
    → Parsing
    → Cleaning
    → Cutting Segment
    → 1분 Window
    → Feature 14종
    → XGBoost
    → Anomaly Score
    """

    model, feature_columns, baseline = (
        load_model_artifacts()
    )

    parsed_df, written_rows, skipped_lines = (
        parse_raw_log(log_path)
    )

    clean_df = prepare_clean_data(
        parsed_df
    )

    segmented_df = (
        extract_cutting_segments(
            clean_df
        )
    )

    # 실제 절삭 데이터가 하나도 없는 파일
    if segmented_df.empty:
        return {
            "written_rows": written_rows,
            "skipped_lines": skipped_lines,
            "clean_rows": len(clean_df),
            "cutting_rows": 0,
            "results": pd.DataFrame(),
            "skipped_windows": pd.DataFrame(),
        }

    (
        rows_with_window,
        window_table,
    ) = build_one_minute_windows(
        segmented_df
    )

    (
        valid_rows,
        valid_window_table,
        skipped_window_table,
    ) = filter_valid_windows(
        rows_with_window,
        window_table,
    )

    # 충분한 표본을 가진 Window가 없는 경우
    if valid_window_table.empty:
        return {
            "written_rows": written_rows,
            "skipped_lines": skipped_lines,
            "clean_rows": len(clean_df),
            "cutting_rows": len(
                segmented_df
            ),
            "results": pd.DataFrame(),
            "skipped_windows":
                skipped_window_table,
        }

    feature_df = calculate_base_features(
        valid_rows,
        valid_window_table,
    )

    feature_df = apply_tool_baseline(
        feature_df,
        baseline,
    )

    result_df = predict_windows(
        feature_df,
        model,
        feature_columns,
    )

    return {
        "written_rows": written_rows,
        "skipped_lines": skipped_lines,
        "clean_rows": len(clean_df),
        "cutting_rows": len(
            segmented_df
        ),
        "results": result_df,
        "skipped_windows":
            skipped_window_table,
    }