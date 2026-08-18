from pathlib import Path
import sys

import pandas as pd


# =========================================================
# 프로젝트 경로 설정
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference_pipeline import (
    run_inference,
    parse_raw_log,
    prepare_clean_data,
    extract_cutting_segments,
    build_one_minute_windows,
)


# =========================================================
# 분석 대상
# =========================================================

RAW_LOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Data.log_20260529.log"
)


def inspect_top_anomaly():

    # ---------------------------------------------------------
    # 1. 5월 29일 전체 추론
    # ---------------------------------------------------------
    output = run_inference(
        RAW_LOG_PATH
    )

    result_df = output[
        "results"
    ].copy()

    supported_df = result_df[
        result_df["supported"]
    ].copy()

    if supported_df.empty:
        print("분석 가능한 Window가 없습니다.")
        return

    # ---------------------------------------------------------
    # 2. Anomaly Score가 가장 높은 Window 찾기
    # ---------------------------------------------------------
    top_window = (
        supported_df
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .iloc[0]
    )

    segment_id = int(
        top_window[
            "source_segment_id"
        ]
    )

    window_id = int(
        top_window[
            "window_id"
        ]
    )


        # ---------------------------------------------------------
    # 최고점 Window 전후 2개 Window 확인
    # ---------------------------------------------------------

    top_segment_id = int(
        top_window["source_segment_id"]
    )

    top_window_id = int(
        top_window["window_id"]
    )

    neighbor_windows = supported_df[
        (
            supported_df["source_segment_id"]
            == top_segment_id
        )
        &
        (
            supported_df["window_id"]
            >= top_window_id - 2
        )
        &
        (
            supported_df["window_id"]
            <= top_window_id + 2
        )
    ].copy()

    neighbor_windows = neighbor_windows.sort_values(
        "window_id"
    )

    print("\n[최고점 Window 전후 구간]")

    print(
        neighbor_windows[
            [
                "window_id",
                "window_start",
                "window_end",
                "current_tool",
                "sample_count",
                "anomaly_score",
                "rpm_std",
                "load_max",
                "rpm_error_absmax",
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 3. 원본 로그를 다시 전처리하여
    #    해당 Window의 실제 센서 행 추출
    # ---------------------------------------------------------
    parsed_df, _, _ = parse_raw_log(
        RAW_LOG_PATH
    )

    clean_df = prepare_clean_data(
        parsed_df
    )

    # ---------------------------------------------------------
    # 최고점 Window 종료 직전/직후 상태 확인
    # ---------------------------------------------------------

    check_df = clean_df.copy()

    check_df["_timestamp_dt"] = pd.to_datetime(
        check_df["timestamp"]
    )

    target_end_time = pd.to_datetime(
        top_window["window_end"]
    )

    before_time = target_end_time - pd.Timedelta(
        seconds=30
    )

    after_time = target_end_time + pd.Timedelta(
        seconds=90
    )

    around_end = check_df[
        (
            check_df["_timestamp_dt"] >= before_time
        )
        &
        (
            check_df["_timestamp_dt"] <= after_time
        )
    ].copy()

    around_end["rpm_error"] = (
        around_end["spindle_rate"]
        - around_end["spindle_rpm"]
    )

    print("\n[이상 Window 종료 직전/직후 상태]")

    print(
        around_end[
            [
                "timestamp",
                "status",
                "current_tool",
                "feed_rate",
                "spindle_rate",
                "spindle_rpm",
                "spindle_load",
                "rpm_error",
                "has_event",
            ]
        ].to_string(
            index=False
        )
    )


    # ---------------------------------------------------------
    # 해당 Segment의 전체 Window 확인
    # ---------------------------------------------------------

    same_segment = supported_df[
        supported_df["source_segment_id"]
        == top_segment_id
    ].copy()

    same_segment = same_segment.sort_values(
        "window_id"
    )

    print("\n[해당 절삭 Segment 전체 Window]")

    print(
        same_segment[
            [
                "window_id",
                "window_start",
                "window_end",
                "sample_count",
                "anomaly_score",
                "rpm_std",
                "load_max",
                "rpm_error_absmax",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nSegment 마지막 Window ID:",
        int(same_segment["window_id"].max())
    )

    print(
        "최고점 Window ID:",
        top_window_id
    )

    print(
        "최고점이 Segment 마지막 Window인가?:",
        top_window_id
        == int(same_segment["window_id"].max())
    )


    segmented_df = extract_cutting_segments(
        clean_df
    )

    rows_with_window, _ = (
        build_one_minute_windows(
            segmented_df
        )
    )

    target_rows = rows_with_window[
        (
            rows_with_window[
                "source_segment_id"
            ]
            == segment_id
        )
        &
        (
            rows_with_window[
                "window_id"
            ]
            == window_id
        )
    ].copy()

    # RPM_ERROR 추가
    target_rows["rpm_error"] = (
        target_rows["spindle_rate"]
        - target_rows["spindle_rpm"]
    )



    # ---------------------------------------------------------
    # 4. 결과 출력
    # ---------------------------------------------------------
    print("=" * 75)
    print("최고 Anomaly Score Window 상세 분석")
    print("=" * 75)

    print("\n[AI 판정 Window]")

    print(
        "Anomaly Score:",
        top_window["anomaly_score"],
    )

    print(
        "시간:",
        top_window["window_start"],
        "~",
        top_window["window_end"],
    )

    print(
        "공구:",
        int(top_window["current_tool"]),
    )

    print(
        "표본 수:",
        int(top_window["sample_count"]),
    )

    print("\n[정상 대비 Feature]")

    print(
        "load_vs_normal:",
        top_window["load_vs_normal"],
    )

    print(
        "rpm_std_vs_normal:",
        top_window["rpm_std_vs_normal"],
    )

    print(
        "rpm_error_vs_normal:",
        top_window["rpm_error_vs_normal"],
    )

    print("\n[기본 Feature]")

    base_columns = [
        "rpm_mean",
        "rpm_std",
        "rpm_min",
        "rpm_max",
        "rpm_range",
        "load_mean",
        "load_std",
        "load_max",
        "rpm_error_mean",
        "rpm_error_std",
        "rpm_error_absmax",
    ]

    for column in base_columns:
        print(
            f"{column}:",
            top_window[column],
        )

    # ---------------------------------------------------------
    # 실제 센서 원본값
    # ---------------------------------------------------------
    print("\n[해당 1분의 실제 센서값]")

    sensor_columns = [
        "timestamp",
        "current_tool",
        "feed_rate",
        "spindle_rate",
        "spindle_rpm",
        "spindle_load",
        "rpm_error",
    ]

    print(
        target_rows[
            sensor_columns
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 센서값 간단 요약
    # ---------------------------------------------------------
    print("\n[센서 변화 요약]")

    print(
        target_rows[
            [
                "spindle_rate",
                "spindle_rpm",
                "spindle_load",
                "rpm_error",
            ]
        ]
        .describe()
        .round(3)
    )

    print(
        "\n최고 Anomaly Score Window 분석 완료"
    )


if __name__ == "__main__":
    inspect_top_anomaly()