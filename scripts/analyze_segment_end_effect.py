from pathlib import Path
import sys

import pandas as pd


# =========================================================
# 프로젝트 경로
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference_pipeline import run_inference


RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "inference"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    REPORT_DIR
    / "segment_end_effect.csv"
)


def analyze_segment_end_effect():

    log_files = sorted(
        RAW_DATA_DIR.glob(
            "Data.log_*.log"
        )
    )

    if not log_files:
        raise FileNotFoundError(
            f"원본 로그가 없습니다: {RAW_DATA_DIR}"
        )

    all_results = []

    print("=" * 70)
    print("31일 Segment 종료 영향 분석")
    print("=" * 70)

    # =====================================================
    # 날짜별 추론
    # =====================================================

    for index, log_path in enumerate(
        log_files,
        start=1,
    ):

        print(
            f"[{index}/{len(log_files)}] "
            f"{log_path.name}"
        )

        output = run_inference(
            log_path
        )

        result_df = output[
            "results"
        ].copy()

        skipped_df = output[
            "skipped_windows"
        ].copy()

        if result_df.empty:
            continue

        # -------------------------------------------------
        # 실제 Segment의 마지막 Window ID 계산
        #
        # results만 사용하면 마지막 Window가
        # 표본 부족으로 skipped된 경우를 놓칠 수 있음.
        #
        # 따라서 results + skipped_windows를 함께 사용.
        # -------------------------------------------------

        id_frames = [
            result_df[
                [
                    "source_segment_id",
                    "window_id",
                ]
            ]
        ]

        if not skipped_df.empty:

            id_frames.append(
                skipped_df[
                    [
                        "source_segment_id",
                        "window_id",
                    ]
                ]
            )

        all_window_ids = pd.concat(
            id_frames,
            ignore_index=True,
        )

        last_window_by_segment = (
            all_window_ids
            .groupby(
                "source_segment_id"
            )["window_id"]
            .max()
        )

        result_df[
            "segment_last_window_id"
        ] = (
            result_df[
                "source_segment_id"
            ]
            .map(
                last_window_by_segment
            )
        )

        result_df[
            "is_segment_last"
        ] = (
            result_df["window_id"]
            ==
            result_df[
                "segment_last_window_id"
            ]
        )

        result_df.insert(
            0,
            "source_file",
            log_path.name,
        )

        all_results.append(
            result_df
        )

    # =====================================================
    # 전체 결합
    # =====================================================

    if not all_results:

        print(
            "분석할 Window가 없습니다."
        )

        return

    all_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    all_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # 모델이 실제 점수를 낼 수 있는 Window만 사용
    supported_df = all_df[
        (
            all_df["supported"]
        )
        &
        (
            all_df[
                "anomaly_score"
            ].notna()
        )
    ].copy()

    scores = supported_df[
        "anomaly_score"
    ]

    p99_threshold = scores.quantile(
        0.99
    )

    # =====================================================
    # 기본 현황
    # =====================================================

    print("\n" + "=" * 70)
    print("전체 결과")
    print("=" * 70)

    print(
        "\n전체 지원 Window:",
        len(supported_df),
    )

    segment_last_count = int(
        supported_df[
            "is_segment_last"
        ].sum()
    )

    print(
        "Segment 마지막 Window:",
        segment_last_count,
    )

    print(
        "Segment 중간 Window:",
        len(supported_df)
        - segment_last_count,
    )

    # =====================================================
    # Segment 중간 vs 마지막 Score 비교
    # =====================================================

    print("\n" + "=" * 70)
    print("Segment 위치별 Anomaly Score")
    print("=" * 70)

    for is_last, label in [
        (
            False,
            "Segment 중간",
        ),
        (
            True,
            "Segment 마지막",
        ),
    ]:

        subset = supported_df[
            supported_df[
                "is_segment_last"
            ]
            == is_last
        ]

        subset_scores = subset[
            "anomaly_score"
        ]

        print(
            f"\n[{label}]"
        )

        print(
            "Window 수:",
            len(subset)
        )

        if len(subset) > 0:

            print(
                "평균:",
                round(
                    subset_scores.mean(),
                    6,
                )
            )

            print(
                "중앙값:",
                round(
                    subset_scores.median(),
                    6,
                )
            )

            print(
                "P95:",
                round(
                    subset_scores.quantile(
                        0.95
                    ),
                    6,
                )
            )

            print(
                "P99:",
                round(
                    subset_scores.quantile(
                        0.99
                    ),
                    6,
                )
            )

            print(
                "최대:",
                round(
                    subset_scores.max(),
                    6,
                )
            )

    # =====================================================
    # 높은 Score가 마지막 Window에 얼마나 몰리는지
    # =====================================================

    def print_high_score_result(
        title,
        subset,
    ):

        total = len(
            subset
        )

        last_count = int(
            subset[
                "is_segment_last"
            ].sum()
        )

        if total > 0:

            ratio = (
                last_count
                / total
                * 100
            )

        else:

            ratio = 0

        print(
            f"\n[{title}]"
        )

        print(
            "전체 Window:",
            total,
        )

        print(
            "Segment 마지막:",
            last_count,
        )

        print(
            "마지막 Window 비율:",
            f"{ratio:.2f}%",
        )

    # 상위 1%
    top_1_percent = supported_df[
        supported_df[
            "anomaly_score"
        ]
        >= p99_threshold
    ]

    print(
        "\n전체 Score P99 기준:",
        round(
            p99_threshold,
            6,
        )
    )

    print_high_score_result(
        "Score 상위 1%",
        top_1_percent,
    )

    # XGBoost 기본 binary threshold
    model_positive = supported_df[
        supported_df[
            "anomaly_score"
        ]
        >= 0.5
    ]

    print_high_score_result(
        "Anomaly Score 0.5 이상",
        model_positive,
    )

    # 매우 높은 점수
    very_high = supported_df[
        supported_df[
            "anomaly_score"
        ]
        >= 0.9
    ]

    print_high_score_result(
        "Anomaly Score 0.9 이상",
        very_high,
    )

    # =====================================================
    # 최고점 20개
    # =====================================================

    print("\n" + "=" * 70)
    print("Score 상위 20개")
    print("=" * 70)

    top20 = (
        supported_df
        .sort_values(
            "anomaly_score",
            ascending=False,
        )
        .head(20)
    )

    print(
        top20[
            [
                "source_file",
                "window_start",
                "current_tool",
                "anomaly_score",
                "is_segment_last",
                "rpm_std",
                "load_max",
                "rpm_error_absmax",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n저장 파일:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nSegment 종료 영향 분석 완료"
    )


if __name__ == "__main__":
    analyze_segment_end_effect()