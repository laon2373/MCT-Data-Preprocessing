from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =========================================================
# 프로젝트 경로 설정
# =========================================================

# 현재 파일:
# MCT-Data-Preprocessing/scripts/analyze_real_scores.py
#
# parents[1]:
# MCT-Data-Preprocessing/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# src 패키지를 import할 수 있도록 프로젝트 루트를 경로에 추가
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference_pipeline import run_inference


# =========================================================
# 경로
# =========================================================

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "inference"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DAILY_SUMMARY_PATH = (
    REPORT_DIR
    / "real_score_daily_summary.csv"
)

WINDOW_RESULTS_PATH = (
    REPORT_DIR
    / "real_window_scores.csv"
)


# =========================================================
# 전체 실제 로그 분석
# =========================================================

def analyze_all_real_logs():

    # Data.log_20260501.log ~ Data.log_20260531.log
    log_files = sorted(
        RAW_DATA_DIR.glob("Data.log_*.log")
    )

    if not log_files:
        raise FileNotFoundError(
            f"원본 로그 파일이 없습니다: {RAW_DATA_DIR}"
        )

    print("=" * 70)
    print("실제 31일 원본 로그 Anomaly Score 분석")
    print("=" * 70)

    print(f"\n발견된 로그 파일 수: {len(log_files)}")

    daily_rows = []
    all_window_results = []

    # ---------------------------------------------------------
    # 날짜별 파일 하나씩 inference
    # ---------------------------------------------------------
    for index, log_path in enumerate(
        log_files,
        start=1,
    ):

        print(
            f"\n[{index}/{len(log_files)}] "
            f"{log_path.name} 분석 중..."
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

        # -----------------------------------------------------
        # 절삭 Window가 하나도 없는 경우
        # -----------------------------------------------------
        if result_df.empty:

            daily_rows.append(
                {
                    "source_file": log_path.name,
                    "parsed_rows": output[
                        "written_rows"
                    ],
                    "clean_rows": output[
                        "clean_rows"
                    ],
                    "cutting_rows": output[
                        "cutting_rows"
                    ],
                    "feature_windows": 0,
                    "supported_windows": 0,
                    "unsupported_windows": 0,
                    "insufficient_windows": len(
                        skipped_df
                    ),
                    "score_count": 0,
                    "score_mean": np.nan,
                    "score_min": np.nan,
                    "score_p50": np.nan,
                    "score_p90": np.nan,
                    "score_p95": np.nan,
                    "score_p99": np.nan,
                    "score_max": np.nan,
                }
            )

            print(
                "  → 모델 입력 가능한 Window 없음"
            )

            continue

        # 어떤 원본 파일에서 나온 결과인지 기록
        result_df.insert(
            0,
            "source_file",
            log_path.name,
        )

        all_window_results.append(
            result_df
        )

        # -----------------------------------------------------
        # 지원 / 미지원 공구 분리
        # -----------------------------------------------------
        supported_df = result_df[
            result_df["supported"]
        ].copy()

        unsupported_df = result_df[
            ~result_df["supported"]
        ].copy()

        scores = (
            supported_df[
                "anomaly_score"
            ]
            .dropna()
        )

        # -----------------------------------------------------
        # Score 통계
        # -----------------------------------------------------
        if len(scores) > 0:

            score_mean = scores.mean()
            score_min = scores.min()

            score_p50 = scores.quantile(
                0.50
            )

            score_p90 = scores.quantile(
                0.90
            )

            score_p95 = scores.quantile(
                0.95
            )

            score_p99 = scores.quantile(
                0.99
            )

            score_max = scores.max()

        else:

            score_mean = np.nan
            score_min = np.nan
            score_p50 = np.nan
            score_p90 = np.nan
            score_p95 = np.nan
            score_p99 = np.nan
            score_max = np.nan

        daily_rows.append(
            {
                "source_file": log_path.name,

                "parsed_rows": output[
                    "written_rows"
                ],

                "clean_rows": output[
                    "clean_rows"
                ],

                "cutting_rows": output[
                    "cutting_rows"
                ],

                "feature_windows": len(
                    result_df
                ),

                "supported_windows": len(
                    supported_df
                ),

                "unsupported_windows": len(
                    unsupported_df
                ),

                "insufficient_windows": len(
                    skipped_df
                ),

                "score_count": len(
                    scores
                ),

                "score_mean": score_mean,
                "score_min": score_min,

                "score_p50": score_p50,
                "score_p90": score_p90,
                "score_p95": score_p95,
                "score_p99": score_p99,

                "score_max": score_max,
            }
        )

        print(
            f"  절삭 행: {output['cutting_rows']:,}"
        )

        print(
            f"  지원 Window: {len(supported_df):,}"
        )

        print(
            f"  미지원 Window: {len(unsupported_df):,}"
        )

        print(
            f"  표본 부족 Window: {len(skipped_df):,}"
        )

        if len(scores) > 0:

            print(
                f"  Score 평균: {score_mean:.6f}"
            )

            print(
                f"  Score P99:  {score_p99:.6f}"
            )

            print(
                f"  Score 최대: {score_max:.6f}"
            )

    # =========================================================
    # 날짜별 Summary 저장
    # =========================================================

    daily_summary_df = pd.DataFrame(
        daily_rows
    )

    daily_summary_df.to_csv(
        DAILY_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================
    # 모든 Window 결과 저장
    # =========================================================

    if all_window_results:

        all_results_df = pd.concat(
            all_window_results,
            ignore_index=True,
        )

        all_results_df.to_csv(
            WINDOW_RESULTS_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        all_supported = all_results_df[
            all_results_df["supported"]
        ].copy()

        all_scores = (
            all_supported[
                "anomaly_score"
            ]
            .dropna()
        )

    else:

        all_results_df = pd.DataFrame()
        all_scores = pd.Series(
            dtype=float
        )

    # =========================================================
    # 전체 결과 출력
    # =========================================================

    print("\n" + "=" * 70)
    print("31일 전체 분석 결과")
    print("=" * 70)

    print(
        "\n전체 Feature Window:",
        len(all_results_df),
    )

    if not all_results_df.empty:

        print(
            "전체 지원 Window:",
            int(
                all_results_df[
                    "supported"
                ].sum()
            ),
        )

        print(
            "전체 미지원 Window:",
            int(
                (
                    ~all_results_df[
                        "supported"
                    ]
                ).sum()
            ),
        )

    print(
        "\nAnomaly Score 개수:",
        len(all_scores),
    )

    if len(all_scores) > 0:

        print("\nAnomaly Score 전체 분포:")

        print(
            all_scores
            .describe(
                percentiles=[
                    0.50,
                    0.90,
                    0.95,
                    0.99,
                ]
            )
            .round(6)
        )

        print(
            "\nScore가 가장 높은 20개 Window:"
        )

        top20 = (
            all_results_df[
                all_results_df[
                    "supported"
                ]
            ]
            .sort_values(
                "anomaly_score",
                ascending=False,
            )
            .head(20)
        )

        columns_to_show = [
            "source_file",
            "window_start",
            "window_end",
            "current_tool",
            "sample_count",
            "anomaly_score",
            "load_vs_normal",
            "rpm_std_vs_normal",
            "rpm_error_vs_normal",
        ]

        print(
            top20[
                columns_to_show
            ].to_string(
                index=False
            )
        )

    print("\n저장 파일:")

    print(
        DAILY_SUMMARY_PATH
    )

    print(
        WINDOW_RESULTS_PATH
    )

    print(
        "\n31일 실제 로그 분석 완료"
    )


# =========================================================
# 직접 실행
# =========================================================

if __name__ == "__main__":
    analyze_all_real_logs()