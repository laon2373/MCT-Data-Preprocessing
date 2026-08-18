from pathlib import Path
import sys

import pandas as pd


# =========================================================
# 프로젝트 경로
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference_pipeline import (
    parse_raw_log,
    prepare_clean_data,
)


# =========================================================
# 경로 및 설정
# =========================================================

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

SEGMENT_EFFECT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "inference"
    / "segment_end_effect.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "inference"
    / "high_score_transitions.csv"
)

HIGH_SCORE_THRESHOLD = 0.9

# 이상 Window가 끝난 뒤 몇 초까지 볼지
LOOKAHEAD_SECONDS = 90


# =========================================================
# timestamp 변환
# =========================================================

def parse_clean_timestamp(series):

    # clean 데이터 timestamp 형식:
    # 2026-05-29 14:01:29,699

    parsed = pd.to_datetime(
        series,
        format="%Y-%m-%d %H:%M:%S,%f",
        errors="coerce",
    )

    # 혹시 다른 형식이 섞인 경우만 fallback
    missing = parsed.isna()

    if missing.any():

        parsed.loc[missing] = pd.to_datetime(
            series.loc[missing],
            errors="coerce",
        )

    return parsed


# =========================================================
# 메인 분석
# =========================================================

def analyze_high_score_transitions():

    # -----------------------------------------------------
    # 1. 이전 단계 결과 읽기
    # -----------------------------------------------------

    if not SEGMENT_EFFECT_PATH.exists():

        raise FileNotFoundError(
            "segment_end_effect.csv가 없습니다.\n"
            "먼저 analyze_segment_end_effect.py를 실행해주세요."
        )

    all_windows = pd.read_csv(
        SEGMENT_EFFECT_PATH
    )

    # CSV에서 bool이 문자열로 읽힐 가능성까지 처리
    if all_windows["is_segment_last"].dtype != bool:

        all_windows["is_segment_last"] = (
            all_windows["is_segment_last"]
            .astype(str)
            .str.lower()
            .eq("true")
        )

    # -----------------------------------------------------
    # 2. 분석 대상:
    #    Score >= 0.9 + Segment 마지막
    # -----------------------------------------------------

    targets = all_windows[
        (
            all_windows["supported"] == True
        )
        &
        (
            all_windows["anomaly_score"]
            >= HIGH_SCORE_THRESHOLD
        )
        &
        (
            all_windows["is_segment_last"]
        )
    ].copy()

    targets = targets.sort_values(
        "anomaly_score",
        ascending=False,
    )

    print("=" * 75)
    print("고득점 Segment 종료 후 상태 분석")
    print("=" * 75)

    print(
        "\n분석 대상 Window:",
        len(targets),
    )

    print(
        "조건:",
        f"Anomaly Score >= {HIGH_SCORE_THRESHOLD}"
        " AND Segment 마지막",
    )

    if targets.empty:

        print(
            "\n분석 대상이 없습니다."
        )

        return

    # -----------------------------------------------------
    # 3. 파일별로 묶어서 분석
    # -----------------------------------------------------

    transition_rows = []

    grouped = targets.groupby(
        "source_file"
    )

    for file_index, (
        source_file,
        file_targets,
    ) in enumerate(
        grouped,
        start=1,
    ):

        print(
            f"\n[{file_index}/{grouped.ngroups}] "
            f"{source_file}"
        )

        raw_path = (
            RAW_DATA_DIR
            / source_file
        )

        if not raw_path.exists():

            print(
                "  → 원본 파일 없음, 건너뜀"
            )

            continue

        # ---------------------------------------------
        # 원본 로그 파싱 + 전처리
        # ---------------------------------------------

        parsed_df, _, _ = parse_raw_log(
            raw_path
        )

        clean_df = prepare_clean_data(
            parsed_df
        )

        clean_df["_timestamp_dt"] = (
            parse_clean_timestamp(
                clean_df["timestamp"]
            )
        )

        # ---------------------------------------------
        # 해당 날짜의 고득점 Window 하나씩 확인
        # ---------------------------------------------

        for _, target in file_targets.iterrows():

            window_end = pd.to_datetime(
                target["window_end"]
            )

            current_tool = int(
                target["current_tool"]
            )

            lookahead_end = (
                window_end
                + pd.Timedelta(
                    seconds=LOOKAHEAD_SECONDS
                )
            )

            # Window가 끝난 '이후' 90초
            future = clean_df[
                (
                    clean_df["_timestamp_dt"]
                    > window_end
                )
                &
                (
                    clean_df["_timestamp_dt"]
                    <= lookahead_end
                )
            ].copy()

            # -----------------------------------------
            # 이후 데이터가 없는 경우
            # -----------------------------------------

            if future.empty:

                transition_rows.append(
                    {
                        "source_file":
                            source_file,

                        "window_start":
                            target[
                                "window_start"
                            ],

                        "window_end":
                            target[
                                "window_end"
                            ],

                        "current_tool":
                            current_tool,

                        "anomaly_score":
                            target[
                                "anomaly_score"
                            ],

                        "future_rows":
                            0,

                        "spindle_stopped":
                            False,

                        "tool_changed":
                            False,

                        "next_tool":
                            None,

                        "cutting_resumed":
                            False,

                        "event_after":
                            False,

                        "alarm_after":
                            False,

                        "seconds_to_stop":
                            None,

                        "seconds_to_tool_change":
                            None,

                        "seconds_to_resume":
                            None,
                    }
                )

                continue

            # =================================================
            # 4. 스핀들 정지 여부
            # =================================================

            stopped_rows = future[
                future["spindle_rpm"]
                == 0
            ]

            spindle_stopped = (
                not stopped_rows.empty
            )

            if spindle_stopped:

                first_stop_time = (
                    stopped_rows[
                        "_timestamp_dt"
                    ]
                    .iloc[0]
                )

                seconds_to_stop = (
                    first_stop_time
                    - window_end
                ).total_seconds()

            else:

                first_stop_time = None
                seconds_to_stop = None

            # =================================================
            # 5. 공구 변경 여부
            # =================================================

            changed_rows = future[
                future["current_tool"]
                != current_tool
            ]

            tool_changed = (
                not changed_rows.empty
            )

            if tool_changed:

                first_change = (
                    changed_rows.iloc[0]
                )

                next_tool = int(
                    first_change[
                        "current_tool"
                    ]
                )

                first_change_time = (
                    first_change[
                        "_timestamp_dt"
                    ]
                )

                seconds_to_tool_change = (
                    first_change_time
                    - window_end
                ).total_seconds()

            else:

                next_tool = None
                first_change_time = None
                seconds_to_tool_change = None

            # =================================================
            # 6. 이후 다시 절삭이 시작됐는지
            # =================================================

            cutting_mask = (
                (
                    future["status"]
                    == "STATUS_RUN"
                )
                &
                (
                    future["spindle_rpm"]
                    > 0
                )
                &
                (
                    future["feed_rate"]
                    > 0
                )
            )

            future_cutting = future[
                cutting_mask
            ]

            # 단순히 바로 다음 행도 절삭인 경우가 아니라
            # '스핀들 정지 이후 다시 절삭'했는지 확인
            if spindle_stopped:

                resumed_rows = (
                    future_cutting[
                        future_cutting[
                            "_timestamp_dt"
                        ]
                        > first_stop_time
                    ]
                )

            else:

                resumed_rows = (
                    future_cutting.iloc[0:0]
                )

            cutting_resumed = (
                not resumed_rows.empty
            )

            if cutting_resumed:

                first_resume_time = (
                    resumed_rows[
                        "_timestamp_dt"
                    ]
                    .iloc[0]
                )

                seconds_to_resume = (
                    first_resume_time
                    - window_end
                ).total_seconds()

            else:

                seconds_to_resume = None

            # =================================================
            # 7. Event / Alarm 확인
            # =================================================

            if "has_event" in future.columns:

                event_after = bool(
                    future[
                        "has_event"
                    ].any()
                )

            else:

                event_after = False

            if "is_alarm" in future.columns:

                alarm_after = bool(
                    future[
                        "is_alarm"
                    ].any()
                )

            else:

                alarm_after = False

            # =================================================
            # 8. 결과 저장
            # =================================================

            transition_rows.append(
                {
                    "source_file":
                        source_file,

                    "window_start":
                        target[
                            "window_start"
                        ],

                    "window_end":
                        target[
                            "window_end"
                        ],

                    "current_tool":
                        current_tool,

                    "anomaly_score":
                        target[
                            "anomaly_score"
                        ],

                    "future_rows":
                        len(future),

                    "spindle_stopped":
                        spindle_stopped,

                    "tool_changed":
                        tool_changed,

                    "next_tool":
                        next_tool,

                    "cutting_resumed":
                        cutting_resumed,

                    "event_after":
                        event_after,

                    "alarm_after":
                        alarm_after,

                    "seconds_to_stop":
                        seconds_to_stop,

                    "seconds_to_tool_change":
                        seconds_to_tool_change,

                    "seconds_to_resume":
                        seconds_to_resume,
                }
            )

    # =========================================================
    # 9. 결과 DataFrame
    # =========================================================

    result_df = pd.DataFrame(
        transition_rows
    )

    result_df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # =========================================================
    # 10. 전체 통계
    # =========================================================

    print("\n" + "=" * 75)
    print("전체 분석 결과")
    print("=" * 75)

    print(
        "\n분석 완료 Window:",
        len(result_df),
    )

    print(
        "90초 이내 스핀들 정지:",
        int(
            result_df[
                "spindle_stopped"
            ].sum()
        ),
    )

    print(
        "90초 이내 공구 변경:",
        int(
            result_df[
                "tool_changed"
            ].sum()
        ),
    )

    print(
        "정지 후 90초 이내 절삭 재개:",
        int(
            result_df[
                "cutting_resumed"
            ].sum()
        ),
    )

    print(
        "90초 이내 Event:",
        int(
            result_df[
                "event_after"
            ].sum()
        ),
    )

    print(
        "90초 이내 Alarm:",
        int(
            result_df[
                "alarm_after"
            ].sum()
        ),
    )

    # ---------------------------------------------------------
    # 공정 전환 형태
    # ---------------------------------------------------------

    transition_pattern = (
        result_df[
            "spindle_stopped"
        ]
        &
        result_df[
            "tool_changed"
        ]
    )

    print(
        "\n스핀들 정지 + 공구 변경:",
        int(
            transition_pattern.sum()
        ),
        "/",
        len(result_df),
    )

    # =========================================================
    # 11. 개별 결과
    # =========================================================

    print("\n" + "=" * 75)
    print("개별 Window 결과")
    print("=" * 75)

    columns_to_show = [
        "source_file",
        "window_end",
        "current_tool",
        "anomaly_score",
        "spindle_stopped",
        "seconds_to_stop",
        "tool_changed",
        "next_tool",
        "seconds_to_tool_change",
        "cutting_resumed",
        "event_after",
        "alarm_after",
    ]

    print(
        result_df[
            columns_to_show
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
        "\n고득점 Segment 전환 분석 완료"
    )


if __name__ == "__main__":
    analyze_high_score_transitions()