import pandas as pd
import requests
import streamlit as st


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="MCT 예지보전",
    page_icon="⚙️",
    layout="wide",
)

API_URL = "http://127.0.0.1:8000/predict"


# =========================================================
# 제목
# =========================================================

st.title("MCT 예지보전 시스템")

st.write(
    "MCT 원본 로그를 업로드하여 "
    "AI 기반 이상 패턴을 분석합니다."
)

st.info(
    "Anomaly Score는 실제 고장 확률이 아니라, "
    "학습된 이상 패턴과의 유사도를 나타내는 모델 점수입니다."
)


# =========================================================
# 1. LOG 파일 업로드
# =========================================================

st.subheader("1. MCT 로그 업로드")

uploaded_file = st.file_uploader(
    "분석할 MCT 원본 로그(.log)를 선택하세요.",
    type=["log"],
)


# =========================================================
# 2. 업로드 파일 정보
# =========================================================

if uploaded_file is not None:

    file_size_mb = (
        uploaded_file.size
        / 1024
        / 1024
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "파일명",
            uploaded_file.name,
        )

    with col2:
        st.metric(
            "파일 크기",
            f"{file_size_mb:.2f} MB",
        )

    # =====================================================
    # 3. AI 분석
    # =====================================================

    if st.button(
        "AI 분석 시작",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "MCT 로그를 분석하고 있습니다..."
            ):

                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        "text/plain",
                    )
                }

                response = requests.post(
                    API_URL,
                    files=files,
                    timeout=300,
                )

            if response.status_code == 200:

                st.session_state[
                    "analysis_result"
                ] = response.json()

                st.success(
                    "AI 분석이 완료되었습니다."
                )

            else:

                st.error(
                    f"분석에 실패했습니다. "
                    f"(HTTP {response.status_code})"
                )

                try:
                    st.write(
                        response.json()
                    )

                except Exception:
                    st.write(
                        response.text
                    )

        except requests.exceptions.ConnectionError:

            st.error(
                "FastAPI 서버에 연결할 수 없습니다. "
                "서버가 실행 중인지 확인해주세요."
            )

        except requests.exceptions.Timeout:

            st.error(
                "분석 시간이 너무 오래 걸려 요청이 종료되었습니다."
            )

        except Exception as e:

            st.error(
                f"예상하지 못한 오류가 발생했습니다: {e}"
            )


# =========================================================
# 4. 분석 결과 대시보드
# =========================================================

if "analysis_result" in st.session_state:

    result = st.session_state[
        "analysis_result"
    ]

    st.divider()

    st.subheader("2. 분석 결과 요약")

    # -----------------------------------------------------
    # 기본 통계
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "전체 측정 행",
            f"{result['clean_rows']:,}",
        )

    with col2:
        st.metric(
            "실제 절삭 행",
            f"{result['cutting_rows']:,}",
        )

    with col3:
        st.metric(
            "분석 Window",
            f"{result['supported_windows']:,}",
        )

    with col4:
        st.metric(
            "최대 Anomaly Score",
            f"{result['score_max']:.6f}"
            if result["score_max"] is not None
            else "-",
        )

    st.caption(
        f"평균 Anomaly Score: "
        f"{result['score_mean']:.6f}"
        if result["score_mean"] is not None
        else "평균 Anomaly Score: -"
    )

    # -----------------------------------------------------
    # 분석 가능 여부
    # -----------------------------------------------------

    st.subheader("3. 분석 데이터 현황")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "분석 가능 Window",
            f"{result['supported_windows']:,}",
        )

    with col2:
        st.metric(
            "미지원 공구 Window",
            f"{result['unsupported_windows']:,}",
        )

    with col3:
        st.metric(
            "표본 부족 Window",
            f"{result['insufficient_windows']:,}",
        )

    # -----------------------------------------------------
    # 최고점 Window
    # -----------------------------------------------------

    highest = result.get(
        "highest_window"
    )

    if highest is not None:

        st.subheader(
            "4. 가장 높은 이상 패턴 구간"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.write(
                "**분석 시작 시간**"
            )

            st.write(
                highest["window_start"]
            )

            st.write(
                "**분석 종료 시간**"
            )

            st.write(
                highest["window_end"]
            )

        with col2:

            st.metric(
                "공구 번호",
                f"Tool {highest['current_tool']}",
            )

            st.metric(
                "Anomaly Score",
                f"{highest['anomaly_score']:.6f}",
            )


    # -----------------------------------------------------
    # 최고점 Window의 정상 대비 지표
    # -----------------------------------------------------

    top_windows = result.get(
        "top_windows",
        [],
    )

    if highest is not None and top_windows:

        top_window = top_windows[0]

        st.subheader(
            "5. 최고점 구간의 정상 대비 지표"
        )

        col1, col2, col3 = st.columns(3)

        load_ratio = top_window[
            "load_vs_normal"
        ]

        rpm_std_ratio = top_window[
            "rpm_std_vs_normal"
        ]

        rpm_error_ratio = top_window[
            "rpm_error_vs_normal"
        ]

        with col1:

            st.metric(
                "LOAD 상대지표",
                f"{load_ratio:.3f}",
            )

            st.caption(
                "공구별 정상 LOAD 중심값과 비교한 값"
            )

        with col2:

            st.metric(
                "RPM 변동 상대지표",
                f"{rpm_std_ratio:.3f}",
            )

            st.caption(
                "공구별 정상 RPM 변동성과 비교한 값"
            )

        with col3:

            st.metric(
                "RPM ERROR 상대지표",
                f"{rpm_error_ratio:.3f}",
            )

            st.caption(
                "공구별 정상 RPM 오차 기준과 비교한 값"
            )

        st.info(
            "위 값은 센서가 공구별 정상 기준에서 "
            "얼마나 달라졌는지를 보여주는 비교 지표이며, "
            "XGBoost가 해당 Feature를 원인으로 판단했다는 "
            "의미는 아닙니다."
        )


    # -----------------------------------------------------
    # XGBoost 주요 기여 Feature
    # -----------------------------------------------------

    top_contributions = result.get(
        "top_contributions",
        [],
    )

    if top_contributions:

        st.subheader(
            "6. 모델 주요 이상 기여 Feature"
        )

        feature_name_map = {
            "rpm_mean": "RPM 평균",
            "rpm_std": "RPM 변동성",
            "rpm_min": "RPM 최솟값",
            "rpm_max": "RPM 최댓값",
            "rpm_range": "RPM 범위",

            "load_mean": "LOAD 평균",
            "load_std": "LOAD 변동성",
            "load_max": "LOAD 최댓값",

            "rpm_error_mean": "RPM ERROR 평균",
            "rpm_error_std": "RPM ERROR 변동성",
            "rpm_error_absmax": "RPM ERROR 최대 절댓값",

            "load_vs_normal": "정상 대비 LOAD",
            "rpm_std_vs_normal": "정상 대비 RPM 변동",
            "rpm_error_vs_normal": "정상 대비 RPM ERROR",
        }

        contribution_cols = st.columns(
            len(top_contributions)
        )

        for index, item in enumerate(
            top_contributions
        ):

            feature_name = item[
                "feature"
            ]

            contribution = item[
                "contribution"
            ]

            display_name = (
                feature_name_map.get(
                    feature_name,
                    feature_name,
                )
            )

            with contribution_cols[index]:

                st.metric(
                    display_name,
                    f"{contribution:.3f}",
                )

        st.caption(
            "위 값은 최고 Anomaly Score 구간에서 "
            "XGBoost의 이상 방향 점수를 높인 Feature 중 "
            "기여도가 큰 순서입니다. "
            "기여도는 고장 확률이나 센서 이상 정도 자체를 의미하지 않습니다."
        )

    # -----------------------------------------------------
    # Score 상위 10개 Window
    # -----------------------------------------------------

    top_windows = result.get(
        "top_windows",
        [],
    )

    if top_windows:

        st.subheader(
            "7. Anomaly Score 상위 10개 구간"
        )

        top_df = pd.DataFrame(
            top_windows
        )

        top_df = top_df.rename(
            columns={
                "window_start":
                    "시작 시간",

                "window_end":
                    "종료 시간",

                "current_tool":
                    "공구",

                "sample_count":
                    "표본 수",

                "anomaly_score":
                    "Anomaly Score",

                "load_vs_normal":
                    "LOAD 상대지표",

                "rpm_std_vs_normal":
                    "RPM 변동 상대지표",

                "rpm_error_vs_normal":
                    "RPM ERROR 상대지표",
            }
        )

        top_df["공구"] = (
            "Tool "
            + top_df["공구"].astype(str)
        )

        st.dataframe(
            top_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "상대지표는 해당 공구의 학습용 정상 Baseline과 "
            "비교하여 계산한 값입니다."
        )