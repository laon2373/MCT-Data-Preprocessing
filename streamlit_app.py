import hashlib
import html

import pandas as pd
import requests
import streamlit as st


# =========================================================
# 기본 설정
# =========================================================

st.set_page_config(
    page_title="MCT AI 예지보전",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = "http://127.0.0.1:8000/predict"

# 2026년 5월 실제 로그 31일의 Anomaly Score 분포에서 확인한 참고값.
# 고장 임계값이 아니라, 실제 로그 안에서 상대적으로 얼마나 높은 점수인지
# 설명하기 위한 참고 기준이다.
REFERENCE_P95 = 0.018857
REFERENCE_P99 = 0.136186


# =========================================================
# Helper
# =========================================================

FEATURE_NAME_MAP = {
    "rpm_mean": "RPM 평균",
    "rpm_std": "RPM 흔들림",
    "rpm_min": "RPM 최솟값",
    "rpm_max": "RPM 최댓값",
    "rpm_range": "RPM 변화 폭",
    "load_mean": "평균 Spindle Load",
    "load_std": "Spindle Load 변동",
    "load_max": "최대 Spindle Load",
    "rpm_error_mean": "명령 RPM과 실제 RPM의 평균 차이",
    "rpm_error_std": "RPM 오차의 변동",
    "rpm_error_absmax": "명령 RPM과 실제 RPM의 최대 차이",
    "load_vs_normal": "평소 대비 Spindle Load",
    "rpm_std_vs_normal": "평소 대비 RPM 흔들림",
    "rpm_error_vs_normal": "평소 대비 RPM 오차",
}


def format_datetime(value, include_date=True):
    if value is None:
        return "-"

    dt = pd.to_datetime(
        value,
        errors="coerce",
    )

    if pd.isna(dt):
        return str(value)

    if include_date:
        return dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    return dt.strftime(
        "%H:%M:%S"
    )


def get_reference_level(score):
    """
    실제 31일 로그 분포(P95/P99) 기반 참고 수준.
    실제 고장 등급이나 고장 확률이 아니다.
    """
    if score is None:
        return {
            "label": "분석 결과 없음",
            "class": "neutral",
            "description": "분석 가능한 Anomaly Score가 없습니다.",
        }

    if score < REFERENCE_P95:
        return {
            "label": "낮은 이상 유사도",
            "class": "good",
            "description": (
                "파일 내 최고점도 실제 31일 로그의 "
                "상위 5% 경계보다 낮습니다."
            ),
        }

    if score < REFERENCE_P99:
        return {
            "label": "관찰 필요",
            "class": "watch",
            "description": (
                "실제 로그에서 상대적으로 높은 편에 해당하는 "
                "이상 패턴 유사도가 확인되었습니다."
            ),
        }

    return {
        "label": "높은 이상 유사도",
        "class": "alert",
        "description": (
            "실제 31일 로그에서 상위 1% 수준에 해당하는 "
            "높은 이상 패턴 유사도가 확인되었습니다."
        ),
    }


def render_card(
    label,
    value,
    subtext="",
    card_class="",
    eyebrow="",
):
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    safe_subtext = html.escape(str(subtext))
    safe_eyebrow = html.escape(str(eyebrow))

    eyebrow_html = (
        f'<div class="card-eyebrow">{safe_eyebrow}</div>'
        if safe_eyebrow
        else ""
    )

    st.markdown(
        f"""
        <div class="metric-card {card_class}">
            {eyebrow_html}
            <div class="metric-label">{safe_label}</div>
            <div class="metric-value">{safe_value}</div>
            <div class="metric-sub">{safe_subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_reason_lines(
    highest,
    top_contributions,
):
    if highest is None:
        return []

    lines = []

    load_ratio = highest.get(
        "load_vs_normal"
    )

    if load_ratio is not None:
        if load_ratio >= 1.10:
            lines.append(
                "Spindle Load가 이 공구의 평소 기준보다 "
                f"약 {(load_ratio - 1) * 100:.1f}% 높았습니다."
            )
        elif load_ratio >= 1.02:
            lines.append(
                "Spindle Load가 이 공구의 평소 기준보다 "
                f"약 {(load_ratio - 1) * 100:.1f}% 조금 높았습니다."
            )
        elif load_ratio <= 0.90:
            lines.append(
                "Spindle Load는 이 공구의 평소 기준보다 "
                "낮은 수준이었습니다."
            )
        else:
            lines.append(
                "Spindle Load는 이 공구의 평소 수준과 "
                "비슷했습니다."
            )

    rpm_std_ratio = highest.get(
        "rpm_std_vs_normal"
    )

    if rpm_std_ratio is not None:
        if rpm_std_ratio >= 2:
            lines.append(
                "RPM 흔들림이 공구별 정상 기준보다 "
                f"약 {rpm_std_ratio:.1f}배 크게 나타났습니다."
            )
        elif rpm_std_ratio >= 1:
            lines.append(
                "RPM 흔들림이 평소 기준보다 "
                "다소 크게 나타났습니다."
            )
        else:
            lines.append(
                "RPM 흔들림은 평소 기준보다 "
                "크지 않았습니다."
            )

    rpm_error_ratio = highest.get(
        "rpm_error_vs_normal"
    )

    if rpm_error_ratio is not None:
        if rpm_error_ratio >= 1:
            lines.append(
                "명령 RPM과 실제 RPM의 차이가 "
                "공구별 정상 허용폭을 벗어났습니다."
            )
        elif rpm_error_ratio >= 0.5:
            lines.append(
                "명령 RPM과 실제 RPM의 차이가 "
                "정상 허용폭 안에서 비교적 크게 나타났습니다."
            )
        else:
            lines.append(
                "명령 RPM과 실제 RPM의 차이는 "
                "공구별 정상 허용폭 안쪽이었습니다."
            )

    if top_contributions:
        names = [
            FEATURE_NAME_MAP.get(
                item.get("feature"),
                item.get("feature"),
            )
            for item in top_contributions
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
                + " 항목이 이상 방향 점수를 높이는 데 "
                "상대적으로 크게 작용했습니다."
            )

    return lines


def build_check_items(
    top_contributions,
):
    items = [
        "해당 시간대에 공구 교체 또는 공정 전환이 있었는지 확인하세요.",
        "해당 공구의 마모·파손·체결 상태를 필요 시 현장에서 확인하세요.",
    ]

    feature_names = {
        item.get("feature")
        for item in top_contributions
    }

    if any(
        feature.startswith("load")
        for feature in feature_names
        if feature
    ):
        items.append(
            "Spindle Load가 일시적으로 증가했는지와 "
            "가공 조건·절삭 저항 변화를 함께 확인하세요."
        )

    if any(
        (
            feature.startswith("rpm_error")
            or feature == "rpm_error_vs_normal"
        )
        for feature in feature_names
        if feature
    ):
        items.append(
            "명령 RPM과 실제 RPM의 차이가 커진 시점이 "
            "있는지 원본 로그를 확인하세요."
        )

    if any(
        (
            feature.startswith("rpm_std")
            or feature == "rpm_range"
        )
        for feature in feature_names
        if feature
    ):
        items.append(
            "Spindle RPM이 짧은 시간 동안 불안정하게 "
            "흔들렸는지 확인하세요."
        )

    unique_items = []

    for item in items:
        if item not in unique_items:
            unique_items.append(item)

    return unique_items[:4]


# =========================================================
# Service UI CSS
# =========================================================

st.markdown(
    """
    <style>
        :root {
            --navy: #0f172a;
            --blue: #2563eb;
            --blue-soft: #eff6ff;
            --green: #16a34a;
            --green-soft: #f0fdf4;
            --amber: #d97706;
            --amber-soft: #fffbeb;
            --red: #dc2626;
            --red-soft: #fef2f2;
            --border: #e2e8f0;
            --muted: #64748b;
            --panel: #ffffff;
            --canvas: #f8fafc;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        div[data-testid="stToolbar"] {
            visibility: hidden;
            height: 0;
        }

        .stApp {
            background: var(--canvas);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.2rem;
            padding-bottom: 4rem;
        }

        .service-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 1rem;
            background:
                linear-gradient(135deg, #0f172a 0%, #172554 55%, #1e3a8a 100%);
            border-radius: 20px;
            padding: 1.45rem 1.65rem;
            margin-bottom: 1rem;
            color: white;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
        }

        .service-eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            font-weight: 700;
            opacity: 0.7;
            margin-bottom: 0.4rem;
        }

        .service-title {
            font-size: 2rem;
            line-height: 1.1;
            font-weight: 800;
            letter-spacing: -0.04em;
        }

        .service-description {
            margin-top: 0.5rem;
            font-size: 0.92rem;
            line-height: 1.6;
            opacity: 0.82;
        }

        .service-badge {
            flex: 0 0 auto;
            border: 1px solid rgba(255,255,255,0.25);
            background: rgba(255,255,255,0.08);
            border-radius: 999px;
            padding: 0.55rem 0.85rem;
            font-size: 0.78rem;
            font-weight: 650;
            white-space: nowrap;
        }

        .upload-panel {
            background: white;
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.05rem 1.2rem 0.4rem 1.2rem;
            margin: 0.8rem 0 1rem 0;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        }

        .upload-title {
            color: var(--navy);
            font-size: 1.02rem;
            font-weight: 750;
            margin-bottom: 0.25rem;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.55;
        }

        .section-heading {
            margin-top: 1.3rem;
            margin-bottom: 0.7rem;
        }

        .section-kicker {
            color: var(--blue);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }

        .section-title {
            color: var(--navy);
            font-size: 1.35rem;
            line-height: 1.25;
            font-weight: 800;
            letter-spacing: -0.025em;
        }

        .section-description {
            color: var(--muted);
            font-size: 0.86rem;
            margin-top: 0.3rem;
            line-height: 1.55;
        }

        .metric-card {
            height: 148px;
            padding: 1.05rem 1.1rem;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
            overflow: hidden;
        }

        .metric-card.good {
            border-top: 4px solid var(--green);
        }

        .metric-card.watch {
            border-top: 4px solid #f59e0b;
        }

        .metric-card.alert {
            border-top: 4px solid var(--red);
        }

        .metric-card.primary {
            border-top: 4px solid var(--blue);
        }

        .metric-card.neutral {
            border-top: 4px solid #94a3b8;
        }

        .card-eyebrow {
            color: var(--blue);
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            margin-bottom: 0.3rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 650;
            margin-bottom: 0.48rem;
        }

        .metric-value {
            color: var(--navy);
            font-size: 1.72rem;
            line-height: 1.08;
            font-weight: 800;
            letter-spacing: -0.035em;
            margin-bottom: 0.45rem;
        }

        .metric-sub {
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.45;
        }

        .info-panel {
            background: white;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1rem 1.15rem;
            line-height: 1.75;
            color: #334155;
            margin: 0.35rem 0 0.9rem 0;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.025);
        }

        .reason-panel {
            background: var(--amber-soft);
            border: 1px solid #fde68a;
            border-radius: 16px;
            padding: 1rem 1.15rem;
            line-height: 1.75;
            color: #3f3f46;
            margin-bottom: 0.9rem;
        }

        .action-panel {
            background: var(--green-soft);
            border: 1px solid #bbf7d0;
            border-radius: 16px;
            padding: 1rem 1.15rem;
            line-height: 1.75;
            color: #334155;
            margin-bottom: 0.9rem;
        }

        .notice-panel {
            background: var(--blue-soft);
            border: 1px solid #bfdbfe;
            border-radius: 14px;
            padding: 0.8rem 1rem;
            color: #1e3a8a;
            font-size: 0.86rem;
            line-height: 1.6;
            margin: 0.4rem 0 0.9rem 0;
        }

        .reference-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.65rem;
            margin-bottom: 0.6rem;
        }

        .reference-item {
            background: white;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
        }

        .reference-label {
            color: var(--muted);
            font-size: 0.72rem;
            margin-bottom: 0.2rem;
        }

        .reference-value {
            color: var(--navy);
            font-size: 1rem;
            font-weight: 750;
        }

        .footer-note {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border);
            color: var(--muted);
            font-size: 0.78rem;
            line-height: 1.6;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
        }

        div[data-testid="stExpander"] {
            background: white;
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        button[kind="primary"] {
            border-radius: 10px !important;
            font-weight: 700 !important;
        }

        @media (max-width: 760px) {
            .service-header {
                align-items: flex-start;
                flex-direction: column;
            }

            .service-title {
                font-size: 1.6rem;
            }

            .reference-strip {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 서비스 헤더
# =========================================================

st.markdown(
    """
    <div class="service-header">
        <div>
            <div class="service-eyebrow">MCT CONDITION MONITORING</div>
            <div class="service-title">MCT AI 예지보전 분석</div>
            <div class="service-description">
                설비 원본 로그에서 실제 절삭 구간을 추출하고,
                공구별 정상 기준과 AI 모델을 이용해 이상 패턴을 분석합니다.
            </div>
        </div>
        <div class="service-badge">기업 설비 로그 기반 분석 서비스</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice-panel">
        <b>Anomaly Score 안내</b><br>
        실제 고장 확률이 아니라, 현재 1분 절삭 구간이
        모델이 학습한 가상 이상 패턴과 얼마나 유사한지를 나타내는 점수입니다.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 파일 업로드
# =========================================================

st.markdown(
    """
    <div class="upload-panel">
        <div class="upload-title">분석 파일 업로드</div>
        <div class="small-muted">
            기업에서 수집한 MCT 원본 .log 파일을 선택한 뒤
            분석을 시작하세요. 분석 결과와 업로드 데이터 요약이 함께 제공됩니다.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "분석할 MCT 원본 로그(.log)",
    type=["log"],
    label_visibility="collapsed",
)

current_file_hash = None

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()

    current_file_hash = hashlib.sha1(
        file_bytes
    ).hexdigest()

    file_size_mb = (
        uploaded_file.size
        / 1024
        / 1024
    )

    file_col, size_col = st.columns(
        [3, 1]
    )

    with file_col:
        st.markdown(
            f"**선택 파일**　`{uploaded_file.name}`"
        )

    with size_col:
        st.markdown(
            f"**파일 크기**　{file_size_mb:.2f} MB"
        )

    if st.button(
        "AI 분석 시작",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "로그 정제 → 절삭 구간 추출 → AI 이상 패턴 분석을 진행하고 있습니다..."
            ):

                files = {
                    "file": (
                        uploaded_file.name,
                        file_bytes,
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

                st.session_state[
                    "analysis_file_hash"
                ] = current_file_hash

                st.success(
                    "분석이 완료되었습니다."
                )

            else:

                st.error(
                    "분석에 실패했습니다. "
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
                "백엔드 서버가 실행 중인지 확인해주세요."
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
# 분석 결과 표시 여부
# =========================================================

has_matching_result = (
    uploaded_file is not None
    and
    "analysis_result" in st.session_state
    and
    st.session_state.get(
        "analysis_file_hash"
    ) == current_file_hash
)


if has_matching_result:

    result = st.session_state[
        "analysis_result"
    ]

    st.divider()

    tab_ai, tab_data = st.tabs(
        [
            "AI 설비 진단",
            "업로드 데이터 분석",
        ]
    )

    # =====================================================
    # TAB 1. AI 설비 진단
    # =====================================================

    with tab_ai:

        highest = result.get(
            "highest_window"
        )

        top_windows = result.get(
            "top_windows",
            [],
        )

        top_contributions = result.get(
            "top_contributions",
            [],
        )

        score_max = result.get(
            "score_max"
        )

        reference = get_reference_level(
            score_max
        )

        st.markdown(
            """
            <div class="section-heading">
                <div class="section-kicker">AI DIAGNOSIS</div>
                <div class="section-title">설비 이상 패턴 분석 결과</div>
                <div class="section-description">
                    파일 내 최고점 구간과 공구별 정상 기준을 함께 해석합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            render_card(
                "참고 수준",
                reference["label"],
                reference["description"],
                reference["class"],
                "RESULT",
            )

        with col2:
            render_card(
                "최대 Anomaly Score",
                (
                    f"{score_max:.4f}"
                    if score_max is not None
                    else "-"
                ),
                (
                    f"파일 평균 {result['score_mean']:.4f}"
                    if result.get("score_mean") is not None
                    else "평균 점수 없음"
                ),
                "primary",
                "SCORE",
            )

        with col3:

            if highest is not None:
                period_text = (
                    f"{format_datetime(highest['window_start'], False)}"
                    " ~ "
                    f"{format_datetime(highest['window_end'], False)}"
                )
            else:
                period_text = "-"

            render_card(
                "파일 내 최고 점수 구간",
                (
                    f"Tool {highest['current_tool']}"
                    if highest is not None
                    else "-"
                ),
                period_text,
                "primary",
                "TOP WINDOW",
            )

        st.caption(
            "참고 수준은 2026년 5월 실제 로그의 Anomaly Score 분포(P95/P99)를 "
            "이용한 상대적 비교이며 설비 고장 등급이 아닙니다."
        )

        # -------------------------------------------------
        # 한눈에 보기
        # -------------------------------------------------

        st.markdown(
            """
            <div class="section-heading">
                <div class="section-kicker">SUMMARY</div>
                <div class="section-title">분석 요약</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if highest is not None:

            summary_text = (
                f"이번 파일에서 <b>{result['supported_windows']:,}개의 "
                "1분 절삭 구간</b>을 AI가 분석했습니다. "
                f"파일 내 최대 Anomaly Score는 <b>{score_max:.4f}</b>이며, "
                f"<b>Tool {highest['current_tool']}</b>의 "
                f"<b>{format_datetime(highest['window_start'], False)}"
                f" ~ {format_datetime(highest['window_end'], False)}</b> "
                "구간에서 확인되었습니다. "
                f"실제 31일 로그 분포와 비교했을 때 현재 참고 수준은 "
                f"<b>{reference['label']}</b>입니다."
            )

        else:

            summary_text = (
                "이 파일에서는 AI가 점수를 계산할 수 있는 "
                "절삭 구간이 확인되지 않았습니다."
            )

        st.markdown(
            f"""
            <div class="info-panel">
                {summary_text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # 구간 해석 + 현장 점검
        # -------------------------------------------------

        if highest is not None:

            left, right = st.columns(
                [1.25, 1]
            )

            with left:

                st.markdown(
                    """
                    <div class="section-heading">
                        <div class="section-kicker">INTERPRETATION</div>
                        <div class="section-title">이 구간에서 무엇이 달랐나요?</div>
                        <div class="section-description">
                            파일 내 최고점 구간을 공구별 정상 기준과 비교한 결과입니다.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                reason_lines = build_reason_lines(
                    highest,
                    top_contributions,
                )

                reason_html = "".join(
                    f"<li>{html.escape(line)}</li>"
                    for line in reason_lines
                )

                st.markdown(
                    f"""
                    <div class="reason-panel">
                        <b>
                            Tool {highest['current_tool']} ·
                            {format_datetime(highest['window_start'], False)}
                            ~
                            {format_datetime(highest['window_end'], False)}
                        </b>
                        <ul>
                            {reason_html}
                        </ul>
                        <div class="small-muted">
                            정상 Baseline과 모델 내부 기여도를 이해하기 쉬운 문장으로
                            설명한 것이며 실제 고장 원인을 확정한 결과는 아닙니다.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:

                st.markdown(
                    """
                    <div class="section-heading">
                        <div class="section-kicker">ACTION GUIDE</div>
                        <div class="section-title">현장 점검 가이드</div>
                        <div class="section-description">
                            AI 결과를 현장 확인으로 연결할 때 참고할 항목입니다.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                check_items = build_check_items(
                    top_contributions
                )

                check_html = "".join(
                    f"<li>{html.escape(item)}</li>"
                    for item in check_items
                )

                if reference["class"] == "good":
                    action_intro = (
                        "<b>현재는 낮은 이상 유사도입니다.</b> "
                        "즉시 조치보다는 최고점 구간을 검토할 때 "
                        "아래 항목을 참고하는 수준이 적절합니다."
                    )
                else:
                    action_intro = (
                        "<b>상대적으로 높은 점수가 확인되었습니다.</b> "
                        "아래 항목을 우선 확인하는 것을 권장합니다."
                    )

                st.markdown(
                    f"""
                    <div class="action-panel">
                        {action_intro}
                        <ul>
                            {check_html}
                        </ul>
                        <div class="small-muted">
                            공구 교체·공정 전환에서도 높은 점수가 나타날 수 있으며,
                            AI 결과만으로 실제 고장을 확정하지 않습니다.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # -------------------------------------------------
        # 시간 흐름별 Score
        # -------------------------------------------------

        score_series = result.get(
            "score_series",
            [],
        )

        if score_series:

            st.markdown(
                """
                <div class="section-heading">
                    <div class="section-kicker">TREND</div>
                    <div class="section-title">시간 흐름별 Anomaly Score</div>
                    <div class="section-description">
                        파일 전체에서 점수가 언제 상승했는지 확인합니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            max_text = (
                f"{score_max:.4f}"
                if score_max is not None
                else "-"
            )

            p95_ratio_text = "-"

            if (
                score_max is not None
                and REFERENCE_P95 > 0
            ):
                p95_ratio_text = (
                    f"{score_max / REFERENCE_P95 * 100:.1f}%"
                )

            st.markdown(
                f"""
                <div class="reference-strip">
                    <div class="reference-item">
                        <div class="reference-label">파일 내 최대 점수</div>
                        <div class="reference-value">{max_text}</div>
                    </div>
                    <div class="reference-item">
                        <div class="reference-label">실제 로그 P95 참고선</div>
                        <div class="reference-value">{REFERENCE_P95:.4f}</div>
                    </div>
                    <div class="reference-item">
                        <div class="reference-label">최대점수 / P95</div>
                        <div class="reference-value">{p95_ratio_text}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            score_df = pd.DataFrame(
                score_series
            )

            score_df[
                "window_start"
            ] = pd.to_datetime(
                score_df["window_start"],
                errors="coerce",
            )

            score_df = (
                score_df
                .dropna(subset=["window_start"])
                .sort_values("window_start")
            )

            if not score_df.empty:

                # P95가 현재 데이터보다 높더라도 그래프 범위를 지나치게
                # 확대하지 않도록 P95까지만 주 참고선으로 표시한다.
                y_max = max(
                    float(
                        score_df["anomaly_score"].max()
                    ) * 1.15,
                    REFERENCE_P95 * 1.08,
                )

                chart_data = score_df[
                    [
                        "window_start",
                        "anomaly_score",
                        "current_tool",
                    ]
                ].copy()

                st.vega_lite_chart(
                    chart_data,
                    {
                        "layer": [
                            {
                                "mark": {
                                    "type": "line",
                                    "strokeWidth": 2,
                                },
                                "encoding": {
                                    "x": {
                                        "field": "window_start",
                                        "type": "temporal",
                                        "title": "시간",
                                    },
                                    "y": {
                                        "field": "anomaly_score",
                                        "type": "quantitative",
                                        "title": "Anomaly Score",
                                        "scale": {
                                            "domain": [
                                                0,
                                                y_max,
                                            ]
                                        },
                                    },
                                    "tooltip": [
                                        {
                                            "field": "window_start",
                                            "type": "temporal",
                                            "title": "시작 시간",
                                        },
                                        {
                                            "field": "current_tool",
                                            "type": "quantitative",
                                            "title": "Tool",
                                        },
                                        {
                                            "field": "anomaly_score",
                                            "type": "quantitative",
                                            "title": "Anomaly Score",
                                            "format": ".6f",
                                        },
                                    ],
                                },
                            },
                            {
                                "data": {
                                    "values": [
                                        {
                                            "reference": REFERENCE_P95
                                        }
                                    ]
                                },
                                "mark": {
                                    "type": "rule",
                                    "strokeDash": [7, 5],
                                    "strokeWidth": 2,
                                },
                                "encoding": {
                                    "y": {
                                        "field": "reference",
                                        "type": "quantitative",
                                    },
                                },
                            },
                        ],
                        "height": 310,
                    },
                    use_container_width=True,
                )

                st.caption(
                    f"점선은 실제 31일 로그의 P95 참고선({REFERENCE_P95:.4f})입니다. "
                    f"P99 참고값은 {REFERENCE_P99:.4f}이며, 현재 그래프 범위보다 높습니다. "
                    "두 값 모두 고장 판정 임계값이 아니라 실제 로그 분포의 참고값입니다."
                )

        # -------------------------------------------------
        # Top 3
        # -------------------------------------------------

        if top_windows:

            st.markdown(
                """
                <div class="section-heading">
                    <div class="section-kicker">TOP WINDOWS</div>
                    <div class="section-title">파일 내 최고 점수 구간 3개</div>
                    <div class="section-description">
                        파일 내부에서 Anomaly Score가 높은 순서입니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            top3_df = pd.DataFrame(
                top_windows[:3]
            )

            top3_df = top3_df[
                [
                    "window_start",
                    "window_end",
                    "current_tool",
                    "anomaly_score",
                ]
            ].copy()

            top3_df[
                "window_start"
            ] = top3_df[
                "window_start"
            ].map(
                lambda value:
                    format_datetime(value)
            )

            top3_df[
                "window_end"
            ] = top3_df[
                "window_end"
            ].map(
                lambda value:
                    format_datetime(value)
            )

            top3_df[
                "current_tool"
            ] = (
                "Tool "
                + top3_df[
                    "current_tool"
                ].astype(str)
            )

            top3_df[
                "anomaly_score"
            ] = top3_df[
                "anomaly_score"
            ].map(
                lambda value:
                    f"{value:.6f}"
            )

            top3_df = top3_df.rename(
                columns={
                    "window_start":
                        "시작 시간",
                    "window_end":
                        "종료 시간",
                    "current_tool":
                        "공구",
                    "anomaly_score":
                        "Anomaly Score",
                }
            )

            st.dataframe(
                top3_df,
                use_container_width=True,
                hide_index=True,
            )

        # -------------------------------------------------
        # AI 상세 분석
        # -------------------------------------------------

        with st.expander(
            "AI 상세 분석 보기",
            expanded=False,
        ):

            coverage_cols = st.columns(3)

            with coverage_cols[0]:
                st.metric(
                    "AI 분석 가능 구간",
                    f"{result['supported_windows']:,}",
                )

            with coverage_cols[1]:
                st.metric(
                    "정상 기준이 없어 제외",
                    f"{result['unsupported_windows']:,}",
                )

            with coverage_cols[2]:
                st.metric(
                    "측정값이 부족해 제외",
                    f"{result['insufficient_windows']:,}",
                )

            if highest is not None:

                st.markdown(
                    "**최고점 구간의 공구별 정상 기준 대비 값**"
                )

                detail_cols = st.columns(3)

                with detail_cols[0]:
                    st.metric(
                        "Spindle Load 상대지표",
                        f"{highest.get('load_vs_normal', 0):.3f}",
                    )

                with detail_cols[1]:
                    st.metric(
                        "RPM 변동 상대지표",
                        f"{highest.get('rpm_std_vs_normal', 0):.3f}",
                    )

                with detail_cols[2]:
                    st.metric(
                        "RPM ERROR 상대지표",
                        f"{highest.get('rpm_error_vs_normal', 0):.3f}",
                    )

                st.caption(
                    "상대지표는 해당 공구의 학습용 정상 Baseline과 비교한 값입니다."
                )

            if top_contributions:

                st.markdown(
                    "**XGBoost 이상 방향 기여 Feature**"
                )

                contribution_df = pd.DataFrame(
                    top_contributions
                )

                contribution_df[
                    "feature"
                ] = contribution_df[
                    "feature"
                ].map(
                    lambda name:
                        FEATURE_NAME_MAP.get(
                            name,
                            name,
                        )
                )

                contribution_df[
                    "contribution"
                ] = contribution_df[
                    "contribution"
                ].round(4)

                contribution_df = contribution_df.rename(
                    columns={
                        "feature":
                            "Feature",
                        "contribution":
                            "기여도",
                    }
                )

                st.dataframe(
                    contribution_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "기여도는 모델 내부에서 이상 방향 점수를 높인 영향이며 "
                    "고장 원인이나 센서 이상 정도 자체를 의미하지 않습니다."
                )

            if top_windows:

                st.markdown(
                    "**Anomaly Score 상위 10개 구간**"
                )

                detail_df = pd.DataFrame(
                    top_windows
                )

                detail_df[
                    "current_tool"
                ] = (
                    "Tool "
                    + detail_df[
                        "current_tool"
                    ].astype(str)
                )

                detail_df = detail_df.rename(
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

                st.dataframe(
                    detail_df,
                    use_container_width=True,
                    hide_index=True,
                )

    # =====================================================
    # TAB 2. 업로드 데이터 분석
    # =====================================================

    with tab_data:

        st.markdown(
            """
            <div class="section-heading">
                <div class="section-kicker">DATA PROFILE</div>
                <div class="section-title">업로드 파일 데이터 요약</div>
                <div class="section-description">
                    업로드한 로그가 어떤 공정 데이터를 담고 있는지 쉽게 정리합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cutting_ratio = result.get(
            "cutting_ratio",
            0,
        )

        used_tool_count = result.get(
            "used_tool_count",
            0,
        )

        segment_count = result.get(
            "segment_count",
            0,
        )

        interval_mean = result.get(
            "sampling_interval_mean"
        )

        data_cols = st.columns(4)

        with data_cols[0]:
            render_card(
                "전체 측정 데이터",
                f"{result['clean_rows']:,}건",
                "파싱·정제 후 측정 행",
                "primary",
                "DATA",
            )

        with data_cols[1]:
            render_card(
                "실제 절삭 데이터",
                f"{result['cutting_rows']:,}건",
                f"전체의 {cutting_ratio * 100:.1f}%",
                "primary",
                "CUTTING",
            )

        with data_cols[2]:
            render_card(
                "사용 공구",
                f"{used_tool_count}종",
                "절삭 과정에서 확인된 공구",
                "primary",
                "TOOLS",
            )

        with data_cols[3]:
            render_card(
                "연속 절삭 구간",
                f"{segment_count}개",
                "공구·시간 흐름 기준",
                "primary",
                "SEGMENTS",
            )

        interval_text = (
            f"약 {interval_mean:.2f}초"
            if interval_mean is not None
            else "확인 불가"
        )

        data_summary = (
            f"이 파일에는 <b>{result['clean_rows']:,}건</b>의 "
            f"설비 측정 데이터가 있습니다. "
            f"이 중 <b>{result['cutting_rows']:,}건"
            f"({cutting_ratio * 100:.1f}%)</b>이 "
            "실제 절삭 작업으로 판단되었습니다. "
            f"절삭 과정에서는 <b>{used_tool_count}종의 공구</b>와 "
            f"<b>{segment_count}개의 연속 절삭 구간</b>이 확인되었습니다. "
            f"절삭 Window에서 추정한 평균 센서 수집 간격은 "
            f"<b>{interval_text}</b>입니다."
        )

        st.markdown(
            f"""
            <div class="info-panel">
                {data_summary}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if result.get("file_start") and result.get("file_end"):

            st.caption(
                "절삭 데이터 시간 범위　"
                f"{format_datetime(result['file_start'])} "
                "→ "
                f"{format_datetime(result['file_end'])}"
            )

        st.caption(
            "※ 수집 간격은 원본 전체 interval_seconds 평균이 아니라 "
            "절삭 Window의 시작·종료 시간과 표본 수를 이용해 추정한 참고값입니다."
        )

        # -------------------------------------------------
        # 공구 사용 현황
        # -------------------------------------------------

        tool_summary = result.get(
            "tool_summary",
            [],
        )

        if tool_summary:

            st.markdown(
                """
                <div class="section-heading">
                    <div class="section-kicker">TOOL USAGE</div>
                    <div class="section-title">사용 공구 현황</div>
                    <div class="section-description">
                        공구별 절삭 데이터량과 대표 운전 수준을 확인합니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            tool_df = pd.DataFrame(
                tool_summary
            )

            tool_df[
                "공구"
            ] = (
                "Tool "
                + tool_df[
                    "tool"
                ].astype(str)
            )

            tool_df[
                "절삭 비율"
            ] = (
                tool_df[
                    "cutting_ratio"
                ] * 100
            ).round(1)

            tool_df[
                "대표 RPM"
            ] = tool_df[
                "rpm_median"
            ].round(1)

            tool_df[
                "대표 LOAD"
            ] = tool_df[
                "load_median"
            ].round(1)

            display_tool_df = tool_df[
                [
                    "공구",
                    "cutting_rows",
                    "절삭 비율",
                    "대표 RPM",
                    "대표 LOAD",
                ]
            ].rename(
                columns={
                    "cutting_rows":
                        "절삭 데이터 수",
                }
            )

            left, right = st.columns(
                [1.15, 1]
            )

            with left:

                st.dataframe(
                    display_tool_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "대표 RPM/LOAD는 분석 Window의 중앙값을 이용한 참고값입니다."
                )

            with right:

                bar_df = (
                    tool_df[
                        [
                            "공구",
                            "cutting_rows",
                        ]
                    ]
                    .set_index(
                        "공구"
                    )
                    .rename(
                        columns={
                            "cutting_rows":
                                "절삭 데이터 수"
                        }
                    )
                )

                st.bar_chart(
                    bar_df,
                    height=330,
                )

        # -------------------------------------------------
        # 절삭 Timeline
        # -------------------------------------------------

        segments = result.get(
            "cutting_segments",
            [],
        )

        if segments:

            st.markdown(
                """
                <div class="section-heading">
                    <div class="section-kicker">PROCESS TIMELINE</div>
                    <div class="section-title">절삭 작업 흐름</div>
                    <div class="section-description">
                        시간대별로 어떤 공구가 절삭 작업에 사용되었는지 확인합니다.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            segment_df = pd.DataFrame(
                segments
            )

            segment_df[
                "start"
            ] = pd.to_datetime(
                segment_df["start"],
                errors="coerce",
            )

            segment_df[
                "end"
            ] = pd.to_datetime(
                segment_df["end"],
                errors="coerce",
            )

            segment_df = segment_df.dropna(
                subset=["start", "end"]
            )

            if not segment_df.empty:

                segment_df[
                    "tool_label"
                ] = (
                    "Tool "
                    + segment_df[
                        "tool"
                    ].astype(str)
                )

                st.vega_lite_chart(
                    segment_df,
                    {
                        "mark": {
                            "type": "bar",
                            "cornerRadius": 5,
                        },
                        "encoding": {
                            "x": {
                                "field": "start",
                                "type": "temporal",
                                "title": "시간",
                            },
                            "x2": {
                                "field": "end",
                            },
                            "y": {
                                "field": "tool_label",
                                "type": "nominal",
                                "title": "공구",
                            },
                            "tooltip": [
                                {
                                    "field": "tool_label",
                                    "type": "nominal",
                                    "title": "공구",
                                },
                                {
                                    "field": "start",
                                    "type": "temporal",
                                    "title": "시작",
                                },
                                {
                                    "field": "end",
                                    "type": "temporal",
                                    "title": "종료",
                                },
                                {
                                    "field": "duration_minutes",
                                    "type": "quantitative",
                                    "title": "지속(분)",
                                    "format": ".1f",
                                },
                            ],
                        },
                        "height": max(
                            180,
                            min(
                                420,
                                50
                                * segment_df[
                                    "tool_label"
                                ].nunique(),
                            ),
                        ),
                    },
                    use_container_width=True,
                )

                segment_table = segment_df[
                    [
                        "start",
                        "end",
                        "tool_label",
                        "duration_minutes",
                        "cutting_rows",
                    ]
                ].copy()

                segment_table[
                    "start"
                ] = segment_table[
                    "start"
                ].dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                segment_table[
                    "end"
                ] = segment_table[
                    "end"
                ].dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                segment_table[
                    "duration_minutes"
                ] = segment_table[
                    "duration_minutes"
                ].round(1)

                segment_table = segment_table.rename(
                    columns={
                        "start":
                            "시작",
                        "end":
                            "종료",
                        "tool_label":
                            "공구",
                        "duration_minutes":
                            "지속 시간(분)",
                        "cutting_rows":
                            "절삭 데이터 수",
                    }
                )

                with st.expander(
                    "절삭 구간 목록 보기",
                    expanded=False,
                ):
                    st.dataframe(
                        segment_table,
                        use_container_width=True,
                        hide_index=True,
                    )

        # -------------------------------------------------
        # AI 분석 가능 범위
        # -------------------------------------------------

        st.markdown(
            """
            <div class="section-heading">
                <div class="section-kicker">DATA COVERAGE</div>
                <div class="section-title">AI 분석 가능 범위</div>
                <div class="section-description">
                    실제 분석에 사용된 구간과 제외된 구간을 구분합니다.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        coverage_cols = st.columns(3)

        with coverage_cols[0]:
            render_card(
                "AI 분석 가능",
                f"{result['supported_windows']:,}개",
                "정상 Baseline과 충분한 표본을 가진 구간",
                "good",
            )

        with coverage_cols[1]:
            render_card(
                "정상 기준 없음",
                f"{result['unsupported_windows']:,}개",
                "학습용 공구 Baseline이 없어 제외",
                "watch",
            )

        with coverage_cols[2]:
            render_card(
                "측정값 부족",
                f"{result['insufficient_windows']:,}개",
                "1분 Window의 표본 수가 부족해 제외",
                "watch",
            )

        st.markdown(
            """
            <div class="notice-panel">
                <b>실제 절삭 데이터 정의</b><br>
                STATUS_RUN 상태이면서 Spindle RPM &gt; 0,
                Feed Rate &gt; 0인 측정값만 절삭 데이터로 사용합니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =====================================================
    # 공통 Footer
    # =====================================================

    st.markdown(
        """
        <div class="footer-note">
            본 시스템은 설비 상태 판단을 보조하기 위한 AI 분석 프로토타입입니다.
            실제 고장 여부와 정비 시점은 현장 점검, 공정 정보, 정비 이력 등과 함께 판단해야 합니다.
        </div>
        """,
        unsafe_allow_html=True,
    )


elif uploaded_file is not None:

    st.info(
        "파일을 선택했습니다. "
        "'AI 분석 시작' 버튼을 눌러 분석을 실행해주세요."
    )
