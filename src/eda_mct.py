from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "output" / "mct_clean.csv"
REPORT_DIR = PROJECT_DIR / "reports" / "eda"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 최종 정제 데이터 불러오기
df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8-sig",
    low_memory=False,
)

# 시간 열 변환
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%Y-%m-%d %H:%M:%S,%f",
    errors="coerce",
)

# 분석에 사용할 수치형 열
numeric_columns = [
    "feed_rate",
    "spindle_rate",
    "spindle_load",
    "delta_spindle_rpm",
    "x_axis_mm",
    "y_axis_mm",
    "z_axis_mm",
]

numeric_columns = [
    column for column in numeric_columns
    if column in df.columns
]

flag_columns = [
    "is_alarm",
    "is_emergency",
    "is_reset",
    "is_spindle_overload",
    "is_abnormal_interval",
    "has_event",
]

flag_columns = [
    column for column in flag_columns
    if column in df.columns
]

# 1. 데이터 기본 정보 저장
basic_info = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[column].dtype) for column in df.columns],
    "missing_count": [df[column].isna().sum() for column in df.columns],
    "missing_ratio": [
        df[column].isna().mean()
        for column in df.columns
    ],
})

basic_info.to_csv(
    REPORT_DIR / "basic_info.csv",
    index=False,
    encoding="utf-8-sig",
)

# 2. 수치형 변수 통계 저장
numeric_summary = df[numeric_columns].describe().T
numeric_summary.index.name = "feature"

numeric_summary.reset_index().to_csv(
    REPORT_DIR / "numeric_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# 3. 이벤트 발생 건수 저장
flag_summary = df[flag_columns].sum().rename("count")
flag_summary.index.name = "event"

flag_summary.reset_index().to_csv(
    REPORT_DIR / "event_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# 4. 날짜별 데이터 수와 이벤트 수
df["date"] = df["timestamp"].dt.date

daily_summary = (
    df.groupby("date")
    .agg(
        total_count=("timestamp", "size"),
        event_count=("has_event", "sum"),
        alarm_count=("is_alarm", "sum"),
        reset_count=("is_reset", "sum"),
        overload_count=("is_spindle_overload", "sum"),
        abnormal_interval_count=("is_abnormal_interval", "sum"),
    )
    .reset_index()
)

daily_summary.to_csv(
    REPORT_DIR / "daily_event_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# 5. 정상 구간과 이벤트 구간의 주요 변수 비교
event_comparison = (
    df.groupby("has_event")[numeric_columns]
    .agg(["mean", "median", "min", "max"])
)

event_comparison.columns = [
    f"{column}_{stat}"
    for column, stat in event_comparison.columns
]

event_comparison = event_comparison.reset_index()

event_counts = (
    df["has_event"]
    .value_counts()
    .rename_axis("has_event")
    .rename("row_count")
    .reset_index()
)

event_comparison = event_counts.merge(
    event_comparison,
    on="has_event",
)

event_comparison.to_csv(
    REPORT_DIR / "event_comparison.csv",
    index=False,
    encoding="utf-8-sig",
)

# 6. 주요 수치형 변수 분포 그래프
df[numeric_columns].hist(
    figsize=(14, 10),
    bins=50,
)

plt.tight_layout()
plt.savefig(
    REPORT_DIR / "numeric_distributions.png",
    dpi=150,
)

plt.close()

# 7. 정상 구간과 이벤트 구간 비교 그래프
for column in numeric_columns:
    plt.figure(figsize=(6, 4))

    sns.boxplot(
        data=df,
        x="has_event",
        y=column,
    )

    plt.title(f"{column} by Event Status")
    plt.xlabel("has_event")
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(
        REPORT_DIR / f"{column}_event_comparison.png",
        dpi=150,
    )

    plt.close()

# 8. 변수 간 상관관계 그래프
correlation = df[numeric_columns].corr()

correlation.index.name = "feature"

correlation.reset_index().to_csv(
    REPORT_DIR / "correlation.csv",
    index=False,
    encoding="utf-8-sig",
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)

plt.title("Numeric Feature Correlation")
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "correlation_heatmap.png",
    dpi=150,
)

plt.close()

print("[EDA 완료]")
print(f"분석 행 수: {len(df):,}")
print(f"분석 변수 수: {len(numeric_columns):,}")
print(f"결과 저장 위치: {REPORT_DIR}")