from pathlib import Path
import pandas as pd

# 프로젝트 경로
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "output" / "mct_parsed.csv"
REPORT_DIR = PROJECT_DIR / "reports"

# 숫자형으로 검사할 열
NUMERIC_COLUMNS = [
    "feed_rate",
    "spindle_rate",
    "x_axis",
    "y_axis",
    "z_axis",
    "current_tool",
    "x_load",
    "y_load",
    "z_load",
    "spindle_load",
    "spindle_rpm",
    "delta_spindle_rpm",
    "aut_mode",
    "run_mode",
    "emergency",
    "alarm",
]


def main():
    """파싱된 MCT 데이터의 기본 품질을 검사하고 결과를 저장합니다."""
    if not INPUT_FILE.exists():
        print(f"입력 파일을 찾지 못했습니다: {INPUT_FILE}")
        return

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("CSV 파일을 불러오는 중입니다...")
    data = pd.read_csv(
        INPUT_FILE,
        encoding="utf-8-sig",
        low_memory=False,
    )

    original_row_count = len(data)

    # timestamp를 날짜·시간 형식으로 변환.
    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        format="%Y-%m-%d %H:%M:%S,%f",
        errors="coerce",
    )

    # 센서 및 상태 값을 숫자형으로 변환.
    for column in NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # 기본 품질 지표 계산.
    missing_counts = data.isna().sum()
    duplicate_count = data.duplicated().sum()
    duplicate_key_count = data.duplicated(
        subset=["timestamp", "machine"],
    ).sum()

    daily_counts = (
        data.assign(date=data["timestamp"].dt.date)
        .groupby("date", dropna=False)
        .size()
        .rename("row_count")
    )

    numeric_summary = data[NUMERIC_COLUMNS].describe().transpose()

    status_counts = (
        data["status"]
        .fillna("<MISSING>")
        .value_counts(dropna=False)
        .rename("row_count")
    )

    alarm_counts = (
        data["alarm"]
        .fillna(-1)
        .value_counts(dropna=False)
        .sort_index()
        .rename("row_count")
    )

    machine_counts = (
        data["machine"]
        .fillna("<MISSING>")
        .value_counts(dropna=False)
        .rename("row_count")
    )

    # 시간 순서로 정렬한 복사본에서 측정 간격을 계산함.
    interval_data = data.sort_values(["machine", "timestamp"]).copy()
    interval_data["interval_seconds"] = (
        interval_data.groupby("machine")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    interval_summary = (
        interval_data["interval_seconds"]
        .describe()
        .rename("value")
    )

    # 일반적인 약 6초 간격에서 크게 벗어난 구간을 별도로 저장함.
    abnormal_intervals = interval_data[
        (interval_data["interval_seconds"] <= 0)
        | (interval_data["interval_seconds"] > 15)
    ][
        [
            "timestamp",
            "machine",
            "interval_seconds",
            "source_file",
        ]
    ]

    # 결과 CSV 보고서로 저장.
    missing_counts.rename("missing_count").to_csv(
        REPORT_DIR / "missing_counts.csv",
        encoding="utf-8-sig",
    )
    daily_counts.to_csv(
        REPORT_DIR / "daily_counts.csv",
        encoding="utf-8-sig",
    )
    numeric_summary.to_csv(
        REPORT_DIR / "numeric_summary.csv",
        encoding="utf-8-sig",
    )
    status_counts.to_csv(
        REPORT_DIR / "status_counts.csv",
        encoding="utf-8-sig",
    )
    alarm_counts.to_csv(
        REPORT_DIR / "alarm_counts.csv",
        encoding="utf-8-sig",
    )
    machine_counts.to_csv(
        REPORT_DIR / "machine_counts.csv",
        encoding="utf-8-sig",
    )
    interval_summary.to_csv(
        REPORT_DIR / "interval_summary.csv",
        encoding="utf-8-sig",
    )
    abnormal_intervals.to_csv(
        REPORT_DIR / "abnormal_intervals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n데이터 품질 검증이 완료되었습니다!")
    print(f"전체 행 수: {original_row_count:,}")
    print(f"전체 중복 행: {duplicate_count:,}")
    print(f"동일 timestamp·machine 중복: {duplicate_key_count:,}")
    print(f"시간 변환 실패: {data['timestamp'].isna().sum():,}")
    print(f"15초 초과 또는 0초 이하 간격: {len(abnormal_intervals):,}")
    print(f"보고서 저장 위치: {REPORT_DIR}")

    print("\n결측치가 존재하는 열:")
    missing_columns = missing_counts[missing_counts > 0]

    if missing_columns.empty:
        print("없음")
    else:
        print(missing_columns.to_string())

    print("\n설비별 행 수:")
    print(machine_counts.to_string())

    print("\n상태별 행 수:")
    print(status_counts.to_string())

    print("\n알람값별 행 수:")
    print(alarm_counts.to_string())


if __name__ == "__main__":
    main()