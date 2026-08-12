from pathlib import Path

import pandas as pd


# 프로젝트 기준 경로
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "output" / "mct_parsed.csv"
OUTPUT_PATH = PROJECT_DIR / "output" / "mct_clean.csv"

# 정상 수집 간격이 약 5~6초이므로 15초 초과를 수집 공백으로 표시
INTERVAL_THRESHOLD_SECONDS = 15

def main():
    # 파싱 완료 데이터 불러오기
    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    original_rows = len(df)

    # 시간 계산을 위해 timestamp를 datetime으로 변환
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%Y-%m-%d %H:%M:%S,%f",
        errors="raise",
    )

    # 설비별 시간순으로 정렬
    df = df.sort_values(["machine", "timestamp"]).reset_index(drop=True)

    # 서로 다른 설비의 행이 연결되지 않도록 설비별 수집 간격 계산
    df["interval_seconds"] = (
        df.groupby("machine")["timestamp"]
        .diff()
        .dt.total_seconds()
    )    

    # 원본 좌표는 스케일 팩터 10000이 적용된 정수이므로 mm 단위 열 생성
    df["x_axis_mm"] = df["x_axis"] / 10000
    df["y_axis_mm"] = df["y_axis"] / 10000
    df["z_axis_mm"] = df["z_axis"] / 10000
        
    # 분석용 상태 플래그 추가
    df["is_alarm"] = df["alarm"].ne(0)

    # 코드 1은 실제 비상, 코드 2는 RESET 상태로 구분
    df["is_emergency"] = df["emergency"].eq(1)
    df["is_reset"] = df["emergency"].eq(2)

    df["is_spindle_overload"] = df["spindle_load"].gt(100)
    df["is_abnormal_interval"] = df["interval_seconds"].gt(
        INTERVAL_THRESHOLD_SECONDS
    )

    # 어느 하나라도 해당하면 관심 관측치로 표시
    df["has_event"] = df[
        [
            "is_alarm",
            "is_emergency",
            "is_reset",
            "is_spindle_overload",
            "is_abnormal_interval",
        ]
    ].any(axis=1)

    # CSV에서 기존과 같은 timestamp 형식 유지
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S,%f")
    df["timestamp"] = df["timestamp"].str[:-3]

    # 원본 행이 손실되지 않았는지 검증
    if len(df) != original_rows:
        raise RuntimeError("정제 과정에서 행 수가 변경되었습니다.")

    # 최종 분석용 데이터 저장
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print("[최종 정제 데이터 생성 완료]")
    print(f"입력 행 수: {original_rows:,}")
    print(f"출력 행 수: {len(df):,}")
    print(f"알람 행 수: {df['is_alarm'].sum():,}")
    print(f"실제 비상 상태 행 수: {df['is_emergency'].sum():,}")
    print(f"RESET 상태 행 수: {df['is_reset'].sum():,}")
    print(f"스핀들 부하 100% 초과: {df['is_spindle_overload'].sum():,}")
    print(f"수집 간격 15초 초과: {df['is_abnormal_interval'].sum():,}")
    print(f"전체 관심 관측치: {df['has_event'].sum():,}")
    print(f"저장 위치: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()