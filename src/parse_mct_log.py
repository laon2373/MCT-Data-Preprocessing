import csv
import re
from pathlib import Path


# 프로젝트 폴더 경로
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "mct_parsed.csv"

# 일반 로그 형식:
# 2026-05-01 00:00:04,522 [MCT_MCT3호] FEED_RATE: 392
LOG_PATTERN = re.compile(
    r"^(?P<timestamp>.*?)\s+\[(?P<machine>[^\]]+)]\s+"
    r"(?P<key>[^:]+):\s*(?P<value>.*)$"
)

# STATUS 줄 내부 형식
STATUS_PATTERN = re.compile(
    r"^(?P<status>[^,]*),\s*"
    r"AUT_MODE:\s*(?P<aut_mode>[^,]*),\s*"
    r"RUN_MODE:\s*(?P<run_mode>[^,]*),\s*"
    r"EMERGENCY:\s*(?P<emergency>[^,]*),\s*"
    r"ALARM:\s*(?P<alarm>[^,]*),\s*"
    r"ERROR_MESSAGE:\s*(?P<error_message>.*)$"
)

# 출력 CSV 열 순서
FIELDNAMES = [
    "timestamp",
    "machine",
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
    "status",
    "aut_mode",
    "run_mode",
    "emergency",
    "alarm",
    "error_message",
    "source_file",
]

# 로그 키와 CSV 열 이름 연결
KEY_MAPPING = {
    "FEED_RATE": "feed_rate",
    "SPINDLE_RATE": "spindle_rate",
    "X_AXIS": "x_axis",
    "Y_AXIS": "y_axis",
    "Z_AXIS": "z_axis",
    "CURRENT_TOOL": "current_tool",
    "X LOAD": "x_load",
    "Y LOAD": "y_load",
    "Z LOAD": "z_load",
    "SPINDLE_LOAD": "spindle_load",
    "SPINDLE_RPM": "spindle_rpm",
    "DELTA_SPINDLE_RPM": "delta_spindle_rpm",
}


def clean_numeric_value(value):
    """숫자 데이터에서 퍼센트 기호와 공백을 제거합니다."""
    return value.replace("%", "").strip()


def parse_status(value):
    """STATUS 줄을 운전 상태, 모드, 알람 정보로 분리합니다."""
    match = STATUS_PATTERN.match(value)

    if not match:
        return {
            "status": value.strip(),
            "aut_mode": "",
            "run_mode": "",
            "emergency": "",
            "alarm": "",
            "error_message": "",
        }

    return {
        key: item.strip()
        for key, item in match.groupdict().items()
    }


def parse_log_file(log_file, writer):
    """로그 파일 하나를 읽어 측정 묶음별 CSV 행으로 변환합니다."""
    current_record = {}
    written_rows = 0
    skipped_lines = 0

    # 대용량 파일이니 한 줄씩 읽어 메모리 사용량 줄임.
    with log_file.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            match = LOG_PATTERN.match(line.strip())

            if not match:
                skipped_lines += 1
                continue

            timestamp = match.group("timestamp")
            machine = match.group("machine")
            key = match.group("key").strip()
            value = match.group("value").strip()

            # 새 측정 묶음의 시작값을 기록.
            if key == "FEED_RATE":
                # 이전 묶음이 STATUS 없이 종료됐다면 저장하지 않고 새로 시작.
                current_record = {
                    "timestamp": timestamp,
                    "machine": machine,
                    "source_file": log_file.name,
                }

            if not current_record:
                # 파일 시작 부분이 측정 묶음 중간이면 다음 FEED_RATE까지 건너뜀.
                continue

            if key in KEY_MAPPING:
                column_name = KEY_MAPPING[key]
                current_record[column_name] = clean_numeric_value(value)

            elif key == "STATUS":
                current_record.update(parse_status(value))

                # STATUS를 측정 묶음의 끝으로 보고 CSV 한 행을 저장.
                writer.writerow(current_record)
                written_rows += 1
                current_record = {}

    return written_rows, skipped_lines


def main():
    """data 폴더의 모든 로그 파일을 찾아 하나의 CSV로 통합합니다."""
    log_files = sorted(DATA_DIR.rglob("*.log"))

    if not log_files:
        print(f"로그 파일을 찾지 못했습니다: {DATA_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_skipped_lines = 0

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()

        for index, log_file in enumerate(log_files, start=1):
            rows, skipped_lines = parse_log_file(log_file, writer)
            total_rows += rows
            total_skipped_lines += skipped_lines

            print(
                f"[{index}/{len(log_files)}] {log_file.name}: "
                f"{rows:,}행 변환, {skipped_lines:,}줄 제외"
            )

    print("\n전처리가 완료되었습니다!")
    print(f"처리한 로그 파일: {len(log_files)}개")
    print(f"생성한 데이터: {total_rows:,}행")
    print(f"형식 불일치 줄: {total_skipped_lines:,}줄")
    print(f"출력 파일: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()