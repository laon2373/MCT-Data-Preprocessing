# -*- coding: utf-8 -*-
"""
MCT 예지보전 PoC 통합 실행기

실행:
    python run_app.py

한 번의 명령으로:
1) XGBoost FastAPI (127.0.0.1:8000)
2) 통합 Streamlit 대시보드 (기본 8501)

를 함께 실행한다.

M2* 월간 ZIP 분석은 Streamlit 내부 batch pipeline을 사용하고,
XGBoost 분석은 FastAPI /predict를 사용한다.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
STREAMLIT_APP = (
    PROJECT_ROOT
    / "mct_poc_dashboard_common"
    / "dashboard"
    / "streamlit_integrated.py"
)

API_MODULE = "api.main:app"
API_HOST = "127.0.0.1"
API_PORT = 8000
API_HEALTH = f"http://{API_HOST}:{API_PORT}/health"


def wait_for_api(timeout_sec: float = 30.0) -> bool:
    """FastAPI가 실제 요청을 받을 수 있을 때까지 잠시 기다린다."""
    deadline = time.time() + timeout_sec

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(API_HEALTH, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass

        time.sleep(0.5)

    return False


def terminate_process(proc: subprocess.Popen | None):
    if proc is None or proc.poll() is not None:
        return

    proc.terminate()

    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    if not STREAMLIT_APP.exists():
        raise FileNotFoundError(
            f"통합 Streamlit 파일을 찾지 못했습니다: {STREAMLIT_APP}"
        )

    print("=" * 68)
    print("MCT 3호 예지보전 PoC")
    print("=" * 68)
    print("1. XGBoost FastAPI 시작")
    print(f"   {API_HEALTH}")

    api_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            API_MODULE,
            "--host",
            API_HOST,
            "--port",
            str(API_PORT),
        ],
        cwd=PROJECT_ROOT,
    )

    streamlit_proc = None

    try:
        if not wait_for_api():
            terminate_process(api_proc)
            raise RuntimeError(
                "FastAPI가 30초 안에 시작되지 않았습니다. "
                "위쪽 FastAPI 오류 메시지를 확인해주세요."
            )

        print("   FastAPI 연결 완료")
        print()
        print("2. 통합 Streamlit 대시보드 시작")
        print("   브라우저가 자동으로 열립니다.")
        print()
        print("종료하려면 이 터미널에서 Ctrl+C를 누르세요.")
        print("=" * 68)

        streamlit_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(STREAMLIT_APP),
            ],
            cwd=PROJECT_ROOT,
        )

        # Streamlit이 종료될 때까지 실행기 유지
        return_code = streamlit_proc.wait()

        if return_code != 0:
            raise RuntimeError(
                f"Streamlit이 종료 코드 {return_code}로 종료되었습니다."
            )

    except KeyboardInterrupt:
        print("\n서비스를 종료합니다.")

    finally:
        terminate_process(streamlit_proc)
        terminate_process(api_proc)


if __name__ == "__main__":
    main()
