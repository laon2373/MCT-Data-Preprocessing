from pathlib import Path
import sys
import tempfile
import shutil
import xgboost as xgb

from fastapi import FastAPI, UploadFile, File, HTTPException


# =========================================================
# 프로젝트 경로 설정
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference_pipeline import (
    run_inference,
    load_model_artifacts,
)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI(
    title="MCT 예지보전 API",
    version="0.1.0",
)


# =========================================================
# Health Check
# =========================================================

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "MCT Predictive Maintenance API is running",
    }


# =========================================================
# MCT LOG 추론
# =========================================================

@app.post("/predict")
def predict_log(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # 1. 파일명 확인
    # -----------------------------------------------------

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="파일명이 없습니다.",
        )

    # -----------------------------------------------------
    # 2. .log 파일만 허용
    # -----------------------------------------------------

    suffix = Path(
        file.filename
    ).suffix.lower()

    if suffix != ".log":
        raise HTTPException(
            status_code=400,
            detail=".log 파일만 업로드할 수 있습니다.",
        )

    temp_path = None

    try:

        # -------------------------------------------------
        # 3. 업로드 파일 임시 저장
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".log",
        ) as temp_file:

            shutil.copyfileobj(
                file.file,
                temp_file,
            )

            temp_path = Path(
                temp_file.name
            )

        # -------------------------------------------------
        # 4. 기존 추론 파이프라인 실행
        # -------------------------------------------------

        output = run_inference(
            temp_path
        )

        result_df = output[
            "results"
        ].copy()

        skipped_df = output[
            "skipped_windows"
        ].copy()

        # -------------------------------------------------
        # 5. 지원 가능한 Window
        # -------------------------------------------------

        if result_df.empty:

            supported_df = result_df

        else:

            supported_df = result_df[
                result_df["supported"]
            ].copy()

        # -------------------------------------------------
        # 6. Anomaly Score 통계
        # -------------------------------------------------

        if (
            not supported_df.empty
            and
            "anomaly_score"
            in supported_df.columns
        ):

            scores = (
                supported_df[
                    "anomaly_score"
                ]
                .dropna()
            )

        else:

            scores = []

        if len(scores) > 0:

            score_mean = float(
                scores.mean()
            )

            score_max = float(
                scores.max()
            )

            max_index = (
                supported_df[
                    "anomaly_score"
                ]
                .idxmax()
            )

            max_row = supported_df.loc[
                max_index
            ]

            highest_window = {
                "window_start": str(
                    max_row[
                        "window_start"
                    ]
                ),
                "window_end": str(
                    max_row[
                        "window_end"
                    ]
                ),
                "current_tool": int(
                    max_row[
                        "current_tool"
                    ]
                ),
                "anomaly_score": float(
                    max_row[
                        "anomaly_score"
                    ]
                ),
            }

            # -------------------------------------------------
            # 최고점 Window의 XGBoost Feature 기여도
            # -------------------------------------------------

            explain_model, feature_columns, _ = (
                load_model_artifacts()
            )

            explain_X = (
                max_row[
                    feature_columns
                ]
                .astype(float)
                .to_frame()
                .T
            )

            dmatrix = xgb.DMatrix(
                explain_X,
                feature_names=feature_columns,
            )

            contributions = (
                explain_model
                .get_booster()
                .predict(
                    dmatrix,
                    pred_contribs=True,
                    iteration_range=(
                        0,
                        explain_model.best_iteration + 1,
                    ),
                )[0]
            )

            feature_contributions = []

            for feature_name, contribution in zip(
                feature_columns,
                contributions[:-1],
            ):

                feature_contributions.append(
                    {
                        "feature": feature_name,
                        "contribution": float(
                            contribution
                        ),
                    }
                )


            # 이상 점수를 높이는 방향의 Feature만 선택
            positive_contributions = [
                item
                for item in feature_contributions
                if item["contribution"] > 0
            ]

            positive_contributions = sorted(
                positive_contributions,
                key=lambda x: x["contribution"],
                reverse=True,
            )

            top_contributions = (
                positive_contributions[:3]
            )


            # -------------------------------------------------
            # Anomaly Score 상위 10개 Window
            # -------------------------------------------------

            top_df = (
                supported_df
                .sort_values(
                    "anomaly_score",
                    ascending=False,
                )
                .head(10)
            )

            top_windows = []

            for _, row in top_df.iterrows():

                top_windows.append(
                    {
                        "window_start": str(
                            row["window_start"]
                        ),
                        "window_end": str(
                            row["window_end"]
                        ),
                        "current_tool": int(
                            row["current_tool"]
                        ),
                        "sample_count": int(
                            row["sample_count"]
                        ),
                        "anomaly_score": float(
                            row["anomaly_score"]
                        ),
                        "load_vs_normal": float(
                            row["load_vs_normal"]
                        ),
                        "rpm_std_vs_normal": float(
                            row["rpm_std_vs_normal"]
                        ),
                        "rpm_error_vs_normal": float(
                            row["rpm_error_vs_normal"]
                        ),
                    }
                )

        else:

            score_mean = None
            score_max = None
            highest_window = None

            top_windows = []
            top_contributions = []

        # -------------------------------------------------
        # 7. API 결과 반환
        # -------------------------------------------------

        return {
            "filename": file.filename,

            "parsed_rows": int(
                output[
                    "written_rows"
                ]
            ),

            "clean_rows": int(
                output[
                    "clean_rows"
                ]
            ),

            "cutting_rows": int(
                output[
                    "cutting_rows"
                ]
            ),

            "result_windows": int(
                len(
                    result_df
                )
            ),

            "supported_windows": int(
                len(
                    supported_df
                )
            ),

            "unsupported_windows": int(
                len(
                    result_df
                )
                -
                len(
                    supported_df
                )
            ),

            "insufficient_windows": int(
                len(
                    skipped_df
                )
            ),

            "score_mean": score_mean,

            "score_max": score_max,

            "highest_window": highest_window,

            "top_windows": top_windows,

            "top_contributions": top_contributions,
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"분석 중 오류가 발생했습니다: {str(e)}",
        )

    finally:

        # -------------------------------------------------
        # 8. 임시 파일 삭제
        # -------------------------------------------------

        if (
            temp_path is not None
            and
            temp_path.exists()
        ):
            temp_path.unlink()

        file.file.close()