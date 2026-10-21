"""FastAPI application: fairness audits over HTTP, the ChestX-ray14 case study and a web UI."""

import io
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from fairaudit import __version__
from fairaudit.api.schemas import AuditRequest, ErrorResponse, HealthResponse
from fairaudit.audit import ColumnMapping, audit_dataframe
from fairaudit.casestudy.results import (
    MODEL_CARD_FILE,
    PREDICTIONS_FILE,
    REPORT_FILE,
    RESULTS_FILE,
    StudyResults,
)
from fairaudit.config import Settings, get_settings
from fairaudit.errors import AuditInputError
from fairaudit.log import configure_logging
from fairaudit.report import AuditPolicy, AuditReport

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
_MULTIPART_OVERHEAD = 64 * 1024
_ERRORS: dict[int | str, dict[str, object]] = {
    413: {"model": ErrorResponse, "description": "Upload or row limit exceeded."},
    422: {"description": "Invalid mapping, columns or values."},
}
_NO_CASE_STUDY = "Case-study artifacts not found: run `fairaudit train` first."


def _load_case_study(directory: Path) -> StudyResults | None:
    if not (directory / RESULTS_FILE).exists():
        logger.warning(
            "No case-study results in %s; /case-study endpoints will return 404", directory
        )
        return None
    results = StudyResults.load(directory)
    logger.info("Loaded case-study results (%d experiments)", len(results.experiments))
    return results


def _parse_csv(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise AuditInputError(f"Could not parse the CSV file: {exc}") from exc


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    max_bytes = int(settings.max_upload_mb * 1024 * 1024)
    case_study_dir = settings.case_study_dir

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.case_study = _load_case_study(case_study_dir)
        yield

    app = FastAPI(
        title="fairaudit",
        version=__version__,
        summary="Group-fairness audits for binary classifiers.",
        description=(
            "Upload labels, predictions (or scores) and sensitive attributes to get per-group "
            "rates, disparity metrics (disparate impact, demographic parity, equal opportunity, "
            "equalized odds), calibration by group and plain-English findings. "
            "The `/case-study` endpoints expose a reproducible study on NIH ChestX-ray14 "
            "metadata, a fairness case study and not a clinical tool."
        ),
        lifespan=lifespan,
    )
    app.state.case_study = None
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.middleware("http")
    async def limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        length = request.headers.get("content-length", "")
        if length.isdigit() and int(length) > max_bytes + _MULTIPART_OVERHEAD:
            return JSONResponse(
                {"detail": f"Request body exceeds {settings.max_upload_mb:g} MB"}, status_code=413
            )
        return await call_next(request)

    @app.exception_handler(AuditInputError)
    async def audit_input_error(_: Request, exc: AuditInputError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    def _audit(
        df: pd.DataFrame, mapping: ColumnMapping, threshold: float, policy: AuditPolicy
    ) -> AuditReport:
        if len(df) > settings.max_rows:
            raise HTTPException(413, f"{len(df)} rows exceed the limit of {settings.max_rows}")
        started = time.perf_counter()
        report = audit_dataframe(df, mapping, threshold=threshold, policy=policy)
        logger.info(
            "Audited %d rows, attributes=%s, findings=%d in %.0f ms",
            report.n_samples,
            mapping.sensitive,
            len(report.findings),
            (time.perf_counter() - started) * 1000,
        )
        return report

    @app.get("/health", response_model=HealthResponse, tags=["service"])
    def health(request: Request) -> HealthResponse:
        """Liveness probe; also reports whether the case-study results are available."""
        return HealthResponse(
            version=__version__, case_study_available=request.app.state.case_study is not None
        )

    @app.post("/audit", response_model=AuditReport, responses=_ERRORS, tags=["audit"])
    def audit_records(body: AuditRequest) -> AuditReport:
        """Audit samples sent as JSON records."""
        return _audit(
            pd.DataFrame.from_records(body.records), body.mapping, body.threshold, body.policy
        )

    @app.post("/audit/csv", response_model=AuditReport, responses=_ERRORS, tags=["audit"])
    async def audit_csv(
        file: Annotated[UploadFile, File(description="CSV file with one row per sample.")],
        label: Annotated[str, Form(description="Ground-truth column.")],
        sensitive: Annotated[
            list[str], Form(description="Sensitive-attribute column; repeat the field for several.")
        ],
        prediction: Annotated[str | None, Form(description="Binary prediction column.")] = None,
        score: Annotated[str | None, Form(description="Probability column, in [0, 1].")] = None,
        positive_label: Annotated[
            str | None, Form(description="Positive-class value for non-binary columns.")
        ] = None,
        threshold: Annotated[float, Form(ge=0, le=1, description="Applied to scores.")] = 0.5,
    ) -> AuditReport:
        """Audit an uploaded CSV file (multipart form)."""
        try:
            mapping = ColumnMapping(
                label=label,
                prediction=prediction or None,
                score=score or None,
                sensitive=[c for c in sensitive if c],
                positive_label=positive_label or None,
            )
        except ValidationError as exc:
            raise HTTPException(422, "; ".join(e["msg"] for e in exc.errors())) from exc
        content = await file.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb:g} MB")
        df = await run_in_threadpool(_parse_csv, content)
        return await run_in_threadpool(_audit, df, mapping, threshold, AuditPolicy())

    def _case_study(request: Request) -> StudyResults:
        results: StudyResults | None = request.app.state.case_study
        if results is None:
            raise HTTPException(404, _NO_CASE_STUDY)
        return results

    def _case_study_file(name: str) -> Path:
        path = case_study_dir / name
        if not path.exists():
            raise HTTPException(404, _NO_CASE_STUDY)
        return path

    @app.get("/case-study", response_model=StudyResults, tags=["case study"])
    def case_study(request: Request) -> StudyResults:
        """Full results of the ChestX-ray14 metadata case study (every experiment's audit)."""
        return _case_study(request)

    @app.get("/case-study/report", response_class=PlainTextResponse, tags=["case study"])
    def case_study_report() -> PlainTextResponse:
        """The case-study report, as Markdown."""
        text = _case_study_file(REPORT_FILE).read_text(encoding="utf-8")
        return PlainTextResponse(text, media_type="text/markdown")

    @app.get("/case-study/model-card", response_class=PlainTextResponse, tags=["case study"])
    def case_study_model_card() -> PlainTextResponse:
        """The model card of the case-study models, as Markdown."""
        text = _case_study_file(MODEL_CARD_FILE).read_text(encoding="utf-8")
        return PlainTextResponse(text, media_type="text/markdown")

    @app.get("/case-study/sample.csv", response_class=FileResponse, tags=["case study"])
    def case_study_sample() -> FileResponse:
        """Test-set labels, sensitive attributes and every model's predictions (CSV)."""
        return FileResponse(
            _case_study_file(PREDICTIONS_FILE), media_type="text/csv", filename="chestxray_test.csv"
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
