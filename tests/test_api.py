from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fairaudit.api.app import create_app
from fairaudit.config import Settings

CSV = b"y,p,s,sex\n" + b"".join(
    f"{y},{p},{s},{sex}\n".encode()
    for y, p, s, sex in [(1, 1, 0.9, "F"), (0, 0, 0.2, "F"), (1, 0, 0.4, "M"), (0, 0, 0.1, "M")]
    * 10
)


@pytest.fixture
def client(study_dir: Path) -> Iterator[TestClient]:
    settings = Settings(artifacts_dir=study_dir.parent, max_upload_mb=0.01, max_rows=100)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def empty_client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(Settings(artifacts_dir=tmp_path))) as test_client:
        yield test_client


def test_health(client: TestClient, empty_client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["case_study_available"] is True
    assert empty_client.get("/health").json()["case_study_available"] is False


def test_index_and_static(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "fairaudit" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert "/audit/csv" in client.get("/openapi.json").json()["paths"]


def test_audit_json_records(client: TestClient) -> None:
    records = [
        {"label": y, "score": s, "sex": g}
        for y, s, g in [(1, 0.8, "F"), (0, 0.6, "F"), (1, 0.4, "M"), (0, 0.1, "M")] * 5
    ]
    payload = {
        "records": records,
        "mapping": {"label": "label", "score": "score", "sensitive": ["sex"]},
    }
    response = client.post("/audit", json=payload)
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["threshold"] == 0.5
    sex = report["attributes"][0]
    assert {g["group"]: g["selection_rate"] for g in sex["groups"]} == {"F": 1.0, "M": 0.0}
    assert sex["passes_four_fifths_rule"] is False
    assert any("Disparate impact" in f for f in report["findings"])


def test_audit_json_validation_errors(client: TestClient) -> None:
    bad_mapping = {"records": [{"y": 1}], "mapping": {"label": "y", "sensitive": ["g"]}}
    assert client.post("/audit", json=bad_mapping).status_code == 422
    missing_column = {
        "records": [{"y": 1, "p": 1}],
        "mapping": {"label": "y", "prediction": "p", "sensitive": ["g"]},
    }
    response = client.post("/audit", json=missing_column)
    assert response.status_code == 422
    assert "Columns not found" in response.json()["detail"]
    too_many = {
        "records": [{"y": 1, "p": 1, "g": "a"}] * 101,
        "mapping": {"label": "y", "prediction": "p", "sensitive": ["g"]},
    }
    assert client.post("/audit", json=too_many).status_code == 413


def test_audit_csv_upload(client: TestClient) -> None:
    response = client.post(
        "/audit/csv",
        files={"file": ("preds.csv", CSV, "text/csv")},
        data={
            "label": "y",
            "prediction": "p",
            "score": "s",
            "sensitive": ["sex"],
            "threshold": "0.5",
        },
    )
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["n_samples"] == 40
    assert report["has_scores"] is True
    assert report["threshold"] is None  # predictions were supplied


def test_audit_csv_errors(client: TestClient) -> None:
    def post(data: dict[str, object], content: bytes = CSV) -> int:
        files = {"file": ("preds.csv", content, "text/csv")}
        return client.post("/audit/csv", files=files, data=data).status_code

    assert post({"label": "y", "sensitive": ["sex"]}) == 422  # neither prediction nor score
    assert post({"label": "y", "prediction": "p", "sensitive": ["nope"]}) == 422
    assert post({"label": "y", "prediction": "p", "sensitive": ["sex"]}, b"") == 422
    assert post({"label": "y", "prediction": "p", "sensitive": ["sex"]}, CSV * 20) == 413


def test_body_size_limit(client: TestClient) -> None:
    response = client.post(
        "/audit", content=b"x" * 200_000, headers={"content-type": "application/json"}
    )
    assert response.status_code == 413


def test_case_study_endpoints(client: TestClient) -> None:
    study = client.get("/case-study")
    assert study.status_code == 200
    assert len(study.json()["experiments"]) == 7
    report = client.get("/case-study/report")
    assert report.headers["content-type"].startswith("text/markdown")
    assert "not a clinical tool" in report.text
    assert client.get("/case-study/model-card").text.startswith("# Model card")
    sample = client.get("/case-study/sample.csv")
    assert sample.headers["content-type"].startswith("text/csv")
    assert sample.text.startswith("label,")


def test_case_study_missing(empty_client: TestClient) -> None:
    for path in (
        "/case-study",
        "/case-study/report",
        "/case-study/model-card",
        "/case-study/sample.csv",
    ):
        assert empty_client.get(path).status_code == 404
