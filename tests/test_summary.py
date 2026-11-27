import pytest

from demand_insight_agent.graph import deterministic_summary
from demand_insight_agent.models import Product
from demand_insight_agent.scoring import assess


@pytest.mark.parametrize("use_llm_summary", [False, True])
@pytest.mark.parametrize("configured", [False, True])
def test_empty_catalog_returns_template_summary_without_provider_calls(
    monkeypatch, use_llm_summary, configured,
):
    from importlib import import_module

    from fastapi.testclient import TestClient

    from demand_insight_agent.api import app

    monkeypatch.setattr(import_module("demand_insight_agent.graph"), "_catalog", list)
    for variable in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if configured:
            monkeypatch.setenv(variable, "test-only")
        else:
            monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr("langchain_openai.ChatOpenAI",
                        lambda **kwargs: pytest.fail("empty results must not call the provider"))
    response = TestClient(app).post("/runs", json={"use_llm_summary": use_llm_summary})
    assert response.status_code == 200
    assert response.json()["assessments"] == []
    assert response.json()["summary"] == "No products matched this run."


def test_summary_labels_high_scoring_product_rejected_by_margin_gate():
    result = assess(Product(
        product_id="low-margin", name="Low margin product", category="demo",
        demand_signal=1, advertiser_count=8, gross_margin=0.1,
        market_saturation=0, rating=5, evidence_count=15,
    ))
    assert result.score >= 70
    assert result.verdict == "reject"
    summary = deterministic_summary({"assessments": [result.model_dump()]})["summary"]
    assert f"Low margin product ({result.score:.1f}, reject)" in summary
    assert summary.startswith("Top ranked products")


def test_model_summary_extracts_text_from_string_and_content_blocks(monkeypatch):
    from unittest.mock import Mock

    from langchain_core.messages import AIMessage

    from demand_insight_agent.graph import model_summary

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    text = "The top product is promising. Evidence is limited."
    for content in [text, [{"type": "reasoning", "reasoning": "private reasoning"},
                           {"type": "text", "text": "The top product is promising. "},
                           {"type": "text", "text": "Evidence is limited."}]]:
        model = Mock()
        model.invoke.return_value = AIMessage(content=content)
        factory = Mock(return_value=model)
        monkeypatch.setattr("langchain_openai.ChatOpenAI", factory)
        assert model_summary({"assessments": []}) == {"summary": text}
        factory.assert_called_once_with(model="test-model", temperature=0, timeout=20)
        model.invoke.assert_called_once()


@pytest.mark.parametrize("content", ["", " \t\n", [{"type": "reasoning", "reasoning": "private"}]])
def test_model_summary_rejects_responses_without_summary_text(monkeypatch, content):
    from unittest.mock import Mock

    from langchain_core.messages import AIMessage

    from demand_insight_agent.graph import model_summary

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    model = Mock()
    model.invoke.return_value = AIMessage(content=content)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(return_value=model))
    with pytest.raises(ValueError, match="no summary text"):
        model_summary({"assessments": []})


@pytest.mark.parametrize("failure", ["connection", "timeout", "rate_limit", "server"])
def test_temporary_summary_failure_returns_retryable_response(monkeypatch, failure):
    from unittest.mock import Mock

    import httpx
    from fastapi.testclient import TestClient
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    from demand_insight_agent.api import app

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    request = httpx.Request("POST", "https://provider.example.test/responses")
    if failure == "connection":
        error = APIConnectionError(request=request)
    elif failure == "timeout":
        error = APITimeoutError(request=request)
    else:
        status, error_type = (429, RateLimitError) if failure == "rate_limit" else (
            503, InternalServerError)
        error = error_type("private provider detail", body=None,
                           response=httpx.Response(status, request=request))
    model = Mock()
    model.invoke.side_effect = error
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(return_value=model))
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/runs", json={"use_llm_summary": True})
    assert response.status_code == 503
    assert response.json() == {"detail": "model summary temporarily unavailable"}
    assert response.headers["retry-after"] == "1"
    model.invoke.assert_called_once()
    # A provider outage must not prevent local scoring and template summaries.
    assert client.post("/runs", json={}).status_code == 200
    model.invoke.assert_called_once()


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_summary_provider_failure_returns_sanitized_bad_gateway(monkeypatch, status):
    from unittest.mock import Mock

    import httpx
    from fastapi.testclient import TestClient
    from openai import (
        AuthenticationError,
        BadRequestError,
        NotFoundError,
        PermissionDeniedError,
        UnprocessableEntityError,
    )

    from demand_insight_agent.api import app

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    error_type = {400: BadRequestError, 401: AuthenticationError, 403: PermissionDeniedError,
                  404: NotFoundError, 422: UnprocessableEntityError}[status]
    error = error_type("private provider detail", body={"secret": "private"},
                       response=httpx.Response(status, request=httpx.Request(
                           "POST", "https://provider.example.test/responses")))
    model = Mock()
    model.invoke.side_effect = error
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(return_value=model))
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/runs", json={"use_llm_summary": True})
    assert response.status_code == 502
    assert response.json() == {"detail": "model summary provider request failed"}
    assert "retry-after" not in response.headers
    model.invoke.assert_called_once()
    assert client.post("/runs", json={}).status_code == 200
    model.invoke.assert_called_once()


@pytest.mark.parametrize("content", ["", " \t\n", [{"type": "reasoning", "reasoning": "private"}]])
def test_missing_model_summary_returns_bad_gateway_and_local_runs_still_work(monkeypatch, content):
    from unittest.mock import Mock

    from fastapi.testclient import TestClient
    from langchain_core.messages import AIMessage

    from demand_insight_agent.api import app

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    model = Mock()
    model.invoke.return_value = AIMessage(content=content)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", Mock(return_value=model))
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/runs", json={"use_llm_summary": True})
    assert response.status_code == 502
    assert response.json() == {"detail": "model returned no summary text"}
    assert "retry-after" not in response.headers
    model.invoke.assert_called_once()
    assert client.post("/runs", json={}).status_code == 200
    model.invoke.assert_called_once()
    # Caller errors must retain their existing status.
    assert client.post("/runs", json={"mode": "shortlist", "product_ids": ["unknown"]}).status_code == 422
