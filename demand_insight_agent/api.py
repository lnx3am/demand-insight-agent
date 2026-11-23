from fastapi import FastAPI, HTTPException
from openai import APIConnectionError, InternalServerError, RateLimitError

from .graph import run
from .models import RunRequest, RunResponse

app = FastAPI(title="Demand Insight Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=RunResponse)
def create_run(request: RunRequest) -> RunResponse:
    try:
        return RunResponse.model_validate(run(request))
    except (APIConnectionError, InternalServerError, RateLimitError) as exc:
        raise HTTPException(status_code=503, detail="model summary temporarily unavailable",
                            headers={"Retry-After": "1"}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
