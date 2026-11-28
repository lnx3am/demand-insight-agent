from fastapi import FastAPI, HTTPException
from openai import APIConnectionError, APIStatusError

from .graph import InvalidSummaryError, run
from .models import RunRequest, RunResponse

app = FastAPI(title="Demand Insight Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/runs", response_model=RunResponse)
def create_run(request: RunRequest) -> RunResponse:
    try:
        return RunResponse.model_validate(run(request))
    except InvalidSummaryError as exc:
        raise HTTPException(status_code=502, detail="model returned no summary text") from exc
    except (APIConnectionError, APIStatusError) as exc:
        if (isinstance(exc, APIConnectionError) or exc.status_code >= 500
                or exc.status_code in (408, 409, 429)):
            raise HTTPException(status_code=503, detail="model summary temporarily unavailable",
                                headers={"Retry-After": "1"}) from exc
        raise HTTPException(status_code=502, detail="model summary provider request failed") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
