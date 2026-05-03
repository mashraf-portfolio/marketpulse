"""FastAPI application with @asynccontextmanager lifespan."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Detect PyTorch availability at module import
try:
    import torch  # noqa: F401
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load model registry. Shutdown: nothing (uvicorn handles process termination)."""
    app.state.start_time = time.time()
    app.state.registry = None  # ModelRegistry.load_from(...) implemented in Phase 5
    app.state.pytorch_available = PYTORCH_AVAILABLE
    yield


app = FastAPI(title="MarketPulse API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "degraded",
        "models_loaded": [],
        "uptime_seconds": time.time() - app.state.start_time,
        "build_id": "dev",
    }


@app.get("/model/info")
async def model_info():
    raise HTTPException(status_code=501, detail="Not implemented until Phase 5")


@app.post("/forecast/price")
async def forecast_price():
    raise HTTPException(status_code=501, detail="Not implemented until Phase 5")


@app.post("/forecast/volatility")
async def forecast_volatility():
    raise HTTPException(status_code=501, detail="Not implemented until Phase 5")


@app.post("/classify/regime")
async def classify_regime():
    raise HTTPException(status_code=501, detail="Not implemented until Phase 5")
