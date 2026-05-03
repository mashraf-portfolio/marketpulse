"""Pydantic v2 request/response models for FastAPI."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ForecastPriceRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol from the whitelist", examples=["AAPL"])
    horizon: Literal[1, 7] = Field(1, description="Forecast horizon in trading days")
    model_type: Literal["arima", "prophet", "lstm", "tft"] = "arima"
    include_attention: bool = Field(False, description="If True and model_type=tft, include attention weights")


class ForecastPriceResponse(BaseModel):
    point_forecast: float
    lower: float | None = None
    upper: float | None = None
    confidence_alpha: float = 0.1
    model_used: str
    model_version: str
    attention: dict | None = None


class ForecastVolatilityRequest(BaseModel):
    ticker: str
    model_type: Literal["garch", "lstm_vol"] = "garch"


class ForecastVolatilityResponse(BaseModel):
    predicted_log_vol: float
    predicted_vol: float
    high_vol_probability: float
    high_vol_prediction: bool
    threshold_used: float
    model_used: str


class ClassifyRegimeRequest(BaseModel):
    ticker: str
    model_type: Literal["hmm", "xgboost"] = "xgboost"


class ClassifyRegimeResponse(BaseModel):
    predicted_regime: Literal["bull", "bear", "sideways"]
    probability_vector: dict[str, float]
    model_used: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: list[str]
    uptime_seconds: float
    build_id: str = "dev"


class ModelInfoResponse(BaseModel):
    head1: dict
    head2: dict
    head3: dict
    pytorch_available: bool
