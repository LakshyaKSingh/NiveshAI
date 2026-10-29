"""
Prediction and Strategy Signals Schemas
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    id: Optional[int] = None
    cycle_id: str
    symbol: str
    prob_up: float
    prob_neutral: float
    prob_down: float
    predicted_class: str
    model_version: str
    timestamp: datetime


class LatestPredictionsResponse(BaseModel):
    total: int
    predictions: List[PredictionItem]


class SignalItem(BaseModel):
    id: Optional[int] = None
    cycle_id: str
    symbol: str
    signal: str  # BUY, EXIT, HOLD
    reasoning: str
    threshold: float
    timestamp: datetime


class SignalsResponse(BaseModel):
    total: int
    signals: List[SignalItem]


class RunInferenceRequest(BaseModel):
    symbols: Optional[List[str]] = Field(default=None, description="Optional subset of symbols to evaluate")


class ActiveModelResponse(BaseModel):
    model_version: str
    algorithm: str
    num_features: int
    feature_names: List[str]
    classes: List[str]
    metrics: Dict[str, Any]
    artifact_path: str
    trained_at: Optional[str] = None
