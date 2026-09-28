# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Active Learning & Duty Officer Feedback Router
========================================================================
Endpoints:
- POST /api/v1/active-learning/feedback (and /api/active-learning/feedback)
- GET  /api/v1/active-learning/stats    (and /api/active-learning/stats)
- GET  /api/v1/active-learning/queue    (and /api/active-learning/queue)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Body
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from app.services.active_learning_service import ActiveLearningService

router = APIRouter(tags=["Active Learning & Operator Feedback"])


class OperatorFeedbackRequest(BaseModel):
    action: str = Field(..., description="Action: confirm_spill, reject_low_wind, reject_biogenic, reject_wake, manual_roi")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Model prediction confidence")
    model_prediction: str = Field(default="oil_spill", description="Prediction category from model")
    notes: Optional[str] = Field(default=None, description="Optional watchstander comments")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded SAR patch image")
    lat_lon: Optional[List[float]] = Field(default=None, description="[Latitude, Longitude] if georeferenced")
    area_coverage_percent: Optional[float] = Field(default=None, description="Detected or annotated spill area %")
    sensor_type: Optional[str] = Field(default="Sentinel-1", description="SAR or Web source modality")


@router.post("/feedback")
async def submit_feedback(payload: OperatorFeedbackRequest):
    """
    Submits duty officer ground-truth feedback to the active learning queue.
    """
    try:
        result = ActiveLearningService.record_feedback(
            action=payload.action,
            confidence=payload.confidence,
            model_prediction=payload.model_prediction,
            notes=payload.notes,
            image_base64=payload.image_base64,
            lat_lon=payload.lat_lon,
            area_coverage_percent=payload.area_coverage_percent,
            sensor_type=payload.sensor_type
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error processing feedback: {str(e)}")


@router.get("/stats")
async def get_active_learning_stats():
    """
    Retrieves active learning queue counts, mined hard negatives, and retraining readiness.
    """
    return ActiveLearningService.get_stats()


@router.get("/queue")
async def get_priority_queue(limit: int = 50):
    """
    Lists highest-priority feedback samples queued for fine-tuning.
    """
    return {
        "count": len(ActiveLearningService.get_high_priority_queue(limit)),
        "samples": ActiveLearningService.get_high_priority_queue(limit)
    }
