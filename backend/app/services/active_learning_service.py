# -*- coding: utf-8 -*-
"""
Spill Sense (SIH-26143) — Active Learning & Operator Feedback Engine
===================================================================
Implements Plan 2.0 Phase 4 components:
1. Ground-truth feedback intake from maritime operators / watchstanders.
2. Hard negative mining (Low-wind calm sea, biogenic surfactants, ship wakes).
3. Epistemic uncertainty scoring & sample triaging.
4. Secure local active learning buffer for continuous model fine-tuning.
"""

import os
import json
import time
import uuid
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data" / "active_learning"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = DATA_DIR / "feedback_cache.json"


class ActiveLearningService:
    """
    Manages operator feedback, uncertainty tracking, and training replay queue.
    """

    VALID_ACTIONS = {
        "confirm_spill": "True Positive (Confirmed Hydrocarbon)",
        "reject_low_wind": "Hard Negative (Calm Sea / Low Wind)",
        "reject_biogenic": "Hard Negative (Algae / Plant Surfactant)",
        "reject_wake": "Hard Negative (Ship Wake / Turbulent Water)",
        "manual_roi": "Operator Corrected Boundary"
    }

    @classmethod
    def _read_cache(cls) -> List[Dict[str, Any]]:
        if not CACHE_FILE.exists():
            return []
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    @classmethod
    def _write_cache(cls, records: List[Dict[str, Any]]) -> None:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

    @classmethod
    def record_feedback(
        cls,
        action: str,
        confidence: float,
        model_prediction: str,
        notes: Optional[str] = None,
        image_base64: Optional[str] = None,
        lat_lon: Optional[List[float]] = None,
        area_coverage_percent: Optional[float] = None,
        sensor_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Records a duty officer feedback event.
        """
        if action not in cls.VALID_ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Must be one of: {list(cls.VALID_ACTIONS.keys())}")

        feedback_id = f"fb_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Save image patch if provided
        saved_img_path = None
        if image_base64:
            try:
                # Strip data URL prefix if present
                if "," in image_base64:
                    image_base64 = image_base64.split(",", 1)[1]
                img_bytes = base64.b64decode(image_base64)
                img_filename = f"{feedback_id}.png"
                img_path = DATA_DIR / img_filename
                img_path.write_bytes(img_bytes)
                saved_img_path = str(img_path.relative_to(PROJECT_ROOT))
            except Exception as e:
                print(f"Warning: Failed to persist feedback image patch: {e}")

        # Compute priority score: highest for hard negatives and near-boundary predictions
        # (predictions near threshold 0.35 have maximum epistemic uncertainty)
        uncertainty = 1.0 - abs(confidence - 0.35) * 2.0
        uncertainty = max(0.1, min(1.0, uncertainty))

        is_hard_negative = action.startswith("reject_")
        priority = uncertainty * (1.5 if is_hard_negative else 1.0)

        record = {
            "id": feedback_id,
            "timestamp": timestamp,
            "action": action,
            "label_description": cls.VALID_ACTIONS[action],
            "model_prediction": model_prediction,
            "confidence": round(confidence, 4),
            "uncertainty_score": round(uncertainty, 4),
            "priority_score": round(priority, 4),
            "is_hard_negative": is_hard_negative,
            "area_coverage_percent": area_coverage_percent,
            "lat_lon": lat_lon,
            "sensor_type": sensor_type or "Universal-SAR",
            "notes": notes,
            "image_path": saved_img_path
        }

        records = cls._read_cache()
        records.append(record)
        cls._write_cache(records)

        return {
            "success": True,
            "feedback_id": feedback_id,
            "record": record,
            "total_cached": len(records)
        }

    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """
        Retrieves active learning queue statistics.
        """
        records = cls._read_cache()
        total = len(records)
        confirmed_spills = sum(1 for r in records if r["action"] == "confirm_spill")
        hard_negatives = sum(1 for r in records if r.get("is_hard_negative", False))
        
        breakdown = {}
        for r in records:
            act = r["action"]
            breakdown[act] = breakdown.get(act, 0) + 1

        avg_uncertainty = float(np.mean([r["uncertainty_score"] for r in records])) if records else 0.0

        return {
            "total_feedback_events": total,
            "confirmed_spills": confirmed_spills,
            "hard_negatives_mined": hard_negatives,
            "action_breakdown": breakdown,
            "average_uncertainty": round(avg_uncertainty, 3),
            "retraining_readiness": "ready" if total >= 5 else "collecting",
            "threshold_for_retraining": 5
        }

    @classmethod
    def get_high_priority_queue(cls, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Fetches the highest priority samples for fine-tuning.
        """
        records = cls._read_cache()
        # Sort by priority score descending
        sorted_records = sorted(records, key=lambda x: x.get("priority_score", 0.0), reverse=True)
        return sorted_records[:limit]
