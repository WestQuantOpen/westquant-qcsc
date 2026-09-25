"""Closed-loop measurements — compare predicted savings with actual results.

Stores: prediction → execution → observed savings.

This creates a feedback loop that:
1. Records what WestQuant predicted
2. Records what actually happened
3. Computes prediction error
4. Builds a dataset for future learned policy (Phase 9)

Usage:
    from qcsc.closed_loop import ClosedLoopTracker

    tracker = ClosedLoopTracker()
    tracker.record_prediction(rec_id, predicted_saving, context)
    tracker.record_observation(rec_id, actual_saving, actual_quality)
    tracker.compute_error(rec_id)
    report = tracker.generate_report()
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class PredictionRecord:
    """What WestQuant predicted for a transformation."""
    record_id: str
    transformation_id: str
    transformation_name: str
    guarantee_class: str
    predicted_saving: float | None  # e.g., 4.0 means 4x reduction
    predicted_quality_effect: str  # "none", "increased_variance", etc.
    workflow_state: dict[str, Any] = field(default_factory=dict)
    problem_family: str = ""
    hardware: str = ""
    simulator: str = ""
    budget: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "transformation_id": self.transformation_id,
            "transformation_name": self.transformation_name,
            "guarantee_class": self.guarantee_class,
            "predicted_saving": self.predicted_saving,
            "predicted_quality_effect": self.predicted_quality_effect,
            "workflow_state": self.workflow_state,
            "problem_family": self.problem_family,
            "hardware": self.hardware,
            "simulator": self.simulator,
            "budget": self.budget,
            "timestamp": self.timestamp,
        }


@dataclass
class ObservationRecord:
    """What actually happened when the transformation was applied."""
    record_id: str
    actual_saving: float | None  # actual reduction factor
    actual_quality_effect: str  # observed quality change
    success: bool
    failure_type: str = ""  # "none", "quality_degradation", "no_saving", "error", etc.
    notes: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "actual_saving": self.actual_saving,
            "actual_quality_effect": self.actual_quality_effect,
            "success": self.success,
            "failure_type": self.failure_type,
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


@dataclass
class ClosedLoopRecord:
    """Combined prediction + observation + error."""
    record_id: str
    prediction: PredictionRecord
    observation: ObservationRecord | None = None
    prediction_error: float | None = None  # |predicted - actual|
    quality_prediction_correct: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "prediction": self.prediction.to_dict(),
            "observation": self.observation.to_dict() if self.observation else None,
            "prediction_error": self.prediction_error,
            "quality_prediction_correct": self.quality_prediction_correct,
        }


class ClosedLoopTracker:
    """Tracks prediction → execution → observed savings for closed-loop learning."""

    def __init__(self) -> None:
        self.records: dict[str, ClosedLoopRecord] = {}

    def record_prediction(
        self,
        record_id: str,
        transformation_id: str,
        transformation_name: str,
        guarantee_class: str,
        predicted_saving: float | None,
        predicted_quality_effect: str = "none",
        workflow_state: dict[str, Any] | None = None,
        problem_family: str = "",
        hardware: str = "",
        simulator: str = "",
        budget: str = "",
    ) -> None:
        """Record a prediction made by WestQuant."""
        self.records[record_id] = ClosedLoopRecord(
            record_id=record_id,
            prediction=PredictionRecord(
                record_id=record_id,
                transformation_id=transformation_id,
                transformation_name=transformation_name,
                guarantee_class=guarantee_class,
                predicted_saving=predicted_saving,
                predicted_quality_effect=predicted_quality_effect,
                workflow_state=workflow_state or {},
                problem_family=problem_family,
                hardware=hardware,
                simulator=simulator,
                budget=budget,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
        )

    def record_observation(
        self,
        record_id: str,
        actual_saving: float | None,
        actual_quality_effect: str = "none",
        success: bool = True,
        failure_type: str = "none",
        notes: str = "",
    ) -> None:
        """Record what actually happened when the transformation was applied."""
        if record_id not in self.records:
            self.records[record_id] = ClosedLoopRecord(
                record_id=record_id,
                prediction=PredictionRecord(
                    record_id=record_id,
                    transformation_id="unknown",
                    transformation_name="unknown",
                    guarantee_class="unknown",
                    predicted_saving=None,
                ),
            )

        self.records[record_id].observation = ObservationRecord(
            record_id=record_id,
            actual_saving=actual_saving,
            actual_quality_effect=actual_quality_effect,
            success=success,
            failure_type=failure_type,
            notes=notes,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Compute error
        self._compute_error(record_id)

    def _compute_error(self, record_id: str) -> None:
        """Compute prediction error for a record."""
        rec = self.records[record_id]
        if rec.observation is None:
            return

        if rec.prediction.predicted_saving is not None and rec.observation.actual_saving is not None:
            rec.prediction_error = abs(rec.prediction.predicted_saving - rec.observation.actual_saving)

        # Check if quality prediction was correct
        if rec.prediction.predicted_quality_effect == rec.observation.actual_quality_effect:
            rec.quality_prediction_correct = True
        elif rec.observation.actual_quality_effect == "none" and rec.prediction.predicted_quality_effect == "none":
            rec.quality_prediction_correct = True
        else:
            rec.quality_prediction_correct = False

    def generate_report(self) -> dict[str, Any]:
        """Generate a closed-loop measurement report."""
        total = len(self.records)
        observed = sum(1 for r in self.records.values() if r.observation is not None)
        successful = sum(1 for r in self.records.values() if r.observation and r.observation.success)
        failed = observed - successful

        # Compute mean prediction error
        errors = [r.prediction_error for r in self.records.values() if r.prediction_error is not None]
        mean_error = sum(errors) / len(errors) if errors else None

        # Compute quality prediction accuracy
        quality_checks = [r.quality_prediction_correct for r in self.records.values() if r.quality_prediction_correct is not None]
        quality_accuracy = sum(1 for q in quality_checks if q) / len(quality_checks) if quality_checks else None

        # Group by transformation
        by_transformation: dict[str, dict[str, Any]] = {}
        for rec in self.records.values():
            tid = rec.prediction.transformation_id
            if tid not in by_transformation:
                by_transformation[tid] = {
                    "n_predictions": 0,
                    "n_observed": 0,
                    "n_successful": 0,
                    "n_failed": 0,
                    "mean_prediction_error": [],
                }
            by_transformation[tid]["n_predictions"] += 1
            if rec.observation:
                by_transformation[tid]["n_observed"] += 1
                if rec.observation.success:
                    by_transformation[tid]["n_successful"] += 1
                else:
                    by_transformation[tid]["n_failed"] += 1
                if rec.prediction_error is not None:
                    by_transformation[tid]["mean_prediction_error"].append(rec.prediction_error)

        for tid, stats in by_transformation.items():
            errs = stats["mean_prediction_error"]
            stats["mean_prediction_error"] = sum(errs) / len(errs) if errs else None

        return {
            "total_predictions": total,
            "total_observations": observed,
            "successful": successful,
            "failed": failed,
            "mean_prediction_error": mean_error,
            "quality_prediction_accuracy": quality_accuracy,
            "by_transformation": by_transformation,
        }

    def save(self, path: str | Path) -> Path:
        """Save all records to a JSONL file for training data."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for rec in self.records.values():
                f.write(json.dumps(rec.to_dict()) + "\n")
        return p

    def load(self, path: str | Path) -> None:
        """Load records from a JSONL file."""
        p = Path(path)
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            data = json.loads(line)
            pred_data = data["prediction"]
            rec = ClosedLoopRecord(
                record_id=data["record_id"],
                prediction=PredictionRecord(**{k: v for k, v in pred_data.items()}),
                prediction_error=data.get("prediction_error"),
                quality_prediction_correct=data.get("quality_prediction_correct"),
            )
            if data.get("observation"):
                obs_data = data["observation"]
                rec.observation = ObservationRecord(**{k: v for k, v in obs_data.items()})
            self.records[rec.record_id] = rec

    def export_training_data(self, path: str | Path) -> Path:
        """Export records as WQT training data format.

        Each record becomes a training example:
        (workflow_state, transformation) → (predicted_saving, actual_saving, success)
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for rec in self.records.values():
                if rec.observation is None:
                    continue  # Skip unobserved records
                example = {
                    "schema_version": "wq-qcsc-training-v0.1",
                    "record_id": rec.record_id,
                    "input": {
                        "workflow_state": rec.prediction.workflow_state,
                        "transformation_id": rec.prediction.transformation_id,
                        "guarantee_class": rec.prediction.guarantee_class,
                        "problem_family": rec.prediction.problem_family,
                        "hardware": rec.prediction.hardware,
                    },
                    "output": {
                        "predicted_saving": rec.prediction.predicted_saving,
                        "actual_saving": rec.observation.actual_saving,
                        "prediction_error": rec.prediction_error,
                        "success": rec.observation.success,
                        "failure_type": rec.observation.failure_type,
                    },
                    "label": {
                        "quality_prediction_correct": rec.quality_prediction_correct,
                        "actual_quality_effect": rec.observation.actual_quality_effect,
                    },
                }
                f.write(json.dumps(example) + "\n")
        return p
