"""Learned policy — WQT integration for AI-assisted execution planning.

The long-term AI system learns:
    π(W, H_Q, H_C, B) → A

where:
    W = workflow
    H_Q = QPU capability
    H_C = classical/HPC resources
    B = resource/error budget
    A = execution plan

V0.1 implements a simple heuristic policy that:
1. Reads the workflow graph and resource model
2. Uses closed-loop training data (if available) to inform decisions
3. Ranks transformations by predicted saving × confidence
4. Does NOT execute transformations — only ranks them

The AI must explain:
- what is moved
- why
- evidence
- expected saving
- scientific risk

It must NEVER silently rewrite scientific experiments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .transformation_registry import Recommendation, GuaranteeClass, Confidence
from .flow_ir import WorkflowGraph, OperationType, ResourceType
from .resource_model import LocalResourceModel, is_statevector_feasible, statevector_memory_gb


@dataclass
class PolicyDecision:
    """A single decision from the learned policy."""
    transformation_id: str
    transformation_name: str
    rank: int  # 1 = best
    predicted_saving: float
    confidence: float  # 0.0 to 1.0
    guarantee_class: str
    rationale: str
    evidence: str
    scientific_risk: str
    recommended: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "transformation_name": self.transformation_name,
            "rank": self.rank,
            "predicted_saving": self.predicted_saving,
            "confidence": self.confidence,
            "guarantee_class": self.guarantee_class,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "scientific_risk": self.scientific_risk,
            "recommended": self.recommended,
            "metadata": self.metadata,
        }


@dataclass
class PolicyPlan:
    """Complete plan from the learned policy."""
    schema_version: str = "wq-qcsc-policy-v0.1"
    decisions: list[PolicyDecision] = field(default_factory=list)
    workflow_summary: dict[str, Any] = field(default_factory=dict)
    resource_context: dict[str, Any] = field(default_factory=dict)
    total_predicted_saving: float = 1.0
    plan_type: str = "balanced"  # safe, balanced, aggressive
    explanation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "decisions": [d.to_dict() for d in self.decisions],
            "workflow_summary": self.workflow_summary,
            "resource_context": self.resource_context,
            "total_predicted_saving": self.total_predicted_saving,
            "plan_type": self.plan_type,
            "explanation": self.explanation,
        }


class LearnedPolicy:
    """Heuristic policy for ranking QPU-saving transformations.

    V0.1 uses a weighted scoring function:
        score = predicted_saving × confidence_weight × guarantee_weight

    Future versions will use a trained WQT model.
    """

    # Weights by guarantee class
    GUARANTEE_WEIGHTS: dict[GuaranteeClass, float] = {
        GuaranteeClass.EXACT: 1.0,
        GuaranteeClass.EQUIVALENT: 0.8,
        GuaranteeClass.APPROXIMATE: 0.5,
        GuaranteeClass.EXPERIMENTAL: 0.2,
    }

    # Weights by confidence level
    CONFIDENCE_WEIGHTS: dict[Confidence, float] = {
        Confidence.HIGH: 1.0,
        Confidence.MEDIUM: 0.7,
        Confidence.LOW: 0.3,
    }

    def __init__(self, training_data_path: str | Path | None = None) -> None:
        """Initialize the policy.

        Args:
            training_data_path: Optional path to closed-loop training data (JSONL).
                If provided, the policy uses historical accuracy to adjust weights.
        """
        self.training_data: list[dict[str, Any]] = []
        self.historical_accuracy: dict[str, float] = {}  # transformation_id → accuracy

        if training_data_path:
            self._load_training_data(training_data_path)

    def _load_training_data(self, path: str | Path) -> None:
        """Load closed-loop training data to calibrate the policy."""
        p = Path(path)
        if not p.exists():
            return

        # Group by transformation and compute accuracy
        by_transform: dict[str, list[bool]] = {}
        for line in p.read_text(encoding="utf-8").strip().split("\n"):
            if not line:
                continue
            try:
                record = json.loads(line)
                tid = record.get("input", {}).get("transformation_id", "unknown")
                success = record.get("output", {}).get("success", False)
                if tid not in by_transform:
                    by_transform[tid] = []
                by_transform[tid].append(success)
                self.training_data.append(record)
            except json.JSONDecodeError:
                continue

        # Compute historical accuracy per transformation
        for tid, successes in by_transform.items():
            if successes:
                self.historical_accuracy[tid] = sum(successes) / len(successes)

    def rank_transformations(
        self,
        recommendations: list[Recommendation],
        graph: WorkflowGraph,
        summary: dict[str, Any],
        local_resources: LocalResourceModel | None = None,
        plan_type: str = "balanced",
    ) -> PolicyPlan:
        """Rank transformations by predicted value.

        Args:
            recommendations: List of detected QPU-saving opportunities
            graph: Workflow graph from analysis
            summary: Analysis summary
            local_resources: Local compute resources (if detected)
            plan_type: "safe", "balanced", or "aggressive"

        Returns:
            PolicyPlan with ranked decisions
        """
        plan = PolicyPlan(plan_type=plan_type)

        # Build workflow summary for context
        plan.workflow_summary = {
            "frameworks": summary.get("frameworks_detected", []),
            "n_quantum_boundaries": len(summary.get("quantum_boundaries", [])),
            "n_circuits": len(summary.get("circuits", [])),
            "n_loops": len(summary.get("loops", [])),
            "n_quantum_nodes": len(graph.quantum_nodes()),
            "n_classical_nodes": len(graph.classical_nodes()),
        }

        # Build resource context
        if local_resources:
            plan.resource_context = local_resources.to_dict()
        else:
            plan.resource_context = {"detected": False}

        # Score each recommendation
        scored: list[tuple[float, Recommendation]] = []
        for rec in recommendations:
            score = self._score_recommendation(rec, plan_type)
            scored.append((score, rec))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build decisions
        total_saving = 1.0
        for rank, (score, rec) in enumerate(scored, 1):
            # Apply composition: don't simply multiply
            saving = rec.estimated_saving_factor or 1.0
            if saving == float("inf"):
                saving = 100.0  # Cap for scoring

            # Conservative composition: largest factor + 10% per additional
            if rank == 1:
                total_saving = saving
            else:
                total_saving += saving * 0.1

            # Determine if recommended based on plan type
            recommended = self._is_recommended(rec, plan_type)

            decision = PolicyDecision(
                transformation_id=rec.transformation_id,
                transformation_name=rec.transformation_name,
                rank=rank,
                predicted_saving=saving,
                confidence=self._confidence_score(rec),
                guarantee_class=rec.guarantee_class.value,
                rationale=self._generate_rationale(rec, summary, local_resources),
                evidence=rec.evidence,
                scientific_risk=rec.expected_quality_risk,
                recommended=recommended,
                metadata={
                    "score": round(score, 4),
                    "historical_accuracy": self.historical_accuracy.get(rec.transformation_id),
                },
            )
            plan.decisions.append(decision)

        plan.total_predicted_saving = round(total_saving, 2)
        plan.explanation = self._generate_explanation(plan, plan_type)

        return plan

    def _score_recommendation(self, rec: Recommendation, plan_type: str) -> float:
        """Score a recommendation using weighted factors."""
        saving = rec.estimated_saving_factor or 1.0
        if saving == float("inf"):
            saving = 100.0

        gw = self.GUARANTEE_WEIGHTS.get(rec.guarantee_class, 0.5)
        cw = self.CONFIDENCE_WEIGHTS.get(rec.confidence, 0.5)

        # Adjust for plan type
        if plan_type == "safe":
            if rec.guarantee_class != GuaranteeClass.EXACT:
                return 0.0  # Safe plan only considers EXACT
        elif plan_type == "balanced":
            if rec.guarantee_class == GuaranteeClass.EXPERIMENTAL:
                return 0.0  # Balanced excludes EXPERIMENTAL

        # Historical accuracy adjustment
        hist = self.historical_accuracy.get(rec.transformation_id, 0.5)

        return saving * gw * cw * (0.5 + 0.5 * hist)

    def _confidence_score(self, rec: Recommendation) -> float:
        """Convert confidence level to numeric score."""
        return self.CONFIDENCE_WEIGHTS.get(rec.confidence, 0.5)

    def _is_recommended(self, rec: Recommendation, plan_type: str) -> bool:
        """Determine if a recommendation should be applied in this plan type."""
        if plan_type == "safe":
            return rec.guarantee_class == GuaranteeClass.EXACT
        elif plan_type == "balanced":
            return rec.guarantee_class in (GuaranteeClass.EXACT, GuaranteeClass.EQUIVALENT) or (
                rec.guarantee_class == GuaranteeClass.APPROXIMATE and rec.confidence in (Confidence.HIGH, Confidence.MEDIUM)
            )
        else:  # aggressive
            return True

    def _generate_rationale(
        self, rec: Recommendation, summary: dict[str, Any], local: LocalResourceModel | None
    ) -> str:
        """Generate a human-readable rationale for the decision."""
        parts = [rec.what_detected, rec.why_expensive, f"Alternative: {rec.alternative}"]

        if rec.guarantee_class == GuaranteeClass.EXACT:
            parts.append("This is mathematically exact — no quality risk.")
        elif rec.guarantee_class == GuaranteeClass.APPROXIMATE:
            parts.append(f"This is approximate — quality risk: {rec.expected_quality_risk}")
            parts.append(f"Validation required: {rec.validation_experiment}")
        elif rec.guarantee_class == GuaranteeClass.EXPERIMENTAL:
            parts.append("This is experimental — hardware validation required before deployment.")

        if local and local.detected:
            parts.append(f"Local resources: {local.cpu_cores} CPU cores, {local.ram_gb:.0f} GB RAM")
            if local.has_gpu:
                parts.append(f"GPU: {local.gpu_type} ({local.gpu_memory_gb:.0f} GB)")

        return " | ".join(parts)

    def _generate_explanation(self, plan: PolicyPlan, plan_type: str) -> str:
        """Generate an overall explanation for the plan."""
        n_recommended = sum(1 for d in plan.decisions if d.recommended)
        n_exact = sum(1 for d in plan.decisions if d.guarantee_class == "exact")

        explanation = (
            f"Plan '{plan_type}' recommends {n_recommended} of {len(plan.decisions)} transformations. "
            f"Of these, {n_exact} are EXACT (mathematically guaranteed). "
            f"Total predicted saving: {plan.total_predicted_saving}x. "
        )

        if plan_type == "safe":
            explanation += "Safe plan only includes EXACT transformations — no quality risk."
        elif plan_type == "balanced":
            explanation += "Balanced plan includes EXACT and low-risk APPROXIMATE transformations — configure tolerance ε."
        else:
            explanation += "Aggressive plan includes all transformations — validate EXPERIMENTAL ones on hardware."

        return explanation

    def save_plan(self, plan: PolicyPlan, path: str | Path) -> Path:
        """Save the policy plan to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
        return p
