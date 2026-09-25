"""Pareto planner — combines compatible transformations into plans.

SAFE: Only EXACT transformations. Goal: ΔM = 0 to the extent provable.
BALANCED: Allow EQUIVALENT and low-risk APPROXIMATE transformations.
AGGRESSIVE: Include experimental strategies. Always show uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..transformation_registry import (
    Recommendation, GuaranteeClass, Confidence,
    get_composition_relation, CompositionRelation,
)


@dataclass
class Plan:
    name: str
    description: str
    recommendations: list[Recommendation]
    projected_reduction: str
    guarantee_classes: list[str]
    assumptions: list[str] = field(default_factory=list)
    uncertainty: str = "low"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "recommendations": [r.to_dict() for r in self.recommendations],
            "projected_reduction": self.projected_reduction,
            "guarantee_classes": self.guarantee_classes,
            "assumptions": self.assumptions,
            "uncertainty": self.uncertainty,
        }


def _compose_savings(recommendations: list[Recommendation]) -> str:
    """Estimate combined savings — do NOT blindly multiply."""
    if not recommendations:
        return "1.0x (no transformations)"

    factors = [r.estimated_saving_factor for r in recommendations if r.estimated_saving_factor]

    if not factors:
        return "Variable — depends on workload structure"

    # Conservative composition: take the largest factor, add small contributions
    # This avoids the F(A∘B) ≠ F(A)·F(B) problem
    max_factor = max(factors)
    additional = sum(f for f in factors if f != max_factor) * 0.1  # 10% credit for each additional

    total = max_factor + additional
    if total == float("inf"):
        return "Eliminates QPU cost entirely (simulator substitution)"

    return f"{total:.1f}x projected reduction"


def _check_compatibility(recs: list[Recommendation]) -> list[str]:
    """Check composition relations between recommendations."""
    issues: list[str] = []
    for i, a in enumerate(recs):
        for j, b in enumerate(recs):
            if i >= j:
                continue
            rel = get_composition_relation(a.transformation_id, b.transformation_id)
            if rel == CompositionRelation.CONFLICTS_WITH:
                issues.append(f"{a.transformation_id} conflicts with {b.transformation_id}")
            elif rel == CompositionRelation.OVERLAPPING:
                issues.append(f"{a.transformation_id} overlaps with {b.transformation_id} — savings may not compound")
    return issues


def build_plans(recommendations: list[Recommendation]) -> list[Plan]:
    """Build SAFE, BALANCED, and AGGRESSIVE plans from recommendations."""
    if not recommendations:
        return [Plan(
            name="SAFE", description="No QPU-saving opportunities detected",
            recommendations=[], projected_reduction="1.0x",
            guarantee_classes=[], uncertainty="none",
        )]

    # SAFE: only EXACT
    safe_recs = [r for r in recommendations if r.guarantee_class == GuaranteeClass.EXACT]
    safe_issues = _check_compatibility(safe_recs)
    safe_plan = Plan(
        name="SAFE",
        description="Only EXACT transformations. Mathematical result preserved.",
        recommendations=safe_recs,
        projected_reduction=_compose_savings(safe_recs),
        guarantee_classes=["EXACT"],
        assumptions=safe_issues + ["All transformations are mathematically exact"],
        uncertainty="low",
    )

    # BALANCED: EXACT + EQUIVALENT + low-risk APPROXIMATE
    balanced_recs = [
        r for r in recommendations
        if r.guarantee_class in (GuaranteeClass.EXACT, GuaranteeClass.EQUIVALENT)
        or (r.guarantee_class == GuaranteeClass.APPROXIMATE and r.confidence in (Confidence.HIGH, Confidence.MEDIUM))
    ]
    balanced_issues = _check_compatibility(balanced_recs)
    balanced_plan = Plan(
        name="BALANCED",
        description="EXACT + EQUIVALENT + low-risk APPROXIMATE transformations. Requires user-configured tolerance ε.",
        recommendations=balanced_recs,
        projected_reduction=_compose_savings(balanced_recs),
        guarantee_classes=["EXACT", "EQUIVALENT", "APPROXIMATE"],
        assumptions=balanced_issues + ["User must configure tolerance ε for APPROXIMATE transformations"],
        uncertainty="medium",
    )

    # AGGRESSIVE: all transformations including EXPERIMENTAL
    aggressive_recs = recommendations
    aggressive_issues = _check_compatibility(aggressive_recs)
    aggressive_plan = Plan(
        name="AGGRESSIVE",
        description="All transformations including EXPERIMENTAL. Always show uncertainty.",
        recommendations=aggressive_recs,
        projected_reduction=_compose_savings(aggressive_recs),
        guarantee_classes=["EXACT", "EQUIVALENT", "APPROXIMATE", "EXPERIMENTAL"],
        assumptions=aggressive_issues + ["Experimental transformations require hardware validation"],
        uncertainty="high",
    )

    return [safe_plan, balanced_plan, aggressive_plan]
