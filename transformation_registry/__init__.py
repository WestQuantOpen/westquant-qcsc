"""Transformation registry for QPU-saving opportunities.

Each transformation is a versioned detector that identifies a specific
opportunity to reduce QPU workload. Transformations are classified by
guarantee class:

- EXACT: Mathematical result preserved.
- EQUIVALENT: Expected statistically equivalent result within tolerance.
- APPROXIMATE: Known quality/resource trade-off.
- EXPERIMENTAL: Requires user validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class GuaranteeClass(str, Enum):
    EXACT = "exact"
    EQUIVALENT = "equivalent"
    APPROXIMATE = "approximate"
    EXPERIMENTAL = "experimental"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CompositionRelation(str, Enum):
    INDEPENDENT = "independent"
    OVERLAPPING = "overlapping"
    REQUIRES = "requires"
    CONFLICTS_WITH = "conflicts_with"
    UNKNOWN = "unknown"


@dataclass
class Transformation:
    id: str
    version: str
    name: str
    description: str
    applicability: str
    guarantee_class: GuaranteeClass
    required_evidence: str
    estimated_resource_effect: str
    quality_risk: str
    validation_method: str
    frameworks_supported: list[str]
    composition: dict[str, CompositionRelation] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "applicability": self.applicability,
            "guarantee_class": self.guarantee_class.value,
            "required_evidence": self.required_evidence,
            "estimated_resource_effect": self.estimated_resource_effect,
            "quality_risk": self.quality_risk,
            "validation_method": self.validation_method,
            "frameworks_supported": self.frameworks_supported,
            "composition": {k: v.value for k, v in self.composition.items()},
        }


@dataclass
class Recommendation:
    transformation_id: str
    transformation_name: str
    guarantee_class: GuaranteeClass
    confidence: Confidence
    what_detected: str
    why_expensive: str
    alternative: str
    estimated_qpu_saving: str
    required_local_compute: str
    expected_quality_risk: str
    validation_experiment: str
    evidence: str
    affected_nodes: list[str] = field(default_factory=list)
    estimated_saving_factor: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "transformation_id": self.transformation_id,
            "transformation_name": self.transformation_name,
            "guarantee_class": self.guarantee_class.value,
            "confidence": self.confidence.value,
            "what_detected": self.what_detected,
            "why_expensive": self.why_expensive,
            "alternative": self.alternative,
            "estimated_qpu_saving": self.estimated_qpu_saving,
            "required_local_compute": self.required_local_compute,
            "expected_quality_risk": self.expected_quality_risk,
            "validation_experiment": self.validation_experiment,
            "evidence": self.evidence,
            "affected_nodes": self.affected_nodes,
            "estimated_saving_factor": self.estimated_saving_factor,
            "metadata": self.metadata,
        }


# The 15 initial transformations (T01-T15)
TRANSFORMATIONS: list[Transformation] = [
    Transformation(
        id="T01", version="0.1", name="Redundant Measurement Bases",
        description="Detect observables obtainable from the same measurement basis.",
        applicability="Multiple measurement bases where some are redundant",
        guarantee_class=GuaranteeClass.EXACT,
        required_evidence="Observable commutation analysis with measurement basis",
        estimated_resource_effect="Reduces number of measurement bases by up to N",
        quality_risk="None — exact reconstruction from same basis",
        validation_method="Unit-test observable reconstruction from archived bitstrings",
        frameworks_supported=["qiskit", "pennylane"],
        composition={"T02": CompositionRelation.OVERLAPPING, "T03": CompositionRelation.OVERLAPPING},
    ),
    Transformation(
        id="T02", version="0.1", name="Measurement Reuse",
        description="Identify observables recoverable from existing samples.",
        applicability="Multiple observables measured from same or compatible bases",
        guarantee_class=GuaranteeClass.EXACT,
        required_evidence="Observable compatibility with stored bitstrings",
        estimated_resource_effect="Additional QPU cost: 0 for recovered observables",
        quality_risk="None — exact reconstruction from existing samples",
        validation_method="Verify observable reconstruction from archived bitstrings",
        frameworks_supported=["qiskit", "pennylane"],
        composition={"T01": CompositionRelation.OVERLAPPING, "T03": CompositionRelation.OVERLAPPING},
    ),
    Transformation(
        id="T03", version="0.1", name="Commuting Measurement Grouping",
        description="Group compatible Pauli observables for simultaneous measurement.",
        applicability="Multiple Pauli observables in estimation workflow",
        guarantee_class=GuaranteeClass.EXACT,
        required_evidence="Pauli commutation graph analysis",
        estimated_resource_effect="Reduces measurement settings by grouping factor",
        quality_risk="None — commuting observables measured simultaneously",
        validation_method="Verify commutation relations and reconstruction",
        frameworks_supported=["qiskit", "pennylane"],
        composition={"T01": CompositionRelation.OVERLAPPING, "T02": CompositionRelation.OVERLAPPING},
    ),
    Transformation(
        id="T04", version="0.1", name="Shot Reduction Opportunity",
        description="Detect uniform/high shot budgets and propose an ablation experiment.",
        applicability="Workloads with fixed high shot counts across all circuits",
        guarantee_class=GuaranteeClass.APPROXIMATE,
        required_evidence="Variance analysis of existing measurement results",
        estimated_resource_effect="Reduces shots by 2-10x depending on variance",
        quality_risk="Increased estimator variance — requires tolerance configuration",
        validation_method="Subsample existing bitstrings at reduced shot counts, compare metrics",
        frameworks_supported=["qiskit", "pennylane", "pytket", "pulser"],
    ),
    Transformation(
        id="T05", version="0.1", name="Adaptive Shot Allocation",
        description="Suggest variable shot budgets based on variance/optimization stage.",
        applicability="Variational optimization loops with uniform shot budgets",
        guarantee_class=GuaranteeClass.APPROXIMATE,
        required_evidence="Per-iteration variance data",
        estimated_resource_effect="Reduces total shots by 2-5x for variational workflows",
        quality_risk="Some iterations may have higher variance",
        validation_method="Compare fixed vs adaptive shot allocation on existing traces",
        frameworks_supported=["qiskit", "pennylane"],
    ),
    Transformation(
        id="T06", version="0.1", name="Simulator Pretraining",
        description="Detect repeated variational parameter optimization and recommend simulator pretraining.",
        applicability="Variational loops with many parameters and moderate circuit width",
        guarantee_class=GuaranteeClass.APPROXIMATE,
        required_evidence="Circuit width within simulator memory limit",
        estimated_resource_effect="Reduces QPU calls by 5-50x for parameter pretraining",
        quality_risk="Simulator-optimized parameters may need fine-tuning on QPU",
        validation_method="Pretrain on simulator, validate on small QPU sample",
        frameworks_supported=["qiskit", "pennylane", "pytket"],
    ),
    Transformation(
        id="T07", version="0.1", name="Parameter Transfer",
        description="Detect structurally related problem instances sharing circuit architecture.",
        applicability="Multiple problem instances with same circuit family",
        guarantee_class=GuaranteeClass.APPROXIMATE,
        required_evidence="Structural similarity between problem instances",
        estimated_resource_effect="Reduces QPU optimization by 3-15x via parameter transfer",
        quality_risk="Transferred parameters may not be optimal for new instance",
        validation_method="Transfer parameters, measure improvement vs cold start",
        frameworks_supported=["qiskit", "pennylane"],
    ),
    Transformation(
        id="T08", version="0.1", name="Repeated-Circuit Caching",
        description="Detect identical circuits/parameters being executed repeatedly.",
        applicability="Loops that resubmit identical circuits",
        guarantee_class=GuaranteeClass.EXACT,
        required_evidence="Circuit and parameter equality check",
        estimated_resource_effect="Eliminates redundant QPU submissions entirely",
        quality_risk="None — cached results are identical",
        validation_method="Hash circuit+parameters, verify cache hits",
        frameworks_supported=["qiskit", "pennylane", "pytket", "pulser"],
    ),
    Transformation(
        id="T09", version="0.1", name="Graph/Problem Preprocessing",
        description="Detect graph-optimization workloads and suggest classical preprocessing.",
        applicability="Graph-based optimization problems (MaxCut, QUBO, Ising)",
        guarantee_class=GuaranteeClass.EXACT,
        required_evidence="Graph structure analysis (components, symmetry, sparsity)",
        estimated_resource_effect="Reduces problem size by removing trivial components",
        quality_risk="None — preprocessing preserves exact problem structure",
        validation_method="Verify reduced graph produces same optimal solution",
        frameworks_supported=["qiskit", "pennylane"],
    ),
    Transformation(
        id="T10", version="0.1", name="Workload Packing",
        description="Detect independent circuits whose total width fits the target QPU.",
        applicability="Multiple independent circuits below QPU width",
        guarantee_class=GuaranteeClass.EXPERIMENTAL,
        required_evidence="Circuit independence + total width <= QPU width",
        estimated_resource_effect="Throughput improvement up to k=floor(nQPU/ncircuit)",
        quality_risk="Potential crosstalk — requires hardware validation",
        validation_method="Pack circuits, compare results vs separate execution",
        frameworks_supported=["qiskit"],
    ),
    Transformation(
        id="T11", version="0.1", name="Candidate Pruning",
        description="Detect large parameter sweeps and suggest surrogate filtering.",
        applicability="Large parameter sweep loops with many candidates",
        guarantee_class=GuaranteeClass.APPROXIMATE,
        required_evidence="Surrogate model accuracy on validation set",
        estimated_resource_effect="Reduces QPU evaluations by 5-20x via pre-filtering",
        quality_risk="Surrogate may filter good candidates",
        validation_method="Train surrogate on subset, evaluate pruning accuracy",
        frameworks_supported=["qiskit", "pennylane"],
    ),
    Transformation(
        id="T12", version="0.1", name="Circuit Cutting Candidate",
        description="Identify circuits whose interaction graph may permit cutting.",
        applicability="Circuits with sparse qubit interaction graphs",
        guarantee_class=GuaranteeClass.EXACT,
        required_evidence="Interaction graph connectivity analysis",
        estimated_resource_effect="Reduces effective circuit width at cost of more shots",
        quality_risk="Increased shot overhead for reconstruction",
        validation_method="Verify reconstruction fidelity after cutting",
        frameworks_supported=["qiskit", "pytket"],
    ),
    Transformation(
        id="T13", version="0.1", name="CPU/GPU Simulator Routing",
        description="Recommend best available simulator class for each workload.",
        applicability="Any quantum workload within simulator feasibility",
        guarantee_class=GuaranteeClass.EXACT,
        required_evidence="Circuit width, depth, and structure analysis",
        estimated_resource_effect="Eliminates QPU cost entirely for simulatable workloads",
        quality_risk="None — exact simulation preserves all results",
        validation_method="Compare simulator vs QPU results on small instances",
        frameworks_supported=["qiskit", "pennylane", "pytket", "pulser"],
    ),
    Transformation(
        id="T14", version="0.1", name="Circuit Architecture Reduction",
        description="Identify removable gates where mathematically justified.",
        applicability="Circuits with redundant or simplifiable gate sequences",
        guarantee_class=GuaranteeClass.EXACT,
        required_evidence="Gate-level analysis or compiler optimization passes",
        estimated_resource_effect="Reduces gate count and circuit depth",
        quality_risk="None — mathematically equivalent circuit",
        validation_method="Verify unitary equivalence after reduction",
        frameworks_supported=["qiskit", "pytket"],
    ),
    Transformation(
        id="T15", version="0.1", name="Quantum/Classical Decomposition",
        description="Detect workloads where only a sampling component must be quantum.",
        applicability="Workloads with classical post-processing on quantum samples",
        guarantee_class=GuaranteeClass.EQUIVALENT,
        required_evidence="Decomposition of quantum vs classical components",
        estimated_resource_effect="Reduces QPU workload to sampling only",
        quality_risk="Statistical equivalence within tolerance",
        validation_method="Compare decomposed vs monolithic execution",
        frameworks_supported=["qiskit", "pennylane"],
    ),
]


def get_transformation(tid: str) -> Transformation | None:
    for t in TRANSFORMATIONS:
        if t.id == tid:
            return t
    return None


def get_composition_relation(tid_a: str, tid_b: str) -> CompositionRelation:
    t = get_transformation(tid_a)
    if t and tid_b in t.composition:
        return t.composition[tid_b]
    return CompositionRelation.UNKNOWN
