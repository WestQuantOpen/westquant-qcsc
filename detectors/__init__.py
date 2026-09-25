"""QPU-saving transformation detectors.

Each detector analyzes the workflow graph and analyzer summary to produce
Recommendations for specific QPU-saving opportunities.
"""
from __future__ import annotations

from typing import Any

from ..flow_ir import WorkflowGraph, OperationType, ResourceType
from ..transformation_registry import (
    Recommendation, GuaranteeClass, Confidence, TRANSFORMATIONS,
    get_transformation,
)
from ..resource_model import (
    is_statevector_feasible, is_stabilizer_feasible, is_mps_feasible,
    statevector_memory_gb, LocalResourceModel,
)


def detect_t01_redundant_bases(summary: dict[str, Any], graph: WorkflowGraph) -> list[Recommendation]:
    """T01 — Redundant Measurement Bases."""
    observables = summary.get("observables", [])
    if len(observables) < 2:
        return []

    # Check if all observables are Z-basis (diagonal)
    z_basis_count = sum(1 for o in observables if "Z" in str(o.get("type", "")))
    if z_basis_count >= 2:
        t = get_transformation("T01")
        assert t is not None
        return [Recommendation(
            transformation_id="T01",
            transformation_name=t.name,
            guarantee_class=t.guarantee_class,
            confidence=Confidence.HIGH,
            what_detected=f"{len(observables)} observables detected, all diagonal in Z basis",
            why_expensive="Multiple measurement bases require separate QPU executions",
            alternative="All observables can be reconstructed from Z-basis samples alone",
            estimated_qpu_saving="2.0x reduction in measurement bases",
            required_local_compute="Classical post-processing for observable reconstruction",
            expected_quality_risk="None — exact reconstruction from same basis",
            validation_experiment="Unit-test observable reconstruction from archived bitstrings",
            evidence="All observables are diagonal in Z basis — no additional bases needed",
            estimated_saving_factor=2.0,
        )]
    return []


def detect_t02_measurement_reuse(summary: dict[str, Any], graph: WorkflowGraph) -> list[Recommendation]:
    """T02 — Measurement Reuse."""
    shots = summary.get("shots_detected", [])
    if len(shots) < 2:
        return []

    # Check if same shots are used across multiple calls
    shot_values = [s.get("shots") for s in shots if s.get("shots")]
    if len(set(shot_values)) == 1 and len(shot_values) >= 2:
        t = get_transformation("T02")
        assert t is not None
        return [Recommendation(
            transformation_id="T02",
            transformation_name=t.name,
            guarantee_class=t.guarantee_class,
            confidence=Confidence.MEDIUM,
            what_detected=f"{len(shot_values)} measurement calls with identical shot budget ({shot_values[0]})",
            why_expensive="Repeated measurements with same configuration may be reusable",
            alternative="Cache and reuse measurement results for identical circuit+parameter combinations",
            estimated_qpu_saving="Eliminates redundant QPU submissions entirely for cached results",
            required_local_compute="Storage for cached bitstrings",
            expected_quality_risk="None — cached results are identical",
            validation_experiment="Hash circuit+parameters, verify cache hits produce same results",
            evidence=f"Multiple measurement calls with identical shots={shot_values[0]}",
            estimated_saving_factor=None,
        )]
    return []


def detect_t03_commuting_groups(summary: dict[str, Any], graph: WorkflowGraph) -> list[Recommendation]:
    """T03 — Commuting Measurement Grouping."""
    observables = summary.get("observables", [])
    if len(observables) < 2:
        return []

    t = get_transformation("T03")
    assert t is not None
    return [Recommendation(
        transformation_id="T03",
        transformation_name=t.name,
        guarantee_class=t.guarantee_class,
        confidence=Confidence.MEDIUM,
        what_detected=f"{len(observables)} Pauli observables detected",
        why_expensive="Each observable may require a separate measurement basis",
        alternative="Group commuting observables for simultaneous measurement",
        estimated_qpu_saving=f"Reduces measurement settings by up to {len(observables)}x",
        required_local_compute="Classical commutation analysis + post-processing",
        expected_quality_risk="None — commuting observables measured simultaneously",
        validation_experiment="Verify commutation relations and reconstruction",
        evidence="Multiple Pauli observables — grouping depends on commutation structure",
        estimated_saving_factor=float(len(observables)),
    )]


def detect_t04_shot_reduction(summary: dict[str, Any], graph: WorkflowGraph) -> list[Recommendation]:
    """T04 — Shot Reduction Opportunity."""
    shots = summary.get("shots_detected", [])
    if not shots:
        return []

    shot_values = [s["shots"] for s in shots if isinstance(s.get("shots"), int)]
    if not shot_values:
        return []

    max_shots = max(shot_values)
    if max_shots >= 2000:
        t = get_transformation("T04")
        assert t is not None
        return [Recommendation(
            transformation_id="T04",
            transformation_name=t.name,
            guarantee_class=t.guarantee_class,
            confidence=Confidence.MEDIUM,
            what_detected=f"Uniform high shot budget: {max_shots} shots per circuit",
            why_expensive=f"Total shots: {max_shots * len(shots)} across {len(shots)} measurement calls",
            alternative="Subsample existing bitstrings at reduced shot counts, compare metrics",
            estimated_qpu_saving=f"2-10x reduction (e.g., {max_shots} -> {max_shots//4} shots)",
            required_local_compute="Classical subsampling and variance analysis",
            expected_quality_risk="Increased estimator variance — requires tolerance configuration",
            validation_experiment=f"Subsample at {max_shots}, {max_shots//2}, {max_shots//4} shots, compare task metric within 95% CI",
            evidence=f"Shot budget of {max_shots} may be higher than necessary for convergence",
            estimated_saving_factor=4.0,
        )]
    return []


def detect_t06_simulator_pretraining(summary: dict[str, Any], graph: WorkflowGraph) -> list[Recommendation]:
    """T06 — Simulator Pretraining."""
    circuits = summary.get("circuits", [])
    loops = summary.get("loops", [])

    has_variational = any(
        "optimize" in str(l.get("location", "")).lower() or
        "param" in str(l.get("location", "")).lower()
        for l in loops
    )

    max_qubits = max((c.get("n_qubits", 0) or 0 for c in circuits), default=0)

    if has_variational and max_qubits > 0 and max_qubits <= 25:
        t = get_transformation("T06")
        assert t is not None
        mem_gb = statevector_memory_gb(max_qubits)
        return [Recommendation(
            transformation_id="T06",
            transformation_name=t.name,
            guarantee_class=t.guarantee_class,
            confidence=Confidence.MEDIUM,
            what_detected=f"Variational optimization loop with {max_qubits}-qubit circuits",
            why_expensive="Each optimization iteration requires QPU execution for gradient evaluation",
            alternative=f"Pretrain parameters on statevector simulator ({mem_gb:.1f} GB RAM needed)",
            estimated_qpu_saving="5-50x reduction in QPU calls during parameter optimization",
            required_local_compute=f"Statevector simulator with {mem_gb:.1f} GB RAM",
            expected_quality_risk="Simulator-optimized parameters may need fine-tuning on QPU",
            validation_experiment="Pretrain on simulator, validate on small QPU sample",
            evidence=f"Circuit width {max_qubits} qubits is within statevector simulator feasibility",
            estimated_saving_factor=10.0,
        )]
    return []


def detect_t08_caching(summary: dict[str, Any], graph: WorkflowGraph) -> list[Recommendation]:
    """T08 — Repeated-Circuit Caching."""
    shots = summary.get("shots_detected", [])
    loops = summary.get("loops", [])

    # Check for loops that might resubmit identical circuits
    has_repeated_loops = any(l.get("count", 1) > 1 for l in loops)
    if has_repeated_loops and len(shots) > 0:
        t = get_transformation("T08")
        assert t is not None
        return [Recommendation(
            transformation_id="T08",
            transformation_name=t.name,
            guarantee_class=t.guarantee_class,
            confidence=Confidence.LOW,
            what_detected="Loops detected that may resubmit identical circuits",
            why_expensive="Identical circuits submitted multiple times to QPU",
            alternative="Cache results for identical circuit+parameter combinations",
            estimated_qpu_saving="Eliminates redundant QPU submissions entirely",
            required_local_compute="Hash storage for circuit+parameter keys",
            expected_quality_risk="None — cached results are identical",
            validation_experiment="Hash circuit+parameters, verify cache hits",
            evidence="Loop structures detected alongside QPU calls — caching may eliminate redundant submissions",
        )]
    return []


def detect_t09_graph_preprocessing(summary: dict[str, Any], graph: WorkflowGraph) -> list[Recommendation]:
    """T09 — Graph/Problem Preprocessing."""
    frameworks = summary.get("frameworks_detected", [])
    if "networkx" not in frameworks:
        return []

    t = get_transformation("T09")
    assert t is not None
    return [Recommendation(
        transformation_id="T09",
        transformation_name=t.name,
        guarantee_class=t.guarantee_class,
        confidence=Confidence.MEDIUM,
        what_detected="NetworkX graph operations detected alongside quantum workload",
        why_expensive="Graph optimization problems may contain trivially reducible components",
        alternative="Apply classical graph preprocessing: component reduction, symmetry, sparsification, isolated-node removal",
        estimated_qpu_saving="Reduces problem size — savings depend on graph structure",
        required_local_compute="Classical graph analysis (NetworkX)",
        expected_quality_risk="None — preprocessing preserves exact problem structure",
        validation_experiment="Verify reduced graph produces same optimal solution",
        evidence="NetworkX detected — graph preprocessing can reduce QPU workload",
    )]


def detect_t13_simulator_routing(summary: dict[str, Any], graph: WorkflowGraph) -> list[Recommendation]:
    """T13 — CPU/GPU Simulator Routing."""
    circuits = summary.get("circuits", [])
    quantum_calls = [b for b in summary.get("quantum_boundaries", []) if not b.get("is_simulator", False)]

    if not quantum_calls or not circuits:
        return []

    max_qubits = max((c.get("n_qubits", 0) or 0 for c in circuits), default=0)
    if max_qubits == 0:
        return []

    recommendations: list[Recommendation] = []
    t = get_transformation("T13")
    assert t is not None

    if is_statevector_feasible(max_qubits):
        mem = statevector_memory_gb(max_qubits)
        recommendations.append(Recommendation(
            transformation_id="T13",
            transformation_name=t.name,
            guarantee_class=t.guarantee_class,
            confidence=Confidence.HIGH,
            what_detected=f"QPU calls detected for {max_qubits}-qubit circuits (within simulator feasibility)",
            why_expensive="QPU execution for circuits that can be exactly simulated",
            alternative=f"Statevector simulator ({mem:.1f} GB RAM) — exact results, no QPU needed",
            estimated_qpu_saving="Eliminates QPU cost entirely for simulatable workloads",
            required_local_compute=f"{mem:.1f} GB RAM for statevector",
            expected_quality_risk="None — exact simulation preserves all results",
            validation_experiment="Compare simulator vs QPU results on small instances",
            evidence=f"Circuit width {max_qubits} qubits requires {mem:.1f} GB — within typical workstation capacity",
            estimated_saving_factor=float("inf"),
        ))
    elif max_qubits <= 50:
        recommendations.append(Recommendation(
            transformation_id="T13",
            transformation_name=t.name,
            guarantee_class=t.guarantee_class,
            confidence=Confidence.MEDIUM,
            what_detected=f"QPU calls for {max_qubits}-qubit circuits (beyond statevector, MPS may work)",
            why_expensive="QPU execution for circuits that may be simulatable with tensor networks",
            alternative="MPS/tensor network simulator — exact for low-entanglement circuits",
            estimated_qpu_saving="Eliminates QPU cost if MPS is feasible",
            required_local_compute="MPS simulator with moderate RAM",
            expected_quality_risk="None for low-entanglement circuits; may fail for high entanglement",
            validation_experiment="Test MPS simulation on representative circuits",
            evidence=f"Circuit width {max_qubits} qubits — MPS feasible for shallow/low-entanglement circuits",
        ))

    return recommendations


def detect_t10_workload_packing(summary: dict[str, Any], graph: WorkflowGraph) -> list[Recommendation]:
    """T10 — Workload Packing."""
    circuits = summary.get("circuits", [])
    if len(circuits) < 2:
        return []

    max_qubits = max((c.get("n_qubits", 0) or 0 for c in circuits), default=0)
    if max_qubits == 0 or max_qubits > 20:
        return []

    t = get_transformation("T10")
    assert t is not None
    return [Recommendation(
        transformation_id="T10",
        transformation_name=t.name,
        guarantee_class=t.guarantee_class,
        confidence=Confidence.LOW,
        what_detected=f"{len(circuits)} independent circuits with max {max_qubits} qubits each",
        why_expensive="Each circuit submitted separately to QPU",
        alternative=f"Pack multiple circuits into single QPU submission if total width fits",
        estimated_qpu_saving=f"Throughput improvement up to k=floor(nQPU/{max_qubits})",
        required_local_compute="Circuit independence verification",
        expected_quality_risk="Potential crosstalk — requires hardware validation",
        validation_experiment="Pack circuits, compare results vs separate execution",
        evidence=f"Independent circuits with {max_qubits} qubits may be packable on larger QPUs",
    )]


def run_all_detectors(summary: dict[str, Any], graph: WorkflowGraph,
                      local: LocalResourceModel | None = None) -> list[Recommendation]:
    """Run all detectors and return combined recommendations."""
    all_recs: list[Recommendation] = []

    detectors = [
        detect_t01_redundant_bases,
        detect_t02_measurement_reuse,
        detect_t03_commuting_groups,
        detect_t04_shot_reduction,
        detect_t06_simulator_pretraining,
        detect_t08_caching,
        detect_t09_graph_preprocessing,
        detect_t10_workload_packing,
        detect_t13_simulator_routing,
    ]

    for detector in detectors:
        try:
            recs = detector(summary, graph)
            all_recs.extend(recs)
        except Exception:
            pass

    return all_recs
