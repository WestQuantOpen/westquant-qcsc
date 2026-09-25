"""Cost model — estimates baseline QPU workload."""
from __future__ import annotations

from typing import Any

from .flow_ir import WorkflowGraph, OperationType, ResourceType, CostCertainty


def estimate_baseline_cost(summary: dict[str, Any], graph: WorkflowGraph) -> dict[str, Any]:
    """Estimate baseline QPU cost from workflow analysis."""
    quantum_calls = [b for b in summary.get("quantum_boundaries", []) if not b.get("is_simulator", False)]
    circuits = summary.get("circuits", [])
    observables = summary.get("observables", [])
    loops = summary.get("loops", [])
    shots_list = summary.get("shots_detected", [])

    # Estimate number of circuits
    n_circuits = len(circuits)
    if n_circuits == 0:
        n_circuits = len(quantum_calls)

    # Estimate loop expansion
    loop_multiplier = 1
    for loop in loops:
        count = loop.get("count", 1)
        if isinstance(count, int) and count > 1:
            loop_multiplier *= count

    # Estimate shots
    if shots_list:
        shots_per_circuit = max((s.get("shots", 0) for s in shots_list if isinstance(s.get("shots"), int)), default=0)
        if shots_per_circuit == 0:
            shots_per_circuit = 1024  # default assumption
    else:
        shots_per_circuit = 1024

    # Estimate measurement settings
    n_measurement_settings = max(len(observables), 1)

    # Total QPU jobs
    n_qpu_jobs = n_circuits * loop_multiplier

    # Total shots
    total_shots = n_qpu_jobs * n_measurement_settings * shots_per_circuit

    # Estimate qubits
    max_qubits = max((c.get("n_qubits", 0) or 0 for c in circuits), default=0)

    return {
        "n_circuits": n_circuits,
        "n_qpu_jobs": n_qpu_jobs,
        "n_measurement_settings": n_measurement_settings,
        "shots_per_circuit": shots_per_circuit,
        "total_shots": total_shots,
        "max_qubits": max_qubits,
        "loop_multiplier": loop_multiplier,
        "certainty": CostCertainty.ESTIMATED.value,
        "note": "Costs are estimated from static analysis. Runtime values may differ.",
    }
