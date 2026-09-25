# RFC-QCSC-001: WQFlow IR

## Status: Draft v0.1

## Abstract

WQFlow IR is a directed graph representation G=(V,E) for hybrid quantum-classical workflows. Every node represents an operation with type, framework, resource classification, cost estimates, and semantic tags. Edges represent data flow and dependencies.

## Node Schema

```json
{
  "node_id": "n0001",
  "operation_type": "qpu_sample | build_circuit | simulate | ...",
  "framework": "qiskit | pennylane | pytket | pulser | classical",
  "resource_type": "classical | quantum | hybrid",
  "inputs": ["n0000"],
  "outputs": ["n0002"],
  "estimated_cost": {"shots": 2000, "n_qubits": 8},
  "resource_requirements": {"ram_gb": 16},
  "semantic_tags": ["qpu_boundary"],
  "repeat_count": 10000,
  "dependency_ids": ["n0000"],
  "source_location": "file.py:42"
}
```

## Operation Types

LOAD_GRAPH, GENERATE_GRAPH, BUILD_HAMILTONIAN, BUILD_CIRCUIT, OPTIMIZE_PARAMETERS, SIMULATE, TRANSPILE, QPU_SAMPLE, ESTIMATE_OBSERVABLE, POSTPROCESS, TRAIN_MODEL, CLASSICAL_COMPUTE, MEASUREMENT, PARAMETER_SWEEP, UNKNOWN

## Design Principles

1. Framework-agnostic — any quantum framework maps to the same IR
2. Static analysis only — no execution required to build the graph
3. Cost estimates use certainty levels: EXACT, BOUNDED, ESTIMATED, UNKNOWN
4. The IR preserves semantic information that may be lost in lower-level IRs (QIR/MLIR)

---

# RFC-QCSC-002: Resource Model

## Status: Draft v0.1

## Abstract

The resource model represents QPU backend capabilities and local compute resources. It separates known, reported, estimated, and unknown capabilities.

## QPU Backend Fields

backend_id, provider, qubits, connectivity, native_gates, parallel_gate_support, measurement_capabilities, max_shots, queue_model, pricing_model, known_latency

## Simulation Feasibility

- Statevector: M ≈ 16 × 2^n bytes (complex128)
- Stabilizer: Clifford-heavy circuits, up to ~100 qubits
- MPS/tensor network: shallow circuits with limited entanglement

## Local Resource Detection

Only with user consent. Detects CPU cores, RAM, GPU type/memory.

---

# RFC-QCSC-003: Transformation Registry

## Status: Draft v0.1

## Abstract

QPU-saving opportunities are implemented as versioned transformations with guarantee classes, applicability conditions, and composition relations.

## Guarantee Classes

- EXACT: Mathematical result preserved
- EQUIVALENT: Statistically equivalent within tolerance
- APPROXIMATE: Known quality/resource trade-off
- EXPERIMENTAL: Requires user validation

## Initial Transformations (T01-T15)

T01: Redundant Measurement Bases (EXACT)
T02: Measurement Reuse (EXACT)
T03: Commuting Measurement Grouping (EXACT)
T04: Shot Reduction Opportunity (APPROXIMATE)
T05: Adaptive Shot Allocation (APPROXIMATE)
T06: Simulator Pretraining (APPROXIMATE)
T07: Parameter Transfer (APPROXIMATE)
T08: Repeated-Circuit Caching (EXACT)
T09: Graph/Problem Preprocessing (EXACT)
T10: Workload Packing (EXPERIMENTAL)
T11: Candidate Pruning (APPROXIMATE)
T12: Circuit Cutting Candidate (EXACT)
T13: CPU/GPU Simulator Routing (EXACT)
T14: Circuit Architecture Reduction (EXACT)
T15: Quantum/Classical Decomposition (EQUIVALENT)

## Composition Relations

INDEPENDENT, OVERLAPPING, REQUIRES, CONFLICTS_WITH, UNKNOWN

---

# RFC-QCSC-004: Cost Model

## Status: Draft v0.1

## Abstract

The cost model estimates baseline QPU workload from static analysis:

N_QPU_jobs = N_circuits × loop_multiplier
N_total_shots = N_QPU_jobs × N_measurement_settings × shots_per_circuit

## Multi-Objective

J = α·T_Q + β·T_C + γ·$ + δ·E

Where:
- T_Q: QPU resource use
- T_C: classical/HPC time
- $: monetary cost
- E: expected error/quality loss

Returns Pareto frontier, not single "best" plan.

---

# RFC-QCSC-005: Guarantee Classes

## Status: Draft v0.1

## Abstract

Every recommendation must declare a guarantee class and confidence level.

## Guarantee Classes

| Class | Meaning | Auto-apply? |
|-------|---------|-------------|
| EXACT | Mathematical result preserved | Safe to apply |
| EQUIVALENT | Statistically equivalent within ε | Needs tolerance config |
| APPROXIMATE | Known quality/resource trade-off | Needs validation |
| EXPERIMENTAL | Requires user validation | Never auto-apply |

## Confidence Levels

HIGH, MEDIUM, LOW — based on static certainty, mathematical guarantee, framework understanding, and missing runtime values.

## Rule

Only EXACT transformations may be automatically labelled safe.
No ML probabilities in v0.1.
