<div align="center">

# WestQuant QCSC Optimizer

### Use the QPU only where the QPU is actually needed.

**Semantic QPU Minimization for hybrid quantum-classical workflows.**

</div>

---

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/westquant-qcsc)](https://pypi.org/project/westquant-qcsc/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](https://opensource.org/licenses/MIT)
[![Discussions](https://img.shields.io/badge/discussions-join%20us-blue)](https://github.com/orgs/WestQuantOpen/discussions)

</div>

---

An open-source, software-agnostic analyzer and optimizer for hybrid quantum-classical projects. The system inspects a user's quantum codebase and answers:

**Which planned quantum operations actually need QPU hardware?**

## Core distinction

Existing hybrid quantum systems often solve: *Where should task T execute?*

WestQuant solves the deeper problem: **Should task T require a QPU at all?**

This is **Semantic QPU Minimization**.

## Quick start

```bash
pip install westquant-qcsc
```

### Audit a project

```bash
westquant audit ./my_quantum_project
```

Output:
```
westquant_report.html    # Full HTML report
westquant_report.md      # Markdown report
westquant_plan.json      # Machine-readable plan (SAFE/BALANCED/AGGRESSIVE)
westquant_workflow.json  # WQFlow IR graph
```

### QPU Budget Predictor

Estimate the required QPU budget from problem structure — before any quantum execution:

```python
from qcsc import QPUBudgetPredictor, ProblemProfile

profile = ProblemProfile(
    problem="MaxCut",
    n_qubits=16,
    p=3,
    graph_family="GEO",
    quality_target=0.95,
)
predictor = QPUBudgetPredictor()
budget = predictor.estimate(profile)

print(f"Shots: {budget.recommended_shots:,}")
print(f"Reduction vs naive: {budget.reduction_vs_naive:.0f}x")
print(f"Estimated quality: {budget.estimated_quality:.4f}")
print(f"Strategies: {budget.strategies}")
```

Or from the CLI:

```bash
westquant budget --problem MaxCut --qubits 16 --depth 3 --family GEO --compare
```

Output:
```
  [B0_naive]
    Shots:         2,119,680
    Est. quality:  0.9961

  [B1_best_practice]
    Shots:         1,024
    Est. quality:  0.9961
    vs naive:      2070.0x reduction

  [B2_classical]
    Shots:         0
    Est. quality:  1.0000
```

The predictor is calibrated from 12,600 exact simulation results across 6 problems, 7 graph families, and depths p=1,2,3. See [Paper A](https://github.com/WestQuantOpen/qpu-mini) for the underlying research.

## What it detects

| ID | Transformation | Guarantee | Description |
|----|---------------|-----------|-------------|
| T01 | Redundant Measurement Bases | EXACT | Observables obtainable from same basis |
| T02 | Measurement Reuse | EXACT | Observables recoverable from existing samples |
| T03 | Commuting Measurement Grouping | EXACT | Group compatible Pauli observables |
| T04 | Shot Reduction Opportunity | APPROXIMATE | Detect high shot budgets, propose ablation |
| T05 | Adaptive Shot Allocation | APPROXIMATE | Variable shot budgets by optimization stage |
| T06 | Simulator Pretraining | APPROXIMATE | Pretrain variational parameters on simulator |
| T07 | Parameter Transfer | APPROXIMATE | Transfer params between structurally similar instances |
| T08 | Repeated-Circuit Caching | EXACT | Cache identical circuit+parameter submissions |
| T09 | Graph/Problem Preprocessing | EXACT | Classical graph reduction (components, symmetry) |
| T10 | Workload Packing | EXPERIMENTAL | Pack independent circuits on larger QPU |
| T11 | Candidate Pruning | APPROXIMATE | Surrogate filtering for large parameter sweeps |
| T12 | Circuit Cutting Candidate | EXACT | Identify cuttable interaction graphs |
| T13 | CPU/GPU Simulator Routing | EXACT | Recommend best simulator class |
| T14 | Circuit Architecture Reduction | EXACT | Remove mathematically redundant gates |
| T15 | Quantum/Classical Decomposition | EQUIVALENT | Split quantum sampling from classical post-processing |

## Three plans

- **SAFE**: Only EXACT transformations. Mathematical result preserved.
- **BALANCED**: EXACT + EQUIVALENT + low-risk APPROXIMATE. Requires tolerance ε.
- **AGGRESSIVE**: All transformations including EXPERIMENTAL. Always shows uncertainty.

## Architecture

```
qcsc/
├── qpu_budget.py              # QPU budget predictor (Paper A integration)
├── flow_ir/                    # WQFlow IR — directed graph of hybrid workflows
├── resource_model/             # QPU backend + local compute capabilities
├── transformation_registry/    # 15 QPU-saving transformations (T01-T15)
├── analyzer/                   # Framework-specific source analysis (Qiskit first)
├── detectors/                  # Detectors that produce recommendations
├── planner/                    # Pareto planner (SAFE/BALANCED/AGGRESSIVE)
├── reporting/                  # HTML + Markdown + JSON report generation
├── benchmark/                  # Benchmark projects with known opportunities
├── cost_model.py               # QPU cost estimation
├── closed_loop.py              # Closed-loop measurement tracking
├── learned_policy.py           # Learned policy for transformation ranking
├── patch_generator.py         # Advisory patch generation
├── runner.py                   # Supervised plan execution
└── cli.py                      # CLI entry point
```

## Design philosophy

WestQuant asks three questions in order:

1. **Does this operation need to happen?** (Eliminate)
2. **Does it need to be quantum?** (Replace with classical)
3. **Only then: which resource should execute it?** (Schedule)

That ordering is the defining idea of the project.

## CUDA-Q and cuQuantum

Version 0.1.1 adds advisory capability records for CUDA-Q CPU/GPU targets and
cuStateVec, cuTensorNet, cuDensityMat, cuPauliProp, and cuStabilizer. Install
`westquant-qcsc[cudaq]` to compose QPU minimization with `westquant-cudaq`.
The integration is domain-neutral and does not include vertical-specific
scientific methods.

## V0.1 — Advisory only

V0.1 performs static analysis + cost estimation + recommendations. It does NOT:
- modify user code
- execute user code
- submit jobs to hardware
- automatically rewrite experiments

## Community

- **Discussions:** [Join the conversation](https://github.com/orgs/WestQuantOpen/discussions)
- **Contributing:** See [CONTRIBUTING.md](https://github.com/WestQuantOpen/.github/blob/main/CONTRIBUTING.md)
- **Code of Conduct:** See [CODE_OF_CONDUCT.md](https://github.com/WestQuantOpen/.github/blob/main/CODE_OF_CONDUCT.md)

## License

MIT
