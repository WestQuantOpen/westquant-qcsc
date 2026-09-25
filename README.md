# WestQuant QCSC Optimizer

> Use the QPU only where the QPU is actually needed.

An open-source, software-agnostic analyzer and optimizer for hybrid quantum-classical projects. The system inspects a user's quantum codebase and answers:

**Which planned quantum operations actually need QPU hardware?**

## Core distinction

Existing hybrid quantum systems often solve: *Where should task T execute?*

WestQuant solves the deeper problem: **Should task T require a QPU at all?**

This is **Semantic QPU Minimization**.

## V0.1 — Advisory Only

V0.1 performs static analysis + cost estimation + recommendations. It does NOT:
- modify user code
- execute user code
- submit jobs to hardware
- automatically rewrite experiments

## Quick Start

```bash
# Install
pip install -e .

# Audit a project
westquant audit ./my_quantum_project

# Specify output directory
westquant audit ./my_quantum_project -o ./reports
```

## Output

```
westquant_report.html    # Full HTML report
westquant_report.md      # Markdown report
westquant_plan.json      # Machine-readable plan (SAFE/BALANCED/AGGRESSIVE)
westquant_workflow.json  # WQFlow IR graph
```

## What It Detects

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

## Three Plans

- **SAFE**: Only EXACT transformations. Mathematical result preserved.
- **BALANCED**: EXACT + EQUIVALENT + low-risk APPROXIMATE. Requires tolerance ε.
- **AGGRESSIVE**: All transformations including EXPERIMENTAL. Always shows uncertainty.

## Architecture

```
qcsc/
├── flow_ir/              # WQFlow IR — directed graph of hybrid workflows
├── resource_model/       # QPU backend + local compute capabilities
├── transformation_registry/  # 15 QPU-saving transformations (T01-T15)
├── analyzer/             # Framework-specific source analysis (Qiskit first)
├── detectors/            # Detectors that produce recommendations
├── planner/              # Pareto planner (SAFE/BALANCED/AGGRESSIVE)
├── reporting/            # HTML + Markdown + JSON report generation
├── benchmark/            # Benchmark projects with known opportunities
├── experiments/          # Validation experiment protocols
├── rfc/                  # RFC specifications
└── tests/                # Test suite
```

## Design Philosophy

WestQuant asks three questions in order:
1. Does this operation need to happen?
2. Does it need to be quantum?
3. Only then: which resource should execute it?

That ordering is the defining idea of the project.

## License

MIT
