# WestQuant QCSC Optimizer — Audit Report

> Use the QPU only where the QPU is actually needed.

## Executive Summary

- Frameworks detected: networkx, qiskit
- Quantum boundaries detected: 0
- Circuits detected: 0
- Observables detected: 0
- Loops detected: 0
- QPU-saving opportunities: 3 (2 EXACT)

## Frameworks Detected

- networkx
- qiskit

## Workflow Map

- Total nodes: 18
- Quantum nodes: 2
- Classical nodes: 16
- Edges: 17

## Quantum Boundaries

- [QPU] unknown at /Users/davidvesterlund/freedomcode/freedomcode/westquant/qcsc/benchmark/maxcut_project/maxcut_qaoa.py:13
- [QPU] unknown at /Users/davidvesterlund/freedomcode/freedomcode/westquant/qcsc/benchmark/maxcut_project/maxcut_qaoa.py:14
- [QPU] unknown at /Users/davidvesterlund/freedomcode/freedomcode/westquant/qcsc/benchmark/maxcut_project/maxcut_qaoa.py:15
- [QPU] unknown at /Users/davidvesterlund/freedomcode/freedomcode/westquant/qcsc/benchmark/maxcut_project/maxcut_qaoa.py:15
- [QPU] unknown at /Users/davidvesterlund/freedomcode/freedomcode/westquant/qcsc/benchmark/maxcut_project/maxcut_qaoa.py:15
- [QPU] generate_preset_pass_manager at /Users/davidvesterlund/freedomcode/freedomcode/westquant/qcsc/benchmark/maxcut_project/maxcut_qaoa.py:55
- [QPU] sampler.run at /Users/davidvesterlund/freedomcode/freedomcode/westquant/qcsc/benchmark/maxcut_project/maxcut_qaoa.py:69
  - Shots: 2000

## Baseline Quantum Workload

- n_circuits: 1
- n_qpu_jobs: 1120000
- n_measurement_settings: 1
- shots_per_circuit: 2000
- total_shots: 2240000000
- max_qubits: 0
- loop_multiplier: 1120000
- certainty: estimated
- note: Costs are estimated from static analysis. Runtime values may differ.

## QPU-Saving Opportunities

### Opportunity 1: Shot Reduction Opportunity

**Guarantee:** [APPROXIMATE]
  **Confidence:** MEDIUM

**What was detected:** Uniform high shot budget: 2000 shots per circuit
**Why it's expensive:** Total shots: 2000 across 1 measurement calls
**Alternative:** Subsample existing bitstrings at reduced shot counts, compare metrics
**Projected QPU saving:** 2-10x reduction (e.g., 2000 -> 500 shots)
**Required local compute:** Classical subsampling and variance analysis
**Expected quality risk:** Increased estimator variance — requires tolerance configuration
**Validation:** Subsample at 2000, 1000, 500 shots, compare task metric within 95% CI
**Evidence:** Shot budget of 2000 may be higher than necessary for convergence

### Opportunity 2: Repeated-Circuit Caching

**Guarantee:** [EXACT]
  **Confidence:** LOW

**What was detected:** Loops detected that may resubmit identical circuits
**Why it's expensive:** Identical circuits submitted multiple times to QPU
**Alternative:** Cache results for identical circuit+parameter combinations
**Projected QPU saving:** Eliminates redundant QPU submissions entirely
**Required local compute:** Hash storage for circuit+parameter keys
**Expected quality risk:** None — cached results are identical
**Validation:** Hash circuit+parameters, verify cache hits
**Evidence:** Loop structures detected alongside QPU calls — caching may eliminate redundant submissions

### Opportunity 3: Graph/Problem Preprocessing

**Guarantee:** [EXACT]
  **Confidence:** MEDIUM

**What was detected:** NetworkX graph operations detected alongside quantum workload
**Why it's expensive:** Graph optimization problems may contain trivially reducible components
**Alternative:** Apply classical graph preprocessing: component reduction, symmetry, sparsification, isolated-node removal
**Projected QPU saving:** Reduces problem size — savings depend on graph structure
**Required local compute:** Classical graph analysis (NetworkX)
**Expected quality risk:** None — preprocessing preserves exact problem structure
**Validation:** Verify reduced graph produces same optimal solution
**Evidence:** NetworkX detected — graph preprocessing can reduce QPU workload

## Recommended Plans

### SAFE
  Variable — depends on workload structure

  Only EXACT transformations. Mathematical result preserved.
  Transformations: 2
  Assumptions:
    - All transformations are mathematically exact
  Uncertainty: low

### BALANCED
  4.0x projected reduction

  EXACT + EQUIVALENT + low-risk APPROXIMATE transformations. Requires user-configured tolerance ε.
  Transformations: 3
  Assumptions:
    - User must configure tolerance ε for APPROXIMATE transformations
  Uncertainty: medium

### AGGRESSIVE
  4.0x projected reduction

  All transformations including EXPERIMENTAL. Always show uncertainty.
  Transformations: 3
  Assumptions:
    - Experimental transformations require hardware validation
  Uncertainty: high

## Uncertainty and Assumptions

- This is an advisory-only report. No code was modified.
- Cost estimates are based on static analysis and may differ from runtime values.
- Composition of transformations does not simply multiply savings.
- Experimental transformations require hardware validation.
