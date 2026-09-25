"""MaxCut QAOA benchmark project — 10,000 graph instances.

This project has known QPU-saving opportunities:
- Redundant measurement bases (all observables are Z-basis)
- High shot budget (2000 shots, could be reduced)
- Simulator pretraining feasible (circuits <= 8 qubits)
- Graph preprocessing (NetworkX detected)
- Repeated circuit caching opportunity (loop resubmits)
- Workload packing (small circuits on larger QPU)
"""
import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import Sampler, Estimator, Session
from qiskit.transpiler import generate_preset_pass_manager

# Load 10,000 graphs
graphs = []
for i in range(10000):
    g = nx.erdos_renyi_graph(8, 0.5, seed=i)
    graphs.append(g)

# Build QAOA circuit (p=2)
def build_qaoa(graph, p=2):
    n = graph.number_of_nodes()
    qc = QuantumCircuit(n)
    # Initial state
    for q in range(n):
        qc.h(q)
    # QAOA layers
    for layer in range(p):
        # Cost unitary
        for (i, j) in graph.edges():
            qc.cx(i, j)
            qc.rz(0.5, j)
            qc.cx(i, j)
        # Mixer
        for q in range(n):
            qc.rx(0.3, q)
    return qc

# Build observables — all Z-basis (redundant bases opportunity)
observables = []
for i in range(8):
    z_str = "I" * i + "Z" + "I" * (7 - i)
    observables.append(SparsePauliOp.from_list([(z_str, 1.0)]))

# Additional ZZ correlators (measurement reuse opportunity)
for i in range(7):
    zz_str = "I" * i + "ZZ" + "I" * (6 - i)
    observables.append(SparsePauliOp.from_list([(zz_str, 1.0)]))

# Transpile
pm = generate_preset_pass_manager(optimization_level=2)

# Run on QPU — 10,000 graphs × 2 measurement bases × 2000 shots
backend = "ibm_kyiv"
sampler = Sampler(mode=backend)
estimator = Estimator(mode=backend)

results = []
for i, graph in enumerate(graphs):
    qc = build_qaoa(graph)
    tqc = pm.run(qc)

    # Two measurement bases (but all observables are Z-diagonal!)
    for basis in range(2):
        job = sampler.run([tqc], shots=2000)
        result = job.result()
        results.append(result)

# Post-processing
cut_values = []
for result in results:
    # Classical post-processing of bitstrings
    counts = result[0].data.meas.get_counts()
    best_cut = max(counts.keys(), key=lambda x: sum(int(b) for b in x))
    cut_values.append(best_cut)

print(f"Processed {len(results)} jobs")
print(f"Best cut: {max(cut_values)}")
