"""Qiskit source analyzer — static AST + framework-specific semantic analysis.

Layer A: Python AST (loops, function calls, variables, backend calls)
Layer B: Qiskit-specific semantics (QuantumCircuit, Sampler, Estimator, etc.)
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from ..flow_ir import (
    WorkflowGraph, WorkflowNode, WorkflowEdge,
    OperationType, ResourceType, CostCertainty,
)


# Qiskit API patterns to detect
QISKIT_IMPORTS = {"qiskit", "qiskit_aer", "qiskit_ibm_runtime", "qiskit_algorithms"}
QISKIT_QUANTUM_CALLS = {
    "backend.run", "Backend.run",
    "Sampler.run", "Estimator.run",
    "sampler.run", "estimator.run",
    "transpile", "generate_preset_pass_manager",
    "execute",
}
QISKIT_QUANTUM_OBJECTS = {
    "QuantumCircuit", "SparsePauliOp", "Pauli", "Operator",
    "Estimator", "Sampler", "Backend", "BackendV2",
    "Session", "IBMQ", "Provider",
}
QISKIT_SIMULATOR_KEYWORDS = {
    "simulator", "aer", "statevector", "mps", "stabilizer",
    "fake", "mock", "basic", "density_matrix",
}


class QiskitAnalyzer:
    """Static analyzer for Qiskit-based quantum projects."""

    def __init__(self) -> None:
        self.frameworks: set[str] = set()
        self.nodes: list[WorkflowNode] = []
        self.edges: list[WorkflowEdge] = []
        self.quantum_boundaries: list[dict[str, Any]] = []
        self.circuits: list[dict[str, Any]] = []
        self.observables: list[dict[str, Any]] = []
        self.backends: list[dict[str, Any]] = []
        self.loops: list[dict[str, Any]] = []
        self.shots_detected: list[dict[str, Any]] = []
        self._node_counter = 0

    def analyze_project(self, project_path: str | Path) -> WorkflowGraph:
        """Analyze a Python project directory."""
        project = Path(project_path)
        python_files = sorted(project.rglob("*.py"))

        for py_file in python_files:
            # Skip hidden dirs, __pycache__, .venv
            parts = py_file.parts
            if any(p.startswith(".") or p == "__pycache__" or p == ".venv" for p in parts):
                continue
            self._analyze_file(py_file)

        return self._build_graph()

    def _analyze_file(self, filepath: Path) -> None:
        try:
            source = filepath.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source, filename=str(filepath))
        except SyntaxError:
            return
        except Exception:
            return

        file_id = str(filepath)
        has_qiskit = self._check_qiskit_imports(tree)

        if has_qiskit:
            self.frameworks.add("qiskit")

        # Walk AST for patterns
        for node in ast.walk(tree):
            self._visit_node(node, file_id, source)

    def _check_qiskit_imports(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in QISKIT_IMPORTS:
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in QISKIT_IMPORTS:
                    return True
        return False

    def _visit_node(self, node: ast.AST, file_id: str, source: str) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in QISKIT_IMPORTS:
                    self._add_node(
                        OperationType.CLASSICAL_COMPUTE,
                        framework="qiskit",
                        source_location=f"{file_id}:{node.lineno}",
                        semantic_tags=["import"],
                        metadata={"module": alias.name},
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in QISKIT_IMPORTS:
                imported_names = [alias.name for alias in node.names]
                self._add_node(
                    OperationType.CLASSICAL_COMPUTE,
                    framework="qiskit",
                    source_location=f"{file_id}:{node.lineno}",
                    semantic_tags=["import"],
                    metadata={"module": node.module, "names": imported_names},
                )
                # Check for quantum object imports
                for name in imported_names:
                    if name in QISKIT_QUANTUM_OBJECTS:
                        self.quantum_boundaries.append({
                            "type": "import",
                            "object": name,
                            "location": f"{file_id}:{node.lineno}",
                        })

        elif isinstance(node, ast.Call):
            self._visit_call(node, file_id)

        elif isinstance(node, (ast.For, ast.While)):
            self._visit_loop(node, file_id)

    def _visit_call(self, node: ast.Call, file_id: str) -> None:
        func_name = self._get_call_name(node)

        # Backend.run / Sampler.run / Estimator.run — QPU boundary
        if func_name in QISKIT_QUANTUM_CALLS:
            is_sim = self._is_simulator_context(node)
            op_type = OperationType.SIMULATE if is_sim else OperationType.QPU_SAMPLE
            resource = ResourceType.CLASSICAL if is_sim else ResourceType.QUANTUM

            # Extract shots if available
            shots = self._extract_shots(node)

            boundary = {
                "type": "qpu_call" if not is_sim else "simulator_call",
                "call": func_name,
                "location": f"{file_id}:{node.lineno}",
                "shots": shots,
                "is_simulator": is_sim,
            }
            self.quantum_boundaries.append(boundary)

            self._add_node(
                op_type,
                framework="qiskit",
                resource_type=resource,
                source_location=f"{file_id}:{node.lineno}",
                semantic_tags=["qpu_boundary"] if not is_sim else ["simulator"],
                estimated_cost={"shots": shots or "unknown"},
                metadata={"call": func_name, "is_simulator": is_sim},
            )

            if shots:
                self.shots_detected.append({
                    "location": f"{file_id}:{node.lineno}",
                    "shots": shots,
                    "call": func_name,
                })

        # QuantumCircuit construction
        elif func_name == "QuantumCircuit":
            n_qubits = self._extract_n_qubits(node)
            self.circuits.append({
                "location": f"{file_id}:{node.lineno}",
                "n_qubits": n_qubits,
            })
            self._add_node(
                OperationType.BUILD_CIRCUIT,
                framework="qiskit",
                source_location=f"{file_id}:{node.lineno}",
                estimated_cost={"n_qubits": n_qubits or "unknown"},
                semantic_tags=["circuit_construction"],
            )

        # SparsePauliOp / observable construction
        elif func_name in ("SparsePauliOp", "Pauli", "PauliList"):
            self.observables.append({
                "location": f"{file_id}:{node.lineno}",
                "type": func_name,
            })
            self._add_node(
                OperationType.BUILD_HAMILTONIAN,
                framework="qiskit",
                source_location=f"{file_id}:{node.lineno}",
                semantic_tags=["observable"],
                metadata={"type": func_name},
            )

        # transpile
        elif func_name == "transpile":
            self._add_node(
                OperationType.TRANSPILE,
                framework="qiskit",
                source_location=f"{file_id}:{node.lineno}",
                semantic_tags=["transpilation"],
            )

        # NetworkX graph operations
        elif func_name and "nx" in func_name.lower():
            self._add_node(
                OperationType.LOAD_GRAPH,
                framework="networkx",
                source_location=f"{file_id}:{node.lineno}",
                semantic_tags=["graph"],
            )
            self.frameworks.add("networkx")

    def _visit_loop(self, node: ast.For | ast.While, file_id: str) -> None:
        loop_info: dict[str, Any] = {
            "location": f"{file_id}:{node.lineno}",
            "type": type(node).__name__.lower(),
        }

        # Try to determine iteration count
        if isinstance(node, ast.For):
            if isinstance(node.iter, ast.Call):
                iter_name = self._get_call_name(node.iter)
                if iter_name == "range":
                    # Extract range arguments
                    args = node.iter.args
                    if len(args) == 1 and isinstance(args[0], ast.Constant):
                        loop_info["count"] = args[0].value
                    elif len(args) == 2 and all(isinstance(a, ast.Constant) for a in args):
                        loop_info["count"] = args[1].value - args[0].value
                    elif len(args) == 3 and all(isinstance(a, ast.Constant) for a in args):
                        loop_info["count"] = (args[1].value - args[0].value) // args[2].value
            elif isinstance(node.iter, ast.Name):
                loop_info["iter_var"] = node.iter.id

        self.loops.append(loop_info)
        self._add_node(
            OperationType.PARAMETER_SWEEP,
            framework="qiskit",
            source_location=f"{file_id}:{node.lineno}",
            semantic_tags=["loop"],
            repeat_count=loop_info.get("count", 1),
            metadata=loop_info,
        )

    def _get_call_name(self, node: ast.Call) -> str | None:
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None

    def _is_simulator_context(self, node: ast.Call) -> bool:
        """Heuristic: check if the call context suggests a simulator."""
        # Check if any string argument contains simulator keywords
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.lower() in QISKIT_SIMULATOR_KEYWORDS:
                    return True
        for kw in node.keywords:
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                if kw.value.value.lower() in QISKIT_SIMULATOR_KEYWORDS:
                    return True
        return False

    def _extract_shots(self, node: ast.Call) -> int | None:
        """Extract shots from a run() call."""
        for kw in node.keywords:
            if kw.arg == "shots" and isinstance(kw.value, ast.Constant):
                return kw.value.value
        # Check positional args (shots is often 2nd or 3rd arg)
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                if arg.value > 100 and arg.value <= 1000000:
                    return arg.value
        return None

    def _extract_n_qubits(self, node: ast.Call) -> int | None:
        """Extract n_qubits from QuantumCircuit() call."""
        if node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, int):
                return first.value
        for kw in node.keywords:
            if kw.arg in ("num_qubits", "n_qubits") and isinstance(kw.value, ast.Constant):
                return kw.value.value
        return None

    def _add_node(self, op_type: OperationType, **kwargs: Any) -> str:
        self._node_counter += 1
        node_id = f"n{self._node_counter:04d}"
        node = WorkflowNode(
            node_id=node_id,
            operation_type=op_type,
            **kwargs,
        )
        self.nodes.append(node)
        return node_id

    def _build_graph(self) -> WorkflowGraph:
        g = WorkflowGraph()
        g.frameworks_detected = sorted(self.frameworks)

        for node in self.nodes:
            g.add_node(node)

        # Build edges from topological order (sequential data flow)
        node_ids = [n.node_id for n in self.nodes]
        for i in range(len(node_ids) - 1):
            g.add_edge(WorkflowEdge(
                source_id=node_ids[i],
                target_id=node_ids[i + 1],
                edge_type="data_flow",
            ))

        g.metadata = {
            "n_quantum_boundaries": len(self.quantum_boundaries),
            "n_circuits": len(self.circuits),
            "n_observables": len(self.observables),
            "n_loops": len(self.loops),
            "n_shots_detected": len(self.shots_detected),
        }

        return g

    def get_summary(self) -> dict[str, Any]:
        return {
            "frameworks_detected": sorted(self.frameworks),
            "quantum_boundaries": self.quantum_boundaries,
            "circuits": self.circuits,
            "observables": self.observables,
            "loops": self.loops,
            "shots_detected": self.shots_detected,
            "backends": self.backends,
            "n_nodes": len(self.nodes),
        }
