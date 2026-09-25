"""Pulser source analyzer — static AST + framework-specific semantics.

Detects: Sequence, Register, Pulse, Channel, Waveform, simulate,
SerialBackend, QPUBackend, create_pulse, build_sequence.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from ..flow_ir import (
    WorkflowGraph, WorkflowNode, WorkflowEdge,
    OperationType, ResourceType,
)


PULSER_IMPORTS = {"pulser"}
PULSER_QUANTUM_CALLS = {
    "simulate", "build", "run", "get_result",
    "Sequence.simulate", "Sequence.build",
    "QPUBackend.run", "backend.run",
}
PULSER_QUANTUM_OBJECTS = {
    "Sequence", "Register", "Pulse", "Waveform", "Channel",
    "SerialBackend", "QPUBackend", "Emulator",
    "ConstantWaveform", "BlackmanWaveform", "InterpolatedWaveform",
}
PULSER_SIMULATOR_KEYWORDS = {
    "simulator", "emulator", "serial", "virtual",
    "qutip", "numpy", "classical",
}


class PulserAnalyzer:
    """Static analyzer for Pulser-based quantum projects."""

    def __init__(self) -> None:
        self.frameworks: set[str] = set()
        self.nodes: list[WorkflowNode] = []
        self.quantum_boundaries: list[dict[str, Any]] = []
        self.circuits: list[dict[str, Any]] = []
        self.observables: list[dict[str, Any]] = []
        self.loops: list[dict[str, Any]] = []
        self.shots_detected: list[dict[str, Any]] = []
        self._node_counter = 0

    def analyze_project(self, project_path: str | Path) -> WorkflowGraph:
        project = Path(project_path)
        for py_file in sorted(project.rglob("*.py")):
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
        if self._check_pulser_imports(tree):
            self.frameworks.add("pulser")

        for node in ast.walk(tree):
            self._visit_node(node, file_id)

    def _check_pulser_imports(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in PULSER_IMPORTS:
                        return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in PULSER_IMPORTS:
                    return True
        return False

    def _visit_node(self, node: ast.AST, file_id: str) -> None:
        if isinstance(node, ast.Call):
            self._visit_call(node, file_id)
        elif isinstance(node, (ast.For, ast.While)):
            self._visit_loop(node, file_id)

    def _visit_call(self, node: ast.Call, file_id: str) -> None:
        func_name = self._get_call_name(node)

        if func_name in PULSER_QUANTUM_CALLS or (func_name and "simulate" in func_name.lower()):
            is_sim = self._is_simulator_context(node)
            op_type = OperationType.SIMULATE if is_sim else OperationType.QPU_SAMPLE
            resource = ResourceType.CLASSICAL if is_sim else ResourceType.QUANTUM
            shots = self._extract_shots(node)

            self.quantum_boundaries.append({
                "type": "qpu_call" if not is_sim else "simulator_call",
                "call": func_name,
                "location": f"{file_id}:{node.lineno}",
                "shots": shots,
                "is_simulator": is_sim,
            })

            self._add_node(
                op_type, framework="pulser", resource_type=resource,
                source_location=f"{file_id}:{node.lineno}",
                semantic_tags=["qpu_boundary"] if not is_sim else ["simulator"],
                estimated_cost={"shots": shots or "unknown"},
                metadata={"call": func_name},
            )
            if shots:
                self.shots_detected.append({"location": f"{file_id}:{node.lineno}", "shots": shots, "call": func_name})

        elif func_name == "Sequence":
            self.circuits.append({"location": f"{file_id}:{node.lineno}"})
            self._add_node(
                OperationType.BUILD_CIRCUIT, framework="pulser",
                source_location=f"{file_id}:{node.lineno}",
                semantic_tags=["sequence"],
            )

        elif func_name == "Register" or func_name == "Register3D":
            n_atoms = self._extract_n_atoms(node)
            self._add_node(
                OperationType.LOAD_GRAPH, framework="pulser",
                source_location=f"{file_id}:{node.lineno}",
                semantic_tags=["register"],
                estimated_cost={"n_atoms": n_atoms or "unknown"},
            )

        elif func_name and ("Waveform" in func_name or "Pulse" in func_name):
            self._add_node(
                OperationType.BUILD_CIRCUIT, framework="pulser",
                source_location=f"{file_id}:{node.lineno}",
                semantic_tags=["pulse", "waveform"],
            )

    def _visit_loop(self, node: ast.For | ast.While, file_id: str) -> None:
        loop_info: dict[str, Any] = {"location": f"{file_id}:{node.lineno}", "type": type(node).__name__.lower()}
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Call):
            iter_name = self._get_call_name(node.iter)
            if iter_name == "range":
                args = node.iter.args
                if len(args) == 1 and isinstance(args[0], ast.Constant):
                    loop_info["count"] = args[0].value
                elif len(args) == 2 and all(isinstance(a, ast.Constant) for a in args):
                    loop_info["count"] = args[1].value - args[0].value
        self.loops.append(loop_info)
        self._add_node(
            OperationType.PARAMETER_SWEEP, framework="pulser",
            source_location=f"{file_id}:{node.lineno}",
            semantic_tags=["loop"], repeat_count=loop_info.get("count", 1),
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
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value.lower() in PULSER_SIMULATOR_KEYWORDS:
                    return True
        return False

    def _extract_shots(self, node: ast.Call) -> int | None:
        for kw in node.keywords:
            if kw.arg in ("n_shots", "shots", "samples") and isinstance(kw.value, ast.Constant):
                return kw.value.value
        return None

    def _extract_n_atoms(self, node: ast.Call) -> int | None:
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
            return node.args[0].value
        for kw in node.keywords:
            if kw.arg in ("n_atoms", "n_qubits") and isinstance(kw.value, ast.Constant):
                return kw.value.value
        return None

    def _add_node(self, op_type: OperationType, **kwargs: Any) -> str:
        self._node_counter += 1
        node_id = f"n{self._node_counter:04d}"
        self.nodes.append(WorkflowNode(node_id=node_id, operation_type=op_type, **kwargs))
        return node_id

    def _build_graph(self) -> WorkflowGraph:
        g = WorkflowGraph()
        g.frameworks_detected = sorted(self.frameworks)
        for node in self.nodes:
            g.add_node(node)
        node_ids = [n.node_id for n in self.nodes]
        for i in range(len(node_ids) - 1):
            g.add_edge(WorkflowEdge(source_id=node_ids[i], target_id=node_ids[i+1]))
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
            "n_nodes": len(self.nodes),
        }
