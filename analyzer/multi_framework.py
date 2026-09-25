"""Multi-framework analyzer — dispatches to framework-specific analyzers.

Detects which frameworks are used and runs the appropriate analyzers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..flow_ir import WorkflowGraph
from .qiskit_analyzer import QiskitAnalyzer
from .pennylane_analyzer import PennyLaneAnalyzer
from .pytket_analyzer import PytketAnalyzer
from .pulser_analyzer import PulserAnalyzer


class MultiFrameworkAnalyzer:
    """Runs all framework analyzers and merges results."""

    def __init__(self) -> None:
        self.analyzers = {
            "qiskit": QiskitAnalyzer(),
            "pennylane": PennyLaneAnalyzer(),
            "pytket": PytketAnalyzer(),
            "pulser": PulserAnalyzer(),
        }

    def analyze_project(self, project_path: str | Path) -> tuple[WorkflowGraph, dict[str, Any]]:
        """Analyze a project with all framework analyzers.

        Returns merged workflow graph and combined summary.
        """
        project = Path(project_path)
        all_summaries: list[dict[str, Any]] = []
        all_graphs: list[WorkflowGraph] = []

        for name, analyzer in self.analyzers.items():
            graph = analyzer.analyze_project(project)
            if graph.frameworks_detected or len(graph.nodes) > 0:
                summary = analyzer.get_summary()
                all_summaries.append((name, summary))
                all_graphs.append(graph)

        # Merge into single graph
        merged = WorkflowGraph()
        merged.frameworks_detected = sorted(
            set(fw for g in all_graphs for fw in g.frameworks_detected)
        )

        # Merge nodes with framework prefix
        node_offset = 0
        for graph in all_graphs:
            id_map: dict[str, str] = {}
            for node in graph.nodes.values():
                new_id = f"n{node_offset:04d}"
                node_offset += 1
                id_map[node.node_id] = new_id
                node.node_id = new_id
                merged.add_node(node)
            for edge in graph.edges:
                merged.add_edge(type(edge)(
                    source_id=id_map[edge.source_id],
                    target_id=id_map[edge.target_id],
                    edge_type=edge.edge_type,
                    metadata=edge.metadata,
                ))

        # Merge summaries
        combined: dict[str, Any] = {
            "frameworks_detected": merged.frameworks_detected,
            "quantum_boundaries": [],
            "circuits": [],
            "observables": [],
            "loops": [],
            "shots_detected": [],
            "n_nodes": len(merged.nodes),
        }
        for name, summary in all_summaries:
            for key in ("quantum_boundaries", "circuits", "observables", "loops", "shots_detected"):
                for item in summary.get(key, []):
                    if isinstance(item, dict):
                        item["framework"] = name
                    combined[key].append(item)

        return merged, combined
