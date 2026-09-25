"""WQFlow IR — Directed graph representation of hybrid quantum-classical workflows.

Every node represents an operation with:
- node_id: unique identifier
- operation_type: LOAD_GRAPH, BUILD_CIRCUIT, QPU_SAMPLE, etc.
- framework: qiskit, pennylane, pytket, pulser, or "classical"
- classical_or_quantum: "classical" or "quantum"
- inputs/outputs: list of node_ids
- estimated_cost: dict of cost estimates
- resource_requirements: dict of resource needs
- semantic_tags: list of semantic labels
- repeat_count: how many times this operation repeats
- dependency_ids: list of dependent node_ids
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OperationType(str, Enum):
    LOAD_GRAPH = "load_graph"
    GENERATE_GRAPH = "generate_graph"
    BUILD_HAMILTONIAN = "build_hamiltonian"
    BUILD_CIRCUIT = "build_circuit"
    OPTIMIZE_PARAMETERS = "optimize_parameters"
    SIMULATE = "simulate"
    TRANSPILE = "transpile"
    QPU_SAMPLE = "qpu_sample"
    ESTIMATE_OBSERVABLE = "estimate_observable"
    POSTPROCESS = "postprocess"
    TRAIN_MODEL = "train_model"
    CLASSICAL_COMPUTE = "classical_compute"
    MEASUREMENT = "measurement"
    PARAMETER_SWEEP = "parameter_sweep"
    UNKNOWN = "unknown"


class ResourceType(str, Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"


class CostCertainty(str, Enum):
    EXACT = "exact"
    BOUNDED = "bounded"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


@dataclass
class WorkflowNode:
    node_id: str
    operation_type: OperationType
    framework: str = "classical"
    resource_type: ResourceType = ResourceType.CLASSICAL
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    estimated_cost: dict[str, Any] = field(default_factory=dict)
    resource_requirements: dict[str, Any] = field(default_factory=dict)
    semantic_tags: list[str] = field(default_factory=list)
    repeat_count: int = 1
    dependency_ids: list[str] = field(default_factory=list)
    source_location: str | None = None  # file:line
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "operation_type": self.operation_type.value,
            "framework": self.framework,
            "resource_type": self.resource_type.value,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "estimated_cost": self.estimated_cost,
            "resource_requirements": self.resource_requirements,
            "semantic_tags": self.semantic_tags,
            "repeat_count": self.repeat_count,
            "dependency_ids": self.dependency_ids,
            "source_location": self.source_location,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowEdge:
    source_id: str
    target_id: str
    edge_type: str = "data_flow"  # data_flow, control_flow, dependency
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "metadata": self.metadata,
        }


@dataclass
class WorkflowGraph:
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    edges: list[WorkflowEdge] = field(default_factory=list)
    frameworks_detected: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: WorkflowNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"duplicate node id: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: WorkflowEdge) -> None:
        if edge.source_id not in self.nodes:
            raise ValueError(f"missing source node: {edge.source_id}")
        if edge.target_id not in self.nodes:
            raise ValueError(f"missing target node: {edge.target_id}")
        self.edges.append(edge)

    def quantum_nodes(self) -> list[WorkflowNode]:
        return [n for n in self.nodes.values() if n.resource_type == ResourceType.QUANTUM]

    def classical_nodes(self) -> list[WorkflowNode]:
        return [n for n in self.nodes.values() if n.resource_type == ResourceType.CLASSICAL]

    def topological_order(self) -> list[str]:
        """Return node IDs in topological order (dependencies first)."""
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}
        for edge in self.edges:
            adj[edge.source_id].append(edge.target_id)
            in_degree[edge.target_id] += 1

        queue = [nid for nid, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            nid = queue.pop(0)
            order.append(nid)
            for nxt in adj[nid]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)
        return order

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "frameworks_detected": self.frameworks_detected,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowGraph:
        g = cls()
        g.frameworks_detected = data.get("frameworks_detected", [])
        g.metadata = data.get("metadata", {})
        for n in data.get("nodes", []):
            node = WorkflowNode(
                node_id=n["node_id"],
                operation_type=OperationType(n["operation_type"]),
                framework=n.get("framework", "classical"),
                resource_type=ResourceType(n.get("resource_type", "classical")),
                inputs=n.get("inputs", []),
                outputs=n.get("outputs", []),
                estimated_cost=n.get("estimated_cost", {}),
                resource_requirements=n.get("resource_requirements", {}),
                semantic_tags=n.get("semantic_tags", []),
                repeat_count=n.get("repeat_count", 1),
                dependency_ids=n.get("dependency_ids", []),
                source_location=n.get("source_location"),
                metadata=n.get("metadata", {}),
            )
            g.add_node(node)
        for e in data.get("edges", []):
            g.add_edge(WorkflowEdge(
                source_id=e["source_id"],
                target_id=e["target_id"],
                edge_type=e.get("edge_type", "data_flow"),
                metadata=e.get("metadata", {}),
            ))
        return g
