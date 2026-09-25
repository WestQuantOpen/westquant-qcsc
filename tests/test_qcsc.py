"""Tests for the WestQuant QCSC Optimizer."""
import json
import tempfile
from pathlib import Path

import pytest

from qcsc.flow_ir import (
    WorkflowGraph, WorkflowNode, WorkflowEdge,
    OperationType, ResourceType,
)
from qcsc.transformation_registry import (
    TRANSFORMATIONS, Recommendation, GuaranteeClass, Confidence,
    get_transformation, get_composition_relation, CompositionRelation,
)
from qcsc.resource_model import (
    statevector_memory_gb, is_statevector_feasible,
    is_qpu_backend, QPUBackend, ResourceCertainty,
)
from qcsc.analyzer.qiskit_analyzer import QiskitAnalyzer
from qcsc.detectors import run_all_detectors
from qcsc.planner import build_plans
from qcsc.cost_model import estimate_baseline_cost
from qcsc.reporting import write_reports


class TestFlowIR:
    def test_node_creation(self):
        node = WorkflowNode(
            node_id="n001",
            operation_type=OperationType.QPU_SAMPLE,
            framework="qiskit",
            resource_type=ResourceType.QUANTUM,
        )
        assert node.node_id == "n001"
        assert node.operation_type == OperationType.QPU_SAMPLE

    def test_graph_build(self):
        g = WorkflowGraph()
        g.add_node(WorkflowNode(node_id="n1", operation_type=OperationType.BUILD_CIRCUIT))
        g.add_node(WorkflowNode(node_id="n2", operation_type=OperationType.QPU_SAMPLE,
                               resource_type=ResourceType.QUANTUM))
        g.add_edge(WorkflowEdge(source_id="n1", target_id="n2"))
        assert len(g.nodes) == 2
        assert len(g.edges) == 1
        assert len(g.quantum_nodes()) == 1

    def test_topological_order(self):
        g = WorkflowGraph()
        g.add_node(WorkflowNode(node_id="a", operation_type=OperationType.LOAD_GRAPH))
        g.add_node(WorkflowNode(node_id="b", operation_type=OperationType.BUILD_CIRCUIT))
        g.add_node(WorkflowNode(node_id="c", operation_type=OperationType.QPU_SAMPLE))
        g.add_edge(WorkflowEdge(source_id="a", target_id="b"))
        g.add_edge(WorkflowEdge(source_id="b", target_id="c"))
        order = g.topological_order()
        assert order.index("a") < order.index("b")
        assert order.index("b") < order.index("c")

    def test_serialization_roundtrip(self):
        g = WorkflowGraph()
        g.add_node(WorkflowNode(node_id="n1", operation_type=OperationType.QPU_SAMPLE,
                               framework="qiskit", resource_type=ResourceType.QUANTUM))
        g.add_node(WorkflowNode(node_id="n2", operation_type=OperationType.POSTPROCESS))
        g.add_edge(WorkflowEdge(source_id="n1", target_id="n2"))
        d = g.to_dict()
        g2 = WorkflowGraph.from_dict(d)
        assert len(g2.nodes) == 2
        assert len(g2.edges) == 1


class TestTransformationRegistry:
    def test_15_transformations_exist(self):
        assert len(TRANSFORMATIONS) == 15

    def test_transformation_ids(self):
        ids = {t.id for t in TRANSFORMATIONS}
        for i in range(1, 16):
            assert f"T{i:02d}" in ids

    def test_get_transformation(self):
        t = get_transformation("T01")
        assert t is not None
        assert t.name == "Redundant Measurement Bases"
        assert t.guarantee_class == GuaranteeClass.EXACT

    def test_composition_relation(self):
        rel = get_composition_relation("T01", "T02")
        assert rel == CompositionRelation.OVERLAPPING


class TestResourceModel:
    def test_statevector_memory(self):
        # 20 qubits = 16 * 2^20 bytes = ~16 MB
        mem = statevector_memory_gb(20)
        assert 0.01 < mem < 0.02  # ~16 MB

    def test_statevector_feasibility(self):
        assert is_statevector_feasible(10)  # 10 qubits is feasible
        assert not is_statevector_feasible(40)  # 40 qubits is not

    def test_is_qpu_backend(self):
        assert is_qpu_backend("ibm_kyiv")  # real QPU
        assert not is_qpu_backend("aer_simulator")  # simulator
        assert not is_qpu_backend("fake_provider")  # fake


class TestQiskitAnalyzer:
    def test_analyze_maxcut_project(self):
        project = Path(__file__).parent.parent / "benchmark" / "maxcut_project"
        if not project.exists():
            pytest.skip("benchmark project not found")
        analyzer = QiskitAnalyzer()
        graph = analyzer.analyze_project(project)
        assert "qiskit" in graph.frameworks_detected
        assert "networkx" in graph.frameworks_detected
        assert len(graph.nodes) > 0
        # Should detect quantum boundaries
        summary = analyzer.get_summary()
        assert len(summary["quantum_boundaries"]) > 0

    def test_analyze_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = QiskitAnalyzer()
            graph = analyzer.analyze_project(tmpdir)
            assert len(graph.nodes) == 0

    def test_detect_shots(self):
        # Create a simple Python file with shots
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("""
from qiskit_ibm_runtime import Sampler
sampler = Sampler()
job = sampler.run([circuit], shots=4096)
""")
            analyzer = QiskitAnalyzer()
            graph = analyzer.analyze_project(tmpdir)
            summary = analyzer.get_summary()
            assert len(summary["shots_detected"]) > 0
            assert summary["shots_detected"][0]["shots"] == 4096


class TestDetectors:
    def test_run_all_detectors(self):
        summary = {
            "frameworks_detected": ["qiskit", "networkx"],
            "quantum_boundaries": [{"call": "sampler.run", "is_simulator": False, "location": "f:1"}],
            "circuits": [{"n_qubits": 8}],
            "observables": [{"type": "SparsePauliOp"}, {"type": "SparsePauliOp"}],
            "loops": [{"count": 10000}],
            "shots_detected": [{"shots": 2000, "location": "f:1"}],
        }
        graph = WorkflowGraph()
        recs = run_all_detectors(summary, graph)
        assert len(recs) > 0
        # Should detect shot reduction
        rec_ids = [r.transformation_id for r in recs]
        assert "T04" in rec_ids
        # Should detect graph preprocessing
        assert "T09" in rec_ids


class TestPlanner:
    def test_build_plans(self):
        recs = [
            Recommendation(
                transformation_id="T01", transformation_name="Redundant Bases",
                guarantee_class=GuaranteeClass.EXACT, confidence=Confidence.HIGH,
                what_detected="test", why_expensive="test", alternative="test",
                estimated_qpu_saving="2x", required_local_compute="none",
                expected_quality_risk="none", validation_experiment="test",
                evidence="test", estimated_saving_factor=2.0,
            ),
            Recommendation(
                transformation_id="T04", transformation_name="Shot Reduction",
                guarantee_class=GuaranteeClass.APPROXIMATE, confidence=Confidence.MEDIUM,
                what_detected="test", why_expensive="test", alternative="test",
                estimated_qpu_saving="4x", required_local_compute="none",
                expected_quality_risk="low", validation_experiment="test",
                evidence="test", estimated_saving_factor=4.0,
            ),
        ]
        plans = build_plans(recs)
        assert len(plans) == 3
        assert plans[0].name == "SAFE"
        assert len(plans[0].recommendations) == 1  # only EXACT
        assert plans[1].name == "BALANCED"
        assert len(plans[1].recommendations) == 2  # EXACT + APPROXIMATE
        assert plans[2].name == "AGGRESSIVE"
        assert len(plans[2].recommendations) == 2  # all


class TestCostModel:
    def test_estimate_baseline(self):
        summary = {
            "quantum_boundaries": [{"is_simulator": False}],
            "circuits": [{"n_qubits": 8}],
            "observables": [{"type": "Z"}],
            "loops": [{"count": 100}],
            "shots_detected": [{"shots": 2000}],
        }
        graph = WorkflowGraph()
        cost = estimate_baseline_cost(summary, graph)
        assert cost["n_qpu_jobs"] == 100
        assert cost["shots_per_circuit"] == 2000
        assert cost["total_shots"] == 200000


class TestReporting:
    def test_write_reports(self):
        summary = {
            "frameworks_detected": ["qiskit"],
            "quantum_boundaries": [],
            "circuits": [],
            "observables": [],
            "loops": [],
            "shots_detected": [],
        }
        graph = WorkflowGraph()
        graph.frameworks_detected = ["qiskit"]
        recs: list[Recommendation] = []
        plans = build_plans(recs)
        baseline = {"n_qpu_jobs": 0, "total_shots": 0}
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = write_reports(tmpdir, summary, graph, recs, plans, baseline)
            assert paths["markdown"].exists()
            assert paths["html"].exists()
            assert paths["plan"].exists()
            assert paths["workflow"].exists()
            # Verify JSON is valid
            plan_data = json.loads(paths["plan"].read_text())
            assert plan_data["schema_version"] == "wq-qcsc-plan-v0.1"
