"""Tests for the WestQuant QCSC Optimizer — all phases."""
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
    is_qpu_backend, QPUBackend, ResourceCertainty, cudaq_capability_records,
)
from qcsc.analyzer.qiskit_analyzer import QiskitAnalyzer
from qcsc.analyzer.pennylane_analyzer import PennyLaneAnalyzer
from qcsc.analyzer.pytket_analyzer import PytketAnalyzer
from qcsc.analyzer.pulser_analyzer import PulserAnalyzer
from qcsc.analyzer.multi_framework import MultiFrameworkAnalyzer
from qcsc.detectors import run_all_detectors
from qcsc.planner import build_plans
from qcsc.cost_model import estimate_baseline_cost
from qcsc.reporting import write_reports
from qcsc.patch_generator import generate_patches, write_patch_file
from qcsc.runner import run_plan
from qcsc.closed_loop import ClosedLoopTracker
from qcsc.learned_policy import LearnedPolicy


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
        mem = statevector_memory_gb(20)
        assert 0.01 < mem < 0.02

    def test_statevector_feasibility(self):
        assert is_statevector_feasible(10)
        assert not is_statevector_feasible(40)

    def test_is_qpu_backend(self):
        assert is_qpu_backend("ibm_kyiv")
        assert not is_qpu_backend("aer_simulator")
        assert not is_qpu_backend("fake_provider")


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
        summary = analyzer.get_summary()
        assert len(summary["quantum_boundaries"]) > 0

    def test_analyze_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = QiskitAnalyzer()
            graph = analyzer.analyze_project(tmpdir)
            assert len(graph.nodes) == 0

    def test_detect_shots(self):
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


class TestPennyLaneAnalyzer:
    def test_detect_pennylane(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("""
import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", wires=4, shots=1024)

@qml.qnode(dev)
def circuit(x):
    qml.RX(x, wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliZ(0))

for i in range(100):
    result = circuit(0.5)
""")
            analyzer = PennyLaneAnalyzer()
            graph = analyzer.analyze_project(tmpdir)
            assert "pennylane" in graph.frameworks_detected
            summary = analyzer.get_summary()
            assert len(summary["quantum_boundaries"]) > 0
            assert len(summary["devices"]) > 0

    def test_detect_shots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("""
import pennylane as qml
dev = qml.device("default.qubit", wires=2, shots=2048)
""")
            analyzer = PennyLaneAnalyzer()
            graph = analyzer.analyze_project(tmpdir)
            summary = analyzer.get_summary()
            assert len(summary["devices"]) > 0
            assert summary["devices"][0]["name"] == "default.qubit"


class TestPytketAnalyzer:
    def test_detect_pytket(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("""
from pytket import Circuit
from pytket.backends import AerBackend

c = Circuit(4)
c.H(0)
c.CX(0, 1)

backend = AerBackend()
result = backend.run_circuit(c, n_shots=1000)
""")
            analyzer = PytketAnalyzer()
            graph = analyzer.analyze_project(tmpdir)
            assert "pytket" in graph.frameworks_detected
            summary = analyzer.get_summary()
            assert len(summary["circuits"]) > 0

    def test_detect_n_shots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("""
from pytket import Circuit
from pytket.backends import AerBackend
backend = AerBackend()
result = backend.run_circuit(c, n_shots=5000)
""")
            analyzer = PytketAnalyzer()
            graph = analyzer.analyze_project(tmpdir)
            summary = analyzer.get_summary()
            assert len(summary["shots_detected"]) > 0
            assert summary["shots_detected"][0]["shots"] == 5000


class TestPulserAnalyzer:
    def test_detect_pulser(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("""
from pulser import Pulse, Sequence, Register
from pulser.waveforms import ConstantWaveform

reg = Register.square(4, spacing=5.0)
seq = Sequence(reg)
seq.declare_channel("ch0", "rydberg_global")
pulse = Pulse.ConstantPulse(1000, 5.0, 0.0, 0)
seq.add(pulse, "ch0")

sim = seq.simulate()
""")
            analyzer = PulserAnalyzer()
            graph = analyzer.analyze_project(tmpdir)
            assert "pulser" in graph.frameworks_detected
            summary = analyzer.get_summary()
            assert len(summary["circuits"]) > 0

    def test_detect_register(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "test.py"
            p.write_text("""
from pulser import Register
reg = Register.from_coordinates([(0,0), (5,0), (10,0)])
""")
            analyzer = PulserAnalyzer()
            graph = analyzer.analyze_project(tmpdir)
            summary = analyzer.get_summary()
            assert "pulser" in summary["frameworks_detected"]


class TestMultiFrameworkAnalyzer:
    def test_multi_framework(self):
        project = Path(__file__).parent.parent / "benchmark" / "maxcut_project"
        if not project.exists():
            pytest.skip("benchmark project not found")
        analyzer = MultiFrameworkAnalyzer()
        graph, summary = analyzer.analyze_project(project)
        assert "qiskit" in summary["frameworks_detected"]
        assert "networkx" in summary["frameworks_detected"]
        assert len(graph.nodes) > 0


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
        rec_ids = [r.transformation_id for r in recs]
        assert "T04" in rec_ids
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
        assert len(plans[0].recommendations) == 1
        assert plans[1].name == "BALANCED"
        assert len(plans[1].recommendations) == 2
        assert plans[2].name == "AGGRESSIVE"
        assert len(plans[2].recommendations) == 2


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
            "quantum_boundaries": [], "circuits": [], "observables": [],
            "loops": [], "shots_detected": [],
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
            plan_data = json.loads(paths["plan"].read_text())
            assert plan_data["schema_version"] == "wq-qcsc-plan-v0.1"


class TestPatchGenerator:
    def test_generate_patches(self):
        project = Path(__file__).parent.parent / "benchmark" / "maxcut_project"
        if not project.exists():
            pytest.skip("benchmark project not found")
        recs = [
            Recommendation(
                transformation_id="T04", transformation_name="Shot Reduction",
                guarantee_class=GuaranteeClass.APPROXIMATE, confidence=Confidence.MEDIUM,
                what_detected="2000 shots", why_expensive="high cost", alternative="reduce",
                estimated_qpu_saving="4x", required_local_compute="none",
                expected_quality_risk="low", validation_experiment="subsample",
                evidence="high shots",
            ),
            Recommendation(
                transformation_id="T09", transformation_name="Graph Preprocessing",
                guarantee_class=GuaranteeClass.EXACT, confidence=Confidence.MEDIUM,
                what_detected="NetworkX", why_expensive="large graphs", alternative="preprocess",
                estimated_qpu_saving="variable", required_local_compute="NetworkX",
                expected_quality_risk="none", validation_experiment="verify",
                evidence="graph structure",
            ),
        ]
        graph = WorkflowGraph()
        summary = {"quantum_boundaries": [{"location": str(project / "maxcut_qaoa.py") + ":1"}]}
        patches = generate_patches(project, recs, graph, summary)
        assert len(patches) == 2
        assert patches[0].recommendation_id == "T04"
        assert patches[1].recommendation_id == "T09"

    def test_write_patch_file(self):
        from qcsc.patch_generator import ProposedPatch
        patches = [ProposedPatch(
            patch_id="p000", recommendation_id="T01",
            file_path="test.py", description="test patch",
            guarantee_class=GuaranteeClass.EXACT,
            original_lines=["old"], proposed_lines=["new"],
        )]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_patch_file(patches, tmpdir)
            assert path.exists()
            assert "test patch" in path.read_text()


class TestRunner:
    def test_dry_run(self):
        project = Path(__file__).parent.parent / "benchmark" / "maxcut_project"
        if not project.exists():
            pytest.skip("benchmark project not found")
        plan_data = {
            "schema_version": "wq-qcsc-plan-v0.1",
            "plans": [{
                "name": "SAFE", "description": "test",
                "recommendations": [{
                    "transformation_id": "T09",
                    "transformation_name": "Graph Preprocessing",
                    "guarantee_class": "exact",
                    "confidence": "medium",
                    "what_detected": "test", "why_expensive": "test",
                    "alternative": "test", "estimated_qpu_saving": "2x",
                    "required_local_compute": "none", "expected_quality_risk": "none",
                    "validation_experiment": "test", "evidence": "test",
                }],
                "projected_reduction": "2x", "guarantee_classes": ["EXACT"],
                "assumptions": [], "uncertainty": "low",
            }],
            "all_recommendations": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            plan_path = Path(tmpdir) / "plan.json"
            plan_path.write_text(json.dumps(plan_data))
            result = run_plan(plan_path, project, output_dir=tmpdir, dry_run=True, auto_confirm=True)
            assert result.success
            assert result.plan_name == "SAFE"


class TestClosedLoop:
    def test_prediction_and_observation(self):
        tracker = ClosedLoopTracker()
        tracker.record_prediction(
            "rec001", "T04", "Shot Reduction", "approximate",
            predicted_saving=4.0, predicted_quality_effect="increased_variance",
            problem_family="QAOA", hardware="ibm_kyiv",
        )
        tracker.record_observation(
            "rec001", actual_saving=3.5, actual_quality_effect="increased_variance",
            success=True,
        )
        report = tracker.generate_report()
        assert report["total_predictions"] == 1
        assert report["total_observations"] == 1
        assert report["successful"] == 1
        assert report["mean_prediction_error"] == 0.5
        assert report["quality_prediction_accuracy"] == 1.0

    def test_save_and_load(self):
        tracker = ClosedLoopTracker()
        tracker.record_prediction("r1", "T01", "Bases", "exact", 2.0)
        tracker.record_observation("r1", 2.0, "none", True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tracker.save(Path(tmpdir) / "data.jsonl")
            assert path.exists()
            tracker2 = ClosedLoopTracker()
            tracker2.load(path)
            assert len(tracker2.records) == 1
            assert "r1" in tracker2.records

    def test_export_training_data(self):
        tracker = ClosedLoopTracker()
        tracker.record_prediction("r1", "T04", "Shots", "approximate", 4.0,
                                   workflow_state={"n_qubits": 8})
        tracker.record_observation("r1", 3.5, "increased_variance", True)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = tracker.export_training_data(Path(tmpdir) / "train.jsonl")
            assert path.exists()
            data = json.loads(path.read_text().strip())
            assert data["schema_version"] == "wq-qcsc-training-v0.1"
            assert data["input"]["transformation_id"] == "T04"
            assert data["output"]["actual_saving"] == 3.5


class TestLearnedPolicy:
    def test_rank_transformations(self):
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
        graph = WorkflowGraph()
        summary = {"frameworks_detected": ["qiskit"]}
        policy = LearnedPolicy()
        plan = policy.rank_transformations(recs, graph, summary, plan_type="balanced")
        assert len(plan.decisions) == 2
        # T01 (EXACT, HIGH confidence, 2x) should rank higher than T04 (APPROXIMATE, MEDIUM, 4x)
        # because EXACT guarantee + HIGH confidence outweighs higher saving factor
        assert plan.decisions[0].transformation_id == "T01"
        assert plan.decisions[0].rank == 1
        assert plan.total_predicted_saving > 1.0

    def test_safe_plan_excludes_approximate(self):
        recs = [
            Recommendation(
                transformation_id="T04", transformation_name="Shot Reduction",
                guarantee_class=GuaranteeClass.APPROXIMATE, confidence=Confidence.HIGH,
                what_detected="test", why_expensive="test", alternative="test",
                estimated_qpu_saving="4x", required_local_compute="none",
                expected_quality_risk="low", validation_experiment="test",
                evidence="test", estimated_saving_factor=4.0,
            ),
        ]
        graph = WorkflowGraph()
        summary = {"frameworks_detected": ["qiskit"]}
        policy = LearnedPolicy()
        plan = policy.rank_transformations(recs, graph, summary, plan_type="safe")
        # Safe plan should not recommend APPROXIMATE
        assert all(not d.recommended for d in plan.decisions)

    def test_policy_explanation(self):
        recs = [
            Recommendation(
                transformation_id="T01", transformation_name="Bases",
                guarantee_class=GuaranteeClass.EXACT, confidence=Confidence.HIGH,
                what_detected="test", why_expensive="test", alternative="test",
                estimated_qpu_saving="2x", required_local_compute="none",
                expected_quality_risk="none", validation_experiment="test",
                evidence="test", estimated_saving_factor=2.0,
            ),
        ]
        graph = WorkflowGraph()
        summary = {"frameworks_detected": ["qiskit"]}
        policy = LearnedPolicy()
        plan = policy.rank_transformations(recs, graph, summary, plan_type="safe")
        assert len(plan.explanation) > 0
        assert "EXACT" in plan.explanation or "exact" in plan.explanation

    def test_save_plan(self):
        recs = [
            Recommendation(
                transformation_id="T01", transformation_name="Bases",
                guarantee_class=GuaranteeClass.EXACT, confidence=Confidence.HIGH,
                what_detected="test", why_expensive="test", alternative="test",
                estimated_qpu_saving="2x", required_local_compute="none",
                expected_quality_risk="none", validation_experiment="test",
                evidence="test", estimated_saving_factor=2.0,
            ),
        ]
        graph = WorkflowGraph()
        summary = {"frameworks_detected": ["qiskit"]}
        policy = LearnedPolicy()
        plan = policy.rank_transformations(recs, graph, summary)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = policy.save_plan(plan, Path(tmpdir) / "policy.json")
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["schema_version"] == "wq-qcsc-policy-v0.1"


def test_cudaq_capability_records_are_advisory():
    records = cudaq_capability_records(cudaq_available=False, gpu_memory_gb=24.0)
    assert records[0]["backend_id"] == "qpp-cpu"
    assert records[0]["available"]
    assert not next(record for record in records if record["backend_id"] == "nvidia")["available"]
