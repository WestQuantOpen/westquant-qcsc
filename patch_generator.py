"""Patch generator — generates proposed code patches but does NOT apply them.

Usage:
    westquant propose .

Generates a patch file with suggested code changes for each recommendation.
The user must review and apply patches manually.

V0.1 generates advisory patches only. It does NOT:
- apply patches automatically
- execute modified code
- submit modified jobs to hardware
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .transformation_registry import Recommendation, GuaranteeClass
from .flow_ir import WorkflowGraph


@dataclass
class ProposedPatch:
    """A single proposed code change."""
    patch_id: str
    recommendation_id: str
    file_path: str
    description: str
    guarantee_class: GuaranteeClass
    original_lines: list[str] = field(default_factory=list)
    proposed_lines: list[str] = field(default_factory=list)
    line_range: tuple[int, int] = (0, 0)
    rationale: str = ""
    validation_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "recommendation_id": self.recommendation_id,
            "file_path": self.file_path,
            "description": self.description,
            "guarantee_class": self.guarantee_class.value,
            "original_lines": self.original_lines,
            "proposed_lines": self.proposed_lines,
            "line_range": list(self.line_range),
            "rationale": self.rationale,
            "validation_required": self.validation_required,
        }

    def to_diff(self) -> str:
        """Generate a unified diff-style representation."""
        lines = [f"--- {self.file_path} (original)"]
        lines.append(f"+++ {self.file_path} (proposed)")
        lines.append(f"@ {self.line_range[0]}-{self.line_range[1]} @")
        lines.append(f"# {self.description}")
        lines.append(f"# Guarantee: {self.guarantee_class.value.upper()}")
        if self.rationale:
            lines.append(f"# Rationale: {self.rationale}")
        lines.append("")
        for line in self.original_lines:
            lines.append(f"-  {line}")
        for line in self.proposed_lines:
            lines.append(f"+  {line}")
        return "\n".join(lines)


def generate_patches(
    project_path: str | Path,
    recommendations: list[Recommendation],
    graph: WorkflowGraph,
    summary: dict[str, Any],
) -> list[ProposedPatch]:
    """Generate proposed patches from recommendations.

    This is advisory only. Patches are NOT applied automatically.
    """
    project = Path(project_path)
    patches: list[ProposedPatch] = []
    patch_counter = 0

    for rec in recommendations:
        patch = _generate_patch_for_recommendation(
            rec, project, summary, patch_counter
        )
        if patch:
            patches.append(patch)
            patch_counter += 1

    return patches


def _generate_patch_for_recommendation(
    rec: Recommendation,
    project: Path,
    summary: dict[str, Any],
    idx: int,
) -> ProposedPatch | None:
    """Generate a patch for a single recommendation."""
    pid = f"p{idx:03d}"

    if rec.transformation_id == "T01":
        return _patch_redundant_bases(rec, project, summary, pid)
    elif rec.transformation_id == "T03":
        return _patch_commuting_groups(rec, project, summary, pid)
    elif rec.transformation_id == "T04":
        return _patch_shot_reduction(rec, project, summary, pid)
    elif rec.transformation_id == "T06":
        return _patch_simulator_pretraining(rec, project, summary, pid)
    elif rec.transformation_id == "T08":
        return _patch_caching(rec, project, summary, pid)
    elif rec.transformation_id == "T09":
        return _patch_graph_preprocessing(rec, project, summary, pid)
    elif rec.transformation_id == "T13":
        return _patch_simulator_routing(rec, project, summary, pid)
    else:
        return _patch_generic(rec, project, summary, pid)


def _find_qpu_call_file(summary: dict[str, Any], project: Path) -> str | None:
    """Find the file containing QPU calls."""
    for qb in summary.get("quantum_boundaries", []):
        loc = qb.get("location", "")
        if loc:
            filepath = loc.split(":")[0]
            if Path(filepath).exists():
                return filepath
    # Fallback: find first Python file
    for py_file in project.rglob("*.py"):
        parts = py_file.parts
        if not any(p.startswith(".") or p == "__pycache__" for p in parts):
            return str(py_file)
    return None


def _patch_redundant_bases(
    rec: Recommendation, project: Path, summary: dict[str, Any], pid: str
) -> ProposedPatch | None:
    filepath = _find_qpu_call_file(summary, project)
    if not filepath:
        return None
    return ProposedPatch(
        patch_id=pid, recommendation_id=rec.transformation_id,
        file_path=filepath,
        description="Remove redundant measurement basis — all observables are Z-diagonal",
        guarantee_class=rec.guarantee_class,
        original_lines=[
            "for basis in range(2):  # Two measurement bases",
            "    job = sampler.run([tqc], shots=2000)",
        ],
        proposed_lines=[
            "# WestQuant T01: All observables are Z-diagonal — single basis suffices",
            "for basis in range(1):  # Single basis (was 2)",
            "    job = sampler.run([tqc], shots=2000)",
        ],
        line_range=(0, 0),
        rationale=rec.evidence,
        validation_required=False,
    )


def _patch_commuting_groups(
    rec: Recommendation, project: Path, summary: dict[str, Any], pid: str
) -> ProposedPatch | None:
    filepath = _find_qpu_call_file(summary, project)
    if not filepath:
        return None
    return ProposedPatch(
        patch_id=pid, recommendation_id=rec.transformation_id,
        file_path=filepath,
        description="Group commuting Pauli observables for simultaneous measurement",
        guarantee_class=rec.guarantee_class,
        original_lines=[
            "for obs in observables:",
            "    job = estimator.run([tqc], [obs])",
        ],
        proposed_lines=[
            "# WestQuant T03: Group commuting observables",
            "from qiskit.quantum_info import SparsePauliOp",
            "commuting_groups = group_commuting_observables(observables)",
            "for group in commuting_groups:",
            "    job = estimator.run([tqc], group)  # Measure group simultaneously",
        ],
        line_range=(0, 0),
        rationale=rec.evidence,
        validation_required=False,
    )


def _patch_shot_reduction(
    rec: Recommendation, project: Path, summary: dict[str, Any], pid: str
) -> ProposedPatch | None:
    filepath = _find_qpu_call_file(summary, project)
    if not filepath:
        return None
    shots = 2000
    for s in summary.get("shots_detected", []):
        if isinstance(s.get("shots"), int):
            shots = s["shots"]
            break
    reduced_shots = max(shots // 4, 256)
    return ProposedPatch(
        patch_id=pid, recommendation_id=rec.transformation_id,
        file_path=filepath,
        description=f"Reduce shots from {shots} to {reduced_shots} (validate first)",
        guarantee_class=rec.guarantee_class,
        original_lines=[
            f"job = sampler.run([tqc], shots={shots})",
        ],
        proposed_lines=[
            f"# WestQuant T04: Reduced shots (was {shots}) — validate with subsampling first",
            f"# Validation: subsample existing {shots}-shot bitstrings at {reduced_shots} shots,",
            f"# compare task metric within 95% CI before deploying",
            f"job = sampler.run([tqc], shots={reduced_shots})",
        ],
        line_range=(0, 0),
        rationale=rec.evidence,
        validation_required=True,
    )


def _patch_simulator_pretraining(
    rec: Recommendation, project: Path, summary: dict[str, Any], pid: str
) -> ProposedPatch | None:
    filepath = _find_qpu_call_file(summary, project)
    if not filepath:
        return None
    return ProposedPatch(
        patch_id=pid, recommendation_id=rec.transformation_id,
        file_path=filepath,
        description="Pretrain variational parameters on statevector simulator before QPU",
        guarantee_class=rec.guarantee_class,
        original_lines=[
            "# Optimize parameters on QPU directly",
            "optimizer.minimize(cost_fn, initial_params)",
        ],
        proposed_lines=[
            "# WestQuant T06: Pretrain on simulator first",
            "from qiskit_aer import AerSimulator",
            "sim = AerSimulator(method='statevector')",
            "# Pretrain on simulator",
            "optimizer.minimize(cost_fn_sim, initial_params)",
            "# Fine-tune on QPU with pre-trained parameters",
            "optimizer.minimize(cost_fn, pretrained_params)",
        ],
        line_range=(0, 0),
        rationale=rec.evidence,
        validation_required=True,
    )


def _patch_caching(
    rec: Recommendation, project: Path, summary: dict[str, Any], pid: str
) -> ProposedPatch | None:
    filepath = _find_qpu_call_file(summary, project)
    if not filepath:
        return None
    return ProposedPatch(
        patch_id=pid, recommendation_id=rec.transformation_id,
        file_path=filepath,
        description="Cache results for identical circuit+parameter combinations",
        guarantee_class=rec.guarantee_class,
        original_lines=[
            "for i, graph in enumerate(graphs):",
            "    job = sampler.run([tqc], shots=2000)",
            "    result = job.result()",
        ],
        proposed_lines=[
            "# WestQuant T08: Cache identical circuit submissions",
            "cache = {}",
            "for i, graph in enumerate(graphs):",
            "    circuit_hash = hash((str(tqc), tuple(params)))",
            "    if circuit_hash in cache:",
            "        result = cache[circuit_hash]  # Reuse cached result",
            "    else:",
            "        job = sampler.run([tqc], shots=2000)",
            "        result = job.result()",
            "        cache[circuit_hash] = result",
        ],
        line_range=(0, 0),
        rationale=rec.evidence,
        validation_required=False,
    )


def _patch_graph_preprocessing(
    rec: Recommendation, project: Path, summary: dict[str, Any], pid: str
) -> ProposedPatch | None:
    filepath = _find_qpu_call_file(summary, project)
    if not filepath:
        return None
    return ProposedPatch(
        patch_id=pid, recommendation_id=rec.transformation_id,
        file_path=filepath,
        description="Add classical graph preprocessing before QAOA",
        guarantee_class=rec.guarantee_class,
        original_lines=[
            "graphs = []",
            "for i in range(10000):",
            "    g = nx.erdos_renyi_graph(8, 0.5, seed=i)",
            "    graphs.append(g)",
        ],
        proposed_lines=[
            "# WestQuant T09: Classical graph preprocessing",
            "graphs = []",
            "for i in range(10000):",
            "    g = nx.erdos_renyi_graph(8, 0.5, seed=i)",
            "    # Remove isolated nodes (exact — preserves MaxCut structure)",
            "    g.remove_nodes_from(list(nx.isolates(g)))",
            "    # Split disconnected components (exact — independent subproblems)",
            "    components = list(nx.connected_components(g))",
            "    if len(components) > 1:",
            "        for comp in components:",
            "            subgraph = g.subgraph(comp).copy()",
            "            if subgraph.number_of_edges() > 0:",
            "                graphs.append(subgraph)",
            "    else:",
            "        graphs.append(g)",
        ],
        line_range=(0, 0),
        rationale=rec.evidence,
        validation_required=False,
    )


def _patch_simulator_routing(
    rec: Recommendation, project: Path, summary: dict[str, Any], pid: str
) -> ProposedPatch | None:
    filepath = _find_qpu_call_file(summary, project)
    if not filepath:
        return None
    return ProposedPatch(
        patch_id=pid, recommendation_id=rec.transformation_id,
        file_path=filepath,
        description="Replace QPU execution with local statevector simulator",
        guarantee_class=rec.guarantee_class,
        original_lines=[
            'backend = "ibm_kyiv"',
            "sampler = Sampler(mode=backend)",
        ],
        proposed_lines=[
            "# WestQuant T13: Use local simulator instead of QPU",
            "# Circuit width is within statevector simulator feasibility",
            "from qiskit_aer import AerSimulator",
            "backend = AerSimulator(method='statevector')",
            "sampler = Sampler(mode=backend)  # Exact results, no QPU cost",
        ],
        line_range=(0, 0),
        rationale=rec.evidence,
        validation_required=False,
    )


def _patch_generic(
    rec: Recommendation, project: Path, summary: dict[str, Any], pid: str
) -> ProposedPatch | None:
    filepath = _find_qpu_call_file(summary, project)
    if not filepath:
        return None
    return ProposedPatch(
        patch_id=pid, recommendation_id=rec.transformation_id,
        file_path=filepath,
        description=f"Apply {rec.transformation_name}: {rec.alternative}",
        guarantee_class=rec.guarantee_class,
        original_lines=["# Original code — see recommendation for details"],
        proposed_lines=[
            f"# WestQuant {rec.transformation_id}: {rec.transformation_name}",
            f"# {rec.alternative}",
            f"# Validation: {rec.validation_experiment}",
            "# TODO: Apply transformation based on recommendation details",
        ],
        line_range=(0, 0),
        rationale=rec.evidence,
        validation_required=rec.guarantee_class != GuaranteeClass.EXACT,
    )


def write_patch_file(
    patches: list[ProposedPatch],
    output_path: str | Path,
) -> Path:
    """Write patches to a file for user review."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# WestQuant QCSC Optimizer — Proposed Patches",
        "# Advisory only. Review and apply manually.",
        "# Do NOT apply patches without understanding each change.",
        "",
    ]

    for patch in patches:
        lines.append(patch.to_diff())
        lines.append("")
        lines.append("=" * 60)
        lines.append("")

    patch_path = out / "westquant_patches.diff"
    patch_path.write_text("\n".join(lines), encoding="utf-8")

    # Also write machine-readable JSON
    json_path = out / "westquant_patches.json"
    json_path.write_text(
        json.dumps([p.to_dict() for p in patches], indent=2),
        encoding="utf-8",
    )

    return patch_path
