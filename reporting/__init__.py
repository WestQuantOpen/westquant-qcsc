"""Report generator — produces HTML, Markdown, and JSON outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..flow_ir import WorkflowGraph
from ..transformation_registry import Recommendation, GuaranteeClass, Confidence
from ..planner import Plan


def _guarantee_emoji(gc: GuaranteeClass) -> str:
    return {
        GuaranteeClass.EXACT: "[EXACT]",
        GuaranteeClass.EQUIVALENT: "[EQUIVALENT]",
        GuaranteeClass.APPROXIMATE: "[APPROXIMATE]",
        GuaranteeClass.EXPERIMENTAL: "[EXPERIMENTAL]",
    }.get(gc, "[UNKNOWN]")


def _confidence_label(c: Confidence) -> str:
    return c.value.upper()


def generate_markdown_report(
    summary: dict[str, Any],
    graph: WorkflowGraph,
    recommendations: list[Recommendation],
    plans: list[Plan],
    baseline_cost: dict[str, Any],
) -> str:
    """Generate the Markdown audit report."""
    lines: list[str] = []
    lines.append("# WestQuant QCSC Optimizer — Audit Report\n")
    lines.append("> Use the QPU only where the QPU is actually needed.\n")

    # Executive summary
    lines.append("## Executive Summary\n")
    n_recs = len(recommendations)
    n_exact = sum(1 for r in recommendations if r.guarantee_class == GuaranteeClass.EXACT)
    lines.append(f"- Frameworks detected: {', '.join(summary.get('frameworks_detected', ['none']))}")
    lines.append(f"- Quantum boundaries detected: {summary.get('n_quantum_boundaries', 0)}")
    lines.append(f"- Circuits detected: {summary.get('n_circuits', 0)}")
    lines.append(f"- Observables detected: {summary.get('n_observables', 0)}")
    lines.append(f"- Loops detected: {summary.get('n_loops', 0)}")
    lines.append(f"- QPU-saving opportunities: {n_recs} ({n_exact} EXACT)\n")

    # Frameworks
    lines.append("## Frameworks Detected\n")
    for fw in summary.get("frameworks_detected", []):
        lines.append(f"- {fw}")
    lines.append("")

    # Workflow map
    lines.append("## Workflow Map\n")
    lines.append(f"- Total nodes: {len(graph.nodes)}")
    lines.append(f"- Quantum nodes: {len(graph.quantum_nodes())}")
    lines.append(f"- Classical nodes: {len(graph.classical_nodes())}")
    lines.append(f"- Edges: {len(graph.edges)}\n")

    # Quantum boundaries
    lines.append("## Quantum Boundaries\n")
    for b in summary.get("quantum_boundaries", []):
        btype = "QPU" if not b.get("is_simulator") else "Simulator"
        lines.append(f"- [{btype}] {b.get('call', 'unknown')} at {b.get('location', '?')}")
        if b.get("shots"):
            lines.append(f"  - Shots: {b['shots']}")
    lines.append("")

    # Baseline cost
    lines.append("## Baseline Quantum Workload\n")
    for k, v in baseline_cost.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    # Opportunities
    lines.append("## QPU-Saving Opportunities\n")
    if not recommendations:
        lines.append("No QPU-saving opportunities detected.\n")
    else:
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"### Opportunity {i}: {rec.transformation_name}\n")
            lines.append(f"**Guarantee:** {_guarantee_emoji(rec.guarantee_class)}")
            lines.append(f"  **Confidence:** {_confidence_label(rec.confidence)}\n")
            lines.append(f"**What was detected:** {rec.what_detected}")
            lines.append(f"**Why it's expensive:** {rec.why_expensive}")
            lines.append(f"**Alternative:** {rec.alternative}")
            lines.append(f"**Projected QPU saving:** {rec.estimated_qpu_saving}")
            lines.append(f"**Required local compute:** {rec.required_local_compute}")
            lines.append(f"**Expected quality risk:** {rec.expected_quality_risk}")
            lines.append(f"**Validation:** {rec.validation_experiment}")
            lines.append(f"**Evidence:** {rec.evidence}\n")

    # Plans
    lines.append("## Recommended Plans\n")
    for plan in plans:
        lines.append(f"### {plan.name}")
        lines.append(f"  {plan.projected_reduction}\n")
        lines.append(f"  {plan.description}")
        lines.append(f"  Transformations: {len(plan.recommendations)}")
        if plan.assumptions:
            lines.append(f"  Assumptions:")
            for a in plan.assumptions:
                lines.append(f"    - {a}")
        lines.append(f"  Uncertainty: {plan.uncertainty}\n")

    # Uncertainty
    lines.append("## Uncertainty and Assumptions\n")
    lines.append("- This is an advisory-only report. No code was modified.")
    lines.append("- Cost estimates are based on static analysis and may differ from runtime values.")
    lines.append("- Composition of transformations does not simply multiply savings.")
    lines.append("- Experimental transformations require hardware validation.\n")

    return "\n".join(lines)


def generate_html_report(md_content: str) -> str:
    """Wrap Markdown content in a minimal HTML template."""
    # Simple Markdown-to-HTML conversion (no dependencies)
    html_lines: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>WestQuant QCSC Optimizer — Audit Report</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; color: #333; }",
        "h1 { color: #1a1a2e; border-bottom: 2px solid #16213e; padding-bottom: 0.3em; }",
        "h2 { color: #16213e; margin-top: 1.5em; }",
        "h3 { color: #0f3460; }",
        "code { background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }",
        "blockquote { border-left: 4px solid #16213e; margin: 1em 0; padding: 0.5em 1em; background: #f8f8f8; }",
        "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "th, td { border: 1px solid #ddd; padding: 0.5em; text-align: left; }",
        "th { background: #16213e; color: white; }",
        ".exact { color: #2d6a4f; font-weight: bold; }",
        ".approximate { color: #e76f51; font-weight: bold; }",
        ".experimental { color: #9d4edd; font-weight: bold; }",
        "</style>",
        "</head>",
        "<body>",
    ]

    # Very basic markdown to HTML
    in_list = False
    for line in md_content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("> "):
            html_lines.append(f"<blockquote>{stripped[2:]}</blockquote>")
        elif stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{stripped[2:]}</li>")
        elif stripped == "":
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append("")
        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            # Bold conversion
            import re
            html_line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', stripped)
            html_lines.append(f"<p>{html_line}</p>")

    if in_list:
        html_lines.append("</ul>")

    html_lines.extend(["</body>", "</html>"])
    return "\n".join(html_lines)


def generate_plan_json(plans: list[Plan], recommendations: list[Recommendation]) -> dict[str, Any]:
    """Generate machine-readable plan."""
    return {
        "schema_version": "wq-qcsc-plan-v0.1",
        "plans": [p.to_dict() for p in plans],
        "all_recommendations": [r.to_dict() for r in recommendations],
    }


def generate_workflow_json(graph: WorkflowGraph, summary: dict[str, Any]) -> dict[str, Any]:
    """Generate machine-readable workflow."""
    return {
        "schema_version": "wq-flow-ir-v0.1",
        "graph": graph.to_dict(),
        "summary": summary,
    }


def write_reports(
    output_dir: str | Path,
    summary: dict[str, Any],
    graph: WorkflowGraph,
    recommendations: list[Recommendation],
    plans: list[Plan],
    baseline_cost: dict[str, Any],
) -> dict[str, Path]:
    """Write all report files to output directory."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Markdown
    md = generate_markdown_report(summary, graph, recommendations, plans, baseline_cost)
    md_path = out / "westquant_report.md"
    md_path.write_text(md, encoding="utf-8")

    # HTML
    html = generate_html_report(md)
    html_path = out / "westquant_report.html"
    html_path.write_text(html, encoding="utf-8")

    # Plan JSON
    plan_data = generate_plan_json(plans, recommendations)
    plan_path = out / "westquant_plan.json"
    plan_path.write_text(json.dumps(plan_data, indent=2), encoding="utf-8")

    # Workflow JSON
    workflow_data = generate_workflow_json(graph, summary)
    workflow_path = out / "westquant_workflow.json"
    workflow_path.write_text(json.dumps(workflow_data, indent=2), encoding="utf-8")

    return {
        "markdown": md_path,
        "html": html_path,
        "plan": plan_path,
        "workflow": workflow_path,
    }
