"""WestQuant QCSC Optimizer — CLI entry point.

Usage:
    westquant audit .                  # audit current directory
    westquant audit ./project          # audit specific project
    westquant audit ./project -o out/  # specify output directory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer.qiskit_analyzer import QiskitAnalyzer
from .detectors import run_all_detectors
from .planner import build_plans
from .reporting import write_reports
from .cost_model import estimate_baseline_cost


def cmd_audit(args: argparse.Namespace) -> int:
    project_path = Path(args.project).resolve()
    output_path = Path(args.output).resolve()

    if not project_path.exists():
        print(f"Error: project path does not exist: {project_path}", file=sys.stderr)
        return 1

    print(f"WestQuant QCSC Optimizer")
    print(f"  Auditing: {project_path}")
    print(f"  Output:   {output_path}")
    print()

    # Phase 1: Analyze
    print("  [1/4] Analyzing source code...")
    analyzer = QiskitAnalyzer()
    graph = analyzer.analyze_project(project_path)
    summary = analyzer.get_summary()

    print(f"        Frameworks: {', '.join(summary.get('frameworks_detected', [])) or 'none'}")
    print(f"        Nodes: {len(graph.nodes)}")
    print(f"        Quantum boundaries: {len(summary.get('quantum_boundaries', []))}")
    print(f"        Circuits: {len(summary.get('circuits', []))}")
    print(f"        Observables: {len(summary.get('observables', []))}")
    print(f"        Loops: {len(summary.get('loops', []))}")
    print()

    # Phase 2: Estimate baseline cost
    print("  [2/4] Estimating baseline QPU cost...")
    baseline = estimate_baseline_cost(summary, graph)
    print(f"        Estimated QPU jobs: {baseline['n_qpu_jobs']}")
    print(f"        Estimated total shots: {baseline['total_shots']}")
    print(f"        Max qubits: {baseline['max_qubits']}")
    print()

    # Phase 3: Detect QPU-saving opportunities
    print("  [3/4] Detecting QPU-saving opportunities...")
    recommendations = run_all_detectors(summary, graph)
    print(f"        Found {len(recommendations)} opportunities:")
    for rec in recommendations:
        print(f"          [{rec.guarantee_class.value.upper()}] {rec.transformation_name}: {rec.estimated_qpu_saving}")
    print()

    # Phase 4: Build plans and generate reports
    print("  [4/4] Building plans and generating reports...")
    plans = build_plans(recommendations)
    for plan in plans:
        print(f"        {plan.name:12s} {plan.projected_reduction}")
    print()

    # Write reports
    paths = write_reports(output_path, summary, graph, recommendations, plans, baseline)

    print("  Reports generated:")
    for name, path in paths.items():
        print(f"    {name:12s} {path}")
    print()
    print("  Done. Advisory only — no code was modified.")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="westquant",
        description="WestQuant QCSC Optimizer — Semantic QPU Minimization",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit a quantum project for QPU-saving opportunities")
    audit_parser.add_argument("project", nargs="?", default=".", help="Project directory to audit (default: current directory)")
    audit_parser.add_argument("-o", "--output", default=".", help="Output directory for reports (default: current directory)")
    audit_parser.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
