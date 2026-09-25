"""WestQuant QCSC Optimizer — CLI entry point.

Commands:
    westquant audit .                  # audit current directory
    westquant audit ./project          # audit specific project
    westquant propose .                # generate proposed patches (advisory)
    westquant run plan.json             # run approved plan (supervised)
    westquant policy .                  # generate learned policy plan
    westquant closed-loop               # manage closed-loop measurement data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer.multi_framework import MultiFrameworkAnalyzer
from .detectors import run_all_detectors
from .planner import build_plans
from .reporting import write_reports
from .cost_model import estimate_baseline_cost
from .patch_generator import generate_patches, write_patch_file
from .runner import run_plan
from .closed_loop import ClosedLoopTracker
from .learned_policy import LearnedPolicy
from .resource_model import detect_local_resources, LocalResourceModel


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

    # Phase 1: Analyze with all framework analyzers
    print("  [1/4] Analyzing source code...")
    analyzer = MultiFrameworkAnalyzer()
    graph, summary = analyzer.analyze_project(project_path)

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

    paths = write_reports(output_path, summary, graph, recommendations, plans, baseline)

    print("  Reports generated:")
    for name, path in paths.items():
        print(f"    {name:12s} {path}")
    print()
    print("  Done. Advisory only — no code was modified.")

    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    project_path = Path(args.project).resolve()
    output_path = Path(args.output).resolve()

    if not project_path.exists():
        print(f"Error: project path does not exist: {project_path}", file=sys.stderr)
        return 1

    print(f"WestQuant QCSC Optimizer — Patch Generator")
    print(f"  Project: {project_path}")
    print(f"  Output:  {output_path}")
    print()

    # Analyze
    print("  [1/3] Analyzing source code...")
    analyzer = MultiFrameworkAnalyzer()
    graph, summary = analyzer.analyze_project(project_path)
    print(f"        Frameworks: {', '.join(summary.get('frameworks_detected', [])) or 'none'}")
    print()

    # Detect opportunities
    print("  [2/3] Detecting QPU-saving opportunities...")
    recommendations = run_all_detectors(summary, graph)
    print(f"        Found {len(recommendations)} opportunities")
    print()

    # Generate patches
    print("  [3/3] Generating proposed patches...")
    patches = generate_patches(project_path, recommendations, graph, summary)
    print(f"        Generated {len(patches)} patches:")
    for patch in patches:
        print(f"          [{patch.guarantee_class.value.upper()}] {patch.description}")
    print()

    # Write patch files
    patch_path = write_patch_file(patches, output_path)
    print(f"  Patch file: {patch_path}")
    print(f"  JSON file:  {output_path / 'westquant_patches.json'}")
    print()
    print("  Advisory only — review patches before applying.")
    print("  Use 'westquant run plan.json' to apply in supervised mode.")

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    plan_path = Path(args.plan).resolve()
    project_path = Path(args.project).resolve() if args.project else Path(".").resolve()
    output_path = Path(args.output).resolve() if args.output else Path(".").resolve()

    if not plan_path.exists():
        print(f"Error: plan file not found: {plan_path}", file=sys.stderr)
        return 1

    result = run_plan(
        plan_path, project_path,
        output_dir=output_path,
        dry_run=args.dry_run,
        auto_confirm=args.yes,
    )

    return 0 if result.success else 1


def cmd_policy(args: argparse.Namespace) -> int:
    project_path = Path(args.project).resolve()
    output_path = Path(args.output).resolve()

    if not project_path.exists():
        print(f"Error: project path does not exist: {project_path}", file=sys.stderr)
        return 1

    print(f"WestQuant QCSC Optimizer — Learned Policy")
    print(f"  Project: {project_path}")
    print(f"  Output:  {output_path}")
    print()

    # Analyze
    print("  [1/3] Analyzing source code...")
    analyzer = MultiFrameworkAnalyzer()
    graph, summary = analyzer.analyze_project(project_path)
    print(f"        Frameworks: {', '.join(summary.get('frameworks_detected', [])) or 'none'}")
    print()

    # Detect opportunities
    print("  [2/3] Detecting QPU-saving opportunities...")
    recommendations = run_all_detectors(summary, graph)
    print(f"        Found {len(recommendations)} opportunities")
    print()

    # Generate policy plan
    print("  [3/3] Generating learned policy plan...")
    training_data = Path(args.training_data) if args.training_data else None
    policy = LearnedPolicy(training_data_path=training_data if training_data else None)

    # Detect local resources if requested
    local = None
    if args.detect_resources:
        print("        Detecting local resources...")
        local = detect_local_resources()
        print(f"        CPU: {local.cpu_cores} cores, RAM: {local.ram_gb:.0f} GB")

    plan = policy.rank_transformations(
        recommendations, graph, summary, local_resources=local,
        plan_type=args.plan_type,
    )

    # Save plan
    plan_path = policy.save_plan(plan, output_path / "westquant_policy.json")
    print(f"        Plan type: {plan.plan_type}")
    print(f"        Total predicted saving: {plan.total_predicted_saving}x")
    print(f"        Decisions: {len(plan.decisions)}")
    for d in plan.decisions:
        status = "RECOMMENDED" if d.recommended else "skipped"
        print(f"          [{d.rank}] [{d.guarantee_class.upper()}] {d.transformation_name}: {d.predicted_saving}x ({status})")
    print()
    print(f"  Policy plan: {plan_path}")
    print(f"  Explanation: {plan.explanation}")

    return 0


def cmd_closed_loop(args: argparse.Namespace) -> int:
    if args.subcommand == "report":
        tracker = ClosedLoopTracker()
        if args.data:
            tracker.load(args.data)
        report = tracker.generate_report()
        print(json.dumps(report, indent=2))
        return 0

    elif args.subcommand == "export":
        tracker = ClosedLoopTracker()
        if not args.data:
            print("Error: --data required for export", file=sys.stderr)
            return 1
        tracker.load(args.data)
        out = tracker.export_training_data(args.output or "qcsc_training_data.jsonl")
        print(f"Exported {len(tracker.records)} records to {out}")
        return 0

    else:
        print(f"Unknown closed-loop subcommand: {args.subcommand}", file=sys.stderr)
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="westquant",
        description="WestQuant QCSC Optimizer — Semantic QPU Minimization",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # audit
    audit_parser = subparsers.add_parser("audit", help="Audit a quantum project for QPU-saving opportunities")
    audit_parser.add_argument("project", nargs="?", default=".", help="Project directory to audit")
    audit_parser.add_argument("-o", "--output", default=".", help="Output directory for reports")
    audit_parser.set_defaults(func=cmd_audit)

    # propose
    propose_parser = subparsers.add_parser("propose", help="Generate proposed code patches (advisory)")
    propose_parser.add_argument("project", nargs="?", default=".", help="Project directory")
    propose_parser.add_argument("-o", "--output", default=".", help="Output directory for patches")
    propose_parser.set_defaults(func=cmd_propose)

    # run
    run_parser = subparsers.add_parser("run", help="Run an approved plan in supervised mode")
    run_parser.add_argument("plan", help="Path to approved plan JSON")
    run_parser.add_argument("-p", "--project", default=".", help="Project directory")
    run_parser.add_argument("-o", "--output", default=".", help="Output directory")
    run_parser.add_argument("--dry-run", action="store_true", help="Show what would be done without doing it")
    run_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation (for testing)")
    run_parser.set_defaults(func=cmd_run)

    # policy
    policy_parser = subparsers.add_parser("policy", help="Generate learned policy plan")
    policy_parser.add_argument("project", nargs="?", default=".", help="Project directory")
    policy_parser.add_argument("-o", "--output", default=".", help="Output directory")
    policy_parser.add_argument("--training-data", default=None, help="Path to closed-loop training data (JSONL)")
    policy_parser.add_argument("--detect-resources", action="store_true", help="Detect local compute resources")
    policy_parser.add_argument("--plan-type", choices=["safe", "balanced", "aggressive"], default="balanced")
    policy_parser.set_defaults(func=cmd_policy)

    # closed-loop
    cl_parser = subparsers.add_parser("closed-loop", help="Manage closed-loop measurement data")
    cl_sub = cl_parser.add_subparsers(dest="subcommand", required=True)
    cl_report = cl_sub.add_parser("report", help="Generate closed-loop report")
    cl_report.add_argument("--data", default=None, help="Path to closed-loop data (JSONL)")
    cl_export = cl_sub.add_parser("export", help="Export training data")
    cl_export.add_argument("--data", required=True, help="Path to closed-loop data (JSONL)")
    cl_export.add_argument("--output", default=None, help="Output path for training data")
    cl_parser.set_defaults(func=cmd_closed_loop)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
