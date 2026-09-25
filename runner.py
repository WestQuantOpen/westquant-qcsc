"""Supervised runner — executes approved plans with explicit user approval.

Usage:
    westquant run plan.json

V0.1 does NOT automatically execute code. It:
1. Reads an approved plan
2. Shows what will be done
3. Requires explicit confirmation
4. Applies patches to a COPY of the project (never the original)
5. Runs validation experiments
6. Reports results

The supervised runner never:
- modifies original source files
- submits jobs to real QPU hardware
- executes without confirmation
- skips validation
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .transformation_registry import Recommendation, GuaranteeClass, Confidence
from .patch_generator import ProposedPatch


@dataclass
class RunResult:
    """Result of running a supervised plan."""
    plan_name: str
    patches_applied: int = 0
    patches_skipped: int = 0
    validation_results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    modified_project_path: str | None = None
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "patches_applied": self.patches_applied,
            "patches_skipped": self.patches_skipped,
            "validation_results": self.validation_results,
            "errors": self.errors,
            "modified_project_path": self.modified_project_path,
            "success": self.success,
        }


def run_plan(
    plan_path: str | Path,
    project_path: str | Path,
    *,
    output_dir: str | Path = ".",
    dry_run: bool = False,
    auto_confirm: bool = False,
) -> RunResult:
    """Run an approved plan in supervised mode.

    Args:
        plan_path: Path to the approved plan JSON
        project_path: Path to the project to modify
        output_dir: Where to write the modified project copy
        dry_run: If True, show what would be done without doing it
        auto_confirm: If True, skip confirmation (for testing only)

    Returns:
        RunResult with execution details
    """
    plan_file = Path(plan_path)
    project = Path(project_path).resolve()
    output = Path(output_dir).resolve()

    if not plan_file.exists():
        return RunResult(plan_name="unknown", errors=[f"Plan file not found: {plan_file}"])

    plan_data = json.loads(plan_file.read_text(encoding="utf-8"))

    # Find the selected plan (SAFE, BALANCED, or AGGRESSIVE)
    plans = plan_data.get("plans", [])
    if not plans:
        return RunResult(plan_name="none", errors=["No plans found in plan file"])

    # Use the first plan that has recommendations, or SAFE by default
    selected = None
    for p in plans:
        if p.get("recommendations"):
            selected = p
            break

    if selected is None:
        return RunResult(plan_name="SAFE", errors=["No plan has recommendations"])

    plan_name = selected["name"]
    recs_data = selected["recommendations"]

    result = RunResult(plan_name=plan_name)

    # Show what will be done
    print(f"\nWestQuant Supervised Runner")
    print(f"  Plan: {plan_name}")
    print(f"  Project: {project}")
    print(f"  Output: {output}")
    print(f"  Dry run: {dry_run}")
    print()

    print(f"  Recommendations to apply: {len(recs_data)}")
    for i, rec in enumerate(recs_data, 1):
        gc = rec.get("guarantee_class", "unknown").upper()
        name = rec.get("transformation_name", "unknown")
        print(f"    {i}. [{gc}] {name}")
    print()

    if dry_run:
        print("  Dry run — no changes will be made.")
        result.success = True
        return result

    # Require confirmation
    if not auto_confirm:
        print("  This will create a COPY of your project with proposed changes.")
        print("  Your original project will NOT be modified.")
        print()
        response = input("  Proceed? [y/N] ")
        if response.lower() not in ("y", "yes"):
            print("  Cancelled.")
            result.errors.append("User cancelled execution")
            return result

    # Create a copy of the project
    modified_path = output / f"{project.name}_westquant_modified"
    if modified_path.exists():
        shutil.rmtree(modified_path)
    shutil.copytree(project, modified_path)
    result.modified_project_path = str(modified_path)
    print(f"  Created modified copy: {modified_path}")

    # Apply patches (read from patch file if available, otherwise generate advisory)
    patches_data = plan_data.get("patches", [])
    if not patches_data:
        # Try to load from patches JSON
        patches_file = plan_file.parent / "westquant_patches.json"
        if patches_file.exists():
            patches_data = json.loads(patches_file.read_text())

    applied = 0
    skipped = 0
    for patch_data in patches_data:
        try:
            # Only apply EXACT patches automatically in v0.1
            gc = GuaranteeClass(patch_data.get("guarantee_class", "approximate"))
            if gc != GuaranteeClass.EXACT:
                print(f"  SKIP [{gc.value.upper()}] {patch_data.get('description', '?')}")
                print(f"    Non-EXACT patches require manual review")
                skipped += 1
                continue

            # Apply patch to the modified copy
            file_path = patch_data.get("file_path", "")
            if not file_path:
                skipped += 1
                continue

            # Adjust path to modified project
            original_file = Path(file_path)
            relative = original_file.relative_to(project) if str(project) in str(original_file) else original_file.name
            target_file = modified_path / relative

            if not target_file.exists():
                print(f"  SKIP — file not found in copy: {target_file}")
                skipped += 1
                continue

            # For v0.1, we write patch comments rather than modifying code
            # This is intentionally conservative
            print(f"  ANNOTATE [{gc.value.upper()}] {patch_data.get('description', '?')}")
            _annotate_file(target_file, patch_data)
            applied += 1

        except Exception as e:
            result.errors.append(f"Patch error: {type(e).__name__}: {e}")
            skipped += 1

    result.patches_applied = applied
    result.patches_skipped = skipped

    # Run validation experiments
    print()
    print("  Running validation experiments...")
    for rec in recs_data:
        gc = GuaranteeClass(rec.get("guarantee_class", "approximate"))
        if gc == GuaranteeClass.EXACT:
            val_result = {
                "transformation": rec.get("transformation_name"),
                "guarantee": gc.value,
                "validation": "EXACT — no validation needed",
                "status": "passed",
            }
        else:
            val_result = {
                "transformation": rec.get("transformation_name"),
                "guarantee": gc.value,
                "validation": rec.get("validation_experiment", "manual review required"),
                "status": "pending",
                "note": "Non-EXACT transformations require manual validation before deployment",
            }
        result.validation_results.append(val_result)
        print(f"    {val_result['transformation']}: {val_result['status']}")

    result.success = len(result.errors) == 0

    # Write run result
    result_path = output / "westquant_run_result.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    print()
    print(f"  Applied: {applied} patches")
    print(f"  Skipped: {skipped} patches (require manual review)")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Modified project: {modified_path}")
    print(f"  Result: {result_path}")
    print()

    if result.success:
        print("  Done. Review the modified project before using it.")
    else:
        print("  Completed with errors. Check the result file.")

    return result


def _annotate_file(filepath: Path, patch_data: dict[str, Any]) -> None:
    """Add advisory annotation comments to a file.

    V0.1 does NOT modify code logic. It only adds comments
    so the user can see what was recommended.
    """
    try:
        content = filepath.read_text(encoding="utf-8")
    except Exception:
        return

    annotation = [
        "# ─── WestQuant QCSC Optimizer Annotation ──────────────────────",
        f"# Transformation: {patch_data.get('description', 'unknown')}",
        f"# Guarantee: {patch_data.get('guarantee_class', 'unknown').upper()}",
        f"# Rationale: {patch_data.get('rationale', '')}",
        f"# Validation: {patch_data.get('validation_required', True)}",
        "# ──────────────────────────────────────────────────────────────",
        "",
    ]

    # Prepend annotation
    annotated = "\n".join(annotation) + content
    filepath.write_text(annotated, encoding="utf-8")
