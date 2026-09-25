#!/usr/bin/env python3
"""
package_submission.py  —  Build the final submission zip from the project.

Run from the amazon-ml-challenge/ directory:
    python3 package_submission.py --team-name "YourTeamName"

This script:
  1. Validates output/matching_results.tsv and output/candidate_pairs.tsv
  2. Assembles the required submission structure:
       <team_name>_submission.zip
       ├── output/
       │   ├── matching_results.tsv
       │   └── candidate_pairs.tsv
       ├── code/
       │   └── business_entity_resolution/
       │       ├── src/
       │       ├── utils/
       │       ├── notebooks/
       │       ├── README.md
       │       └── requirements.txt
       └── Documentation_template.md
  3. Zips everything up and reports the final zip path.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# Resolve project root (same dir as this script)
PROJECT_ROOT = Path(__file__).resolve().parent


def validate_outputs(test_dir: Path) -> bool:
    """Run the official validator and return True if PASS."""
    matching = PROJECT_ROOT / "output" / "matching_results.tsv"
    candidate = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
    validator = PROJECT_ROOT / "utils" / "validate_submission.py"

    if not validator.exists():
        print(f"[WARN] Validator not found at {validator}. Skipping validation.")
        return True

    if not matching.exists():
        print(f"[FAIL] output/matching_results.tsv not found. Run the pipeline first.")
        return False

    cmd = [
        sys.executable, str(validator),
        "--matching", str(matching),
        "--candidate", str(candidate),
        "--test-dir", str(test_dir),
    ]
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def build_zip(team_name: str, test_dir: Path, skip_validate: bool = False) -> Path:
    """Build the submission zip and return its path."""
    zip_name = f"{team_name}_submission.zip"
    zip_path = PROJECT_ROOT / zip_name

    # ── Validate ──────────────────────────────────────────────────────────
    if not skip_validate:
        print("\n" + "=" * 60)
        print("STEP 1: Validating output files...")
        print("=" * 60)
        ok = validate_outputs(test_dir)
        if not ok:
            print("\n[ABORT] Fix validation errors before packaging.")
            sys.exit(1)
        print("[OK] Validation passed.\n")
    else:
        print("[SKIP] Validation skipped (--skip-validate).")

    # ── Assemble in a temp directory ──────────────────────────────────────
    print("STEP 2: Assembling submission structure...")
    with tempfile.TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)

        # output/
        out_dir = base / "output"
        out_dir.mkdir()
        shutil.copy(PROJECT_ROOT / "output" / "matching_results.tsv", out_dir)
        cand_src = PROJECT_ROOT / "output" / "candidate_pairs.tsv"
        if cand_src.exists():
            shutil.copy(cand_src, out_dir)
        else:
            print("[WARN] output/candidate_pairs.tsv not found — omitting from zip.")

        # code/business_entity_resolution/
        code_root = base / "code" / "business_entity_resolution"
        code_root.mkdir(parents=True)

        # src/
        shutil.copytree(
            PROJECT_ROOT / "src" / "business_entity_resolution",
            code_root / "src" / "business_entity_resolution",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        # utils/
        shutil.copytree(
            PROJECT_ROOT / "utils",
            code_root / "utils",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

        # notebooks/
        nb_src = PROJECT_ROOT / "notebooks"
        if nb_src.exists():
            shutil.copytree(
                nb_src,
                code_root / "notebooks",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

        # README.md + requirements.txt
        shutil.copy(PROJECT_ROOT / "README.md", code_root / "README.md")
        shutil.copy(PROJECT_ROOT / "requirements.txt", code_root / "requirements.txt")

        # Documentation_template.md (top-level of zip)
        shutil.copy(
            PROJECT_ROOT / "Documentation_template.md",
            base / "Documentation_template.md",
        )

        # ── Create zip ────────────────────────────────────────────────────
        print(f"STEP 3: Creating {zip_name}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(base.rglob("*")):
                if file.is_file():
                    arcname = file.relative_to(base)
                    zf.write(file, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n{'=' * 60}")
    print(f"✅ Submission zip created: {zip_path}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"{'=' * 60}")

    # Print contents
    print("\nZip contents:")
    with zipfile.ZipFile(zip_path) as zf:
        for name in sorted(zf.namelist()):
            info = zf.getinfo(name)
            print(f"  {name:<70}  {info.file_size / 1024:>8.1f} KB")

    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Package the submission zip.")
    parser.add_argument(
        "--team-name", "-t",
        required=True,
        help="Team name used as the zip file prefix (e.g. 'TeamAlpha')",
    )
    parser.add_argument(
        "--test-dir",
        default="dataset/test",
        help="Path to test dataset directory for validation (default: dataset/test)",
    )
    parser.add_argument(
        "--skip-validate",
        action="store_true",
        help="Skip running the official validator (not recommended)",
    )
    args = parser.parse_args()

    build_zip(
        team_name=args.team_name,
        test_dir=PROJECT_ROOT / args.test_dir,
        skip_validate=args.skip_validate,
    )


if __name__ == "__main__":
    main()
