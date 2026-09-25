#!/usr/bin/env python3
"""
Build submission zip with the required structure:

    APK_submission.zip
    ├── output/
    │   ├── matching_results.tsv
    │   └── candidate_pairs.tsv
    ├── code/
    │   └── business_entity_resolution/
    │       ├── src/
    │       ├── README.md
    │       └── requirements.txt
    └── Documentation_template.md

Run from project root:
    python package.py --team-name APK
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def validate(test_dir: Path) -> bool:
    validator = ROOT / "utils" / "validate_submission.py"
    if not validator.exists():
        print("[warn] validator not found, skipping")
        return True
    r = subprocess.run([
        sys.executable, str(validator),
        "--matching",  str(ROOT / "output" / "matching_results.tsv"),
        "--candidate", str(ROOT / "output" / "candidate_pairs.tsv"),
        "--test-dir",  str(test_dir),
    ])
    return r.returncode == 0


def build(team: str, test_dir: Path, skip_validate: bool = False):
    zip_path = ROOT / f"{team}_submission.zip"

    if not skip_validate:
        print("validating output files...")
        if not validate(test_dir):
            sys.exit("validation failed — fix errors before packaging")
        print("validation passed\n")

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)

        # output/
        out = base / "output"
        out.mkdir()
        shutil.copy(ROOT / "output" / "matching_results.tsv", out)
        cand = ROOT / "output" / "candidate_pairs.tsv"
        if cand.exists():
            shutil.copy(cand, out)
        else:
            print("[warn] candidate_pairs.tsv not found — omitting")

        # code/business_entity_resolution/
        code_root = base / "code" / "business_entity_resolution"
        code_root.mkdir(parents=True)

        # src/ — copy the whole package
        shutil.copytree(
            ROOT / "src",
            code_root / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        shutil.copy(ROOT / "README.md",       code_root / "README.md")
        shutil.copy(ROOT / "requirements.txt", code_root / "requirements.txt")

        # Documentation_template.md at zip root
        shutil.copy(ROOT / "Documentation_template.md", base / "Documentation_template.md")

        # zip it
        print(f"creating {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(base.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(base))

    size = zip_path.stat().st_size / 1024 / 1024
    print(f"\n✓ {zip_path}  ({size:.1f} MB)")

    with zipfile.ZipFile(zip_path) as zf:
        for n in sorted(zf.namelist()):
            kb = zf.getinfo(n).file_size / 1024
            print(f"  {n:<70}  {kb:>8.1f} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-name", "-t", default="APK")
    ap.add_argument("--test-dir",  default="dataset/test")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()
    build(args.team_name, ROOT / args.test_dir, args.skip_validate)
