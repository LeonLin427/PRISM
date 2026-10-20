"""Portable release entry point. Run with Python from any working directory."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

def run(script, *args):
    command = [sys.executable, str(ROOT / script), *args]
    print("+", script, *args, flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    subprocess.run(command, cwd=ROOT, env=env, check=True)

def validate():
    run("verify_release.py")
    run("test_match_sdks.py")
    run("test_build_remediation.py")
    run("test_training_reproducibility.py")
    # Import after argument parsing so --help needs no scientific packages.
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from train_gnn import load_data, build_subgraph
    from build_graph import build_graph
    from eval_ood import run_verify, CONTROL_SEED
    from match_sdks import load_signatures
    (df, folders, static, diff, sdks, apis, perms, y, cache, diff_sdks) = load_data(True)
    full = build_graph()
    _, edges = build_subgraph(folders, cache, sdks, apis, perms, 1, diff_sdks)
    normalize = lambda items: {(s, d, t, round(w, 6)) for s, d, t, w in items}
    if normalize(full["edges"]) != normalize(edges):
        raise RuntimeError("Cached graph builder disagrees with full graph")
    print("[OK] Cached/full graph edge tables agree", flush=True)
    matrix = df[["sdk_" + name for name in sdks]].to_numpy(float)
    checks = run_verify(
        df, matrix, sdks, matrix.sum(0),
        {s["name"] for s in load_signatures() if s.get("miit_flagged")},
        folders, cache, apis, perms, len(sdks) + len(apis) + len(perms),
        CONTROL_SEED, diff_sdks)
    for name, ok in checks:
        print(f"[{'OK' if ok else 'FAIL'}] {name}", flush=True)
    if not all(ok for _, ok in checks):
        raise RuntimeError("OOD verification failed")
    if not np.isfinite(df[static + diff].to_numpy(float)).all():
        raise RuntimeError("Nonfinite model inputs")

def prepare():
    # Released labels are already cleaned; verify rather than rewrite source data.
    import csv
    from clean_labels import clean_one
    with (ROOT / "labels.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            _, classes = clean_one(row["violation_types"])
            if classes != row["violation_classes"].split("；"):
                raise ValueError("Released class mapping disagrees with taxonomy")
    for script in ["match_sdks.py", "build_dataset.py", "build_remediation.py",
                   "stat_verify.py", "build_graph.py"]:
        run(script)
    # Features may have changed, so do not reuse a pre-existing cache.
    from train_gnn import load_data
    load_data(True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=[
        "validate", "prepare", "baseline", "main", "attribution", "ood", "figures", "all"])
    parser.add_argument("--run-name", default="main_reproduction")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        parser.error("--run-name must contain only letters, digits, '_' or '-'")
    destination = ROOT / "runs" / args.run_name
    if args.stage in ("main", "all") and destination.exists():
        parser.error("Run directory already exists; choose a new --run-name")
    (ROOT / "out").mkdir(exist_ok=True)
    stages = (["prepare", "baseline", "main", "attribution", "ood", "figures"]
              if args.stage == "all" else [args.stage])
    for stage in stages:
        if stage == "validate":
            validate()
        elif stage == "prepare":
            prepare()
        elif stage == "baseline":
            run("train_baseline.py")
        elif stage == "main":
            run("train_gnn.py", "--output-dir", str(destination),
                "--threads", "1", "--rebuild-cache", "--verify")
            for name in ["gnn_report.md", "gnn_summary.json"]:
                shutil.copy2(destination / name, ROOT / "out" / name)
        elif stage == "attribution":
            run("interpret_gnn.py", "--selfcheck", "--threads", "1")
        elif stage == "ood":
            # Unlike the legacy CLI's print-only check, validate fails on violations.
            validate()
            run("eval_ood.py", "--threads", "1")
        elif stage == "figures":
            run("make_ch5_figures.py")

if __name__ == "__main__":
    main()
