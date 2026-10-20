"""Read-only verification of the supplied release and preserved paper results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def require(condition, message):
    if not condition:
        raise RuntimeError(message)
    print("[OK]", message)

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-checksums", action="store_true",
                        help="Use after intentional regeneration of working outputs")
    args = parser.parse_args()
    checksum_file = ROOT / "SHA256SUMS.json"
    if not args.skip_checksums:
        require(checksum_file.exists(), "Release checksum manifest exists")
        expected = json.loads(checksum_file.read_text(encoding="utf-8"))["files"]
        mismatches = []
        for name, expected_hash in expected.items():
            path = ROOT / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
                mismatches.append(name)
        require(not mismatches, f"Release checksums match ({len(expected)} files); "
                f"mismatches: {mismatches}")

    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    features = list((ROOT / "features").glob("*.json"))
    labels = rows(ROOT / "labels.csv")
    require(len(features) == 433, "433 retained feature files")
    for path in features:
        item = json.loads(path.read_text(encoding="utf-8"))
        require_fields = {"folder", "apk_file", "api_calls"}
        if not require_fields.issubset(item):
            raise RuntimeError(f"Missing feature fields: {path.name}")
    require(len(labels) == 217, "217 event labels")
    require(len({r["folder"] for r in labels}) == 212, "212 unique applications")
    require(len({(r["folder"], r["batch"]) for r in labels}) == 217,
            "Unique event identifiers")
    require(len(rows(ROOT / "reference/remediation.csv")) == 217,
            "217 reference remediation records")
    require(len(rows(ROOT / "reference/graph/nodes.csv")) == 389, "389 graph nodes")
    require(len(rows(ROOT / "reference/graph/edges.csv")) == 26061, "26,061 graph edges")
    stats = pd.read_csv(ROOT / "reference/out/stats/remediation_stats.csv")
    require(len(stats) == 156, "156 nonconstant feature/category tests")
    require(not (stats.p_fdr < .05).any(), "No BH-FDR significant association")

    folds = pd.read_csv(ROOT / "reference/main_cv/gnn_fold_assignments.csv")
    require(len(folds) == 217 * 3, "Three complete fold assignments")
    for repeat, frame in folds.groupby("repeat"):
        require(set(frame.event_index) == set(range(217)) and
                set(frame.test_fold) == set(range(5)) and
                frame.groupby("app_group").test_fold.nunique().max() == 1,
                f"Repeat {repeat}: applications do not cross folds")
    model_data = pd.read_csv(ROOT / "reference/graph/app_features.csv")
    arrays = np.load(ROOT / "reference/main_cv/gnn_oof_ensemble.npz", allow_pickle=False)
    target_cols = [f"label_C{i}" for i in range(1, 7)]
    require(np.array_equal(arrays["y_true"], model_data[target_cols].to_numpy()),
            "Stored ensemble targets match graph labels")
    require(np.array_equal(arrays["app_group"], model_data.app_group.to_numpy()),
            "Stored ensemble groups match graph row order")
    summary = json.loads((ROOT / "reference/main_cv/gnn_summary.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(ROOT / "reference/main_cv/gnn_oof_seed_predictions.csv")
    require(set(predictions.variant) == set(summary["models"]), "Eight expected model variants")
    pcols = [f"p_C{i}" for i in range(1, 7)]
    ycols = [f"y_C{i}" for i in range(1, 7)]
    for model, values in summary["models"].items():
        frame = predictions[predictions.variant == model]
        expected_seeds = {-1} if model in {
            "lr_static", "lr_staticdiff", "knn_jaccard"} else {1000, 1001, 1002}
        require(len(frame) == 217 * 3 * len(expected_seeds) and
                not frame.duplicated(["repeat", "seed", "event_index"]).any() and
                set(frame.seed) == expected_seeds,
                f"{model}: complete event/repeat/seed predictions")
        indices = frame.event_index.to_numpy()
        require(np.array_equal(frame[ycols].to_numpy(), arrays["y_true"][indices]),
                f"{model}: per-seed targets match")
        for repeat, sub in frame.groupby("repeat"):
            mapping = folds[folds["repeat"] == repeat].set_index("event_index")
            require(np.array_equal(sub.fold.to_numpy(),
                                   mapping.loc[sub.event_index, "test_fold"].to_numpy()),
                    f"{model}: repeat {repeat} predictions match assigned folds")
        averaged = frame.groupby("event_index")[pcols].mean().sort_index().to_numpy()
        # Neural predictions were averaged in float32 before saving the ensemble;
        # CSV parsing/aggregation uses float64. Allow one float32 rounding unit.
        require(np.allclose(averaged, arrays[model], atol=1e-7, rtol=0),
                f"{model}: CSV averages match saved ensemble")
        auc = roc_auc_score(arrays["y_true"], arrays[model], average="macro")
        auprc = average_precision_score(arrays["y_true"], arrays[model], average="macro")
        require(abs(auc - values["oof_auc"]) < 1e-12 and
                abs(auprc - values["macro_auprc"]) < 1e-12,
                f"{model}: AUC {auc:.6f} and AUPRC reproduce stored summary")
    print("All release checks passed.")

if __name__ == "__main__":
    main()
