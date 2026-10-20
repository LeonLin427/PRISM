# Retained data and output conventions

## Units and identifiers

- `labels.csv`: 217 event records, 212 distinct application identifiers in `folder`.
  An event is identified by the pair `(folder, batch)`. The fields also preserve
  version pairing, original violation text, the six-category mapping, developer,
  and source-channel information.
- `features/*.json`: 433 unique static-extraction records. The two logical sides
  of each of 217 events define 434 positions; one remediated APK is reused across
  adjacent events. File counts therefore need not equal twice the event count.
- `dataset.csv`: 433 APK rows for the issue/remediated discrimination task.
  `group` identifies the application used for grouped cross-validation.
- `remediation.csv`: 217 paired-event rows. Removed/added item counts are
  nonnegative. Sensitive API difference columns use issue minus remediated counts;
  each side uses the maximum over its available versions.
- `graph/app_features.csv`: 217 event rows with 209 model-input features, six
  labels, and identifier/metadata columns. `app_group` links repeated events from
  one application. Identifiers and labels are not model inputs.

## Static-extraction schema

| Field group | Meaning |
| --- | --- |
| `folder`, `apk_file`, `package`, `version_name`, `version_code` | Application and version identifiers |
| `md5`, `size`, `apk_mtime`, `schema_v` | File provenance and extraction metadata |
| `permissions`, `meta_keys` | Manifest permissions and metadata key names |
| `activities`, `services`, `receivers`, `providers` | Manifest component names |
| `prefix2`, `prefix3` | Observed DEX package prefixes and occurrence counts |
| `api_calls` | Sensitive method-name occurrence counts in DEX bytes, not runtime calls |
| `so_libs`, `assets`, `domains` | Native-library/asset names and extracted domain names |

The records are not anonymized app inventories. They do not include raw APK
binaries, but retain third-party identifiers and metadata. The contributors'
research data are provided under the MIT License; this does not license the
underlying third-party applications or SDKs. See the licensing scope in
`README.md`.

## Reference predictions

`reference/main_cv/gnn_fold_assignments.csv` has one row per event and split
repetition. `test_fold` assigns the event's held-out fold. All events from an
application share the same fold within a repetition.

`gnn_oof_seed_predictions.csv` stores model, repetition, fold, seed, event index,
identifiers, and six target/probability pairs. Neural models have three seeds;
deterministic LR/kNN models have one row per event/repetition, marked with seed
`-1` rather than three duplicated observations.

`gnn_oof_ensemble.npz` contains the targets, application groups, and the final
OOF probabilities for each variant. `verify_release.py` checks CSV aggregation
against the saved arrays with absolute tolerance `1e-7` because neural predictions
were averaged in float32. AUC and AUPRC are checked against the saved summary
at tolerance `1e-12`.

The historical run manifest describes the original execution. The release
checksum manifest verifies the distributed source, input, and reference files;
it excludes working derived tables, caches, figures, and `out/` so these can be
regenerated without changing the preserved reference results.
