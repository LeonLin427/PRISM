# Release validation

Date: 2026-10-20. Validation was performed in a separate copy of the release
candidate, using the existing recorded Python 3.11.15 environment on Windows.
No commands were run against the original analysis inputs to modify them.

## Passed

- Parsed all 19 distributed Python source files.
- Verified 433 retained JSONs, 217 event labels, 212 unique applications, 389
  graph nodes, and 26,061 edges.
- Verified three complete main-CV fold assignments and application-group
  isolation within each repetition.
- Checked all eight models' per-seed/deterministic predictions against fold
  assignments and labels. CSV aggregation matched the archived ensemble arrays
  within float32 rounding tolerance.
- Recomputed the eight OOF macro-AUC and macro-AUPRC point estimates from the
  archived arrays; all matched the stored full-precision summaries.
- Passed five SDK-matching tests, two remediation tests, and three
  training/reproducibility tests.
- Rebuilt the disposable graph cache; the cached and full builders produced
  identical relation/endpoint/weight sets.
- Passed temporal and SDK-holdout checks, including application isolation,
  the 37-event SDK usage cap, and zero held-out-SDK training edges.
- Executed `run_pipeline.py --stage prepare` from retained inputs.
  All regenerated CSVs below matched the archived references with numerical
  tolerance `1e-12`:

| Table | Rows | Columns |
| --- | ---: | ---: |
| APK feature dataset | 433 | 181 |
| Event remediation table | 217 | 45 |
| Graph event-feature table | 217 | 220 |
| Graph nodes | 389 | 3 |
| Graph edges | 26,061 | 4 |
| Nonconstant association tests | 156 | 10 |

- Executed the result-figure stage and produced all five Figures 3--7 inside the
  independent test directory, without accessing the original workspace figures.
- Scanned distributed text and workbook XML for common credential formats and
  personal home-directory paths; no matches remained. No file exceeds 100 MiB.
- Verified that the 455 original code/environment/input files recorded before
  assembly were unchanged. Only release copies received packaging edits.

## Not performed in this packaging task

- A fresh dependency installation, a Linux/macOS run, or a complete independent
  rerun of all main-CV, baseline, attribution, and OOD training.
- Re-extraction from original APKs, which are not available.
- Recalculation of all bootstrap intervals and comparison p-values. Archived
  results and code are provided; the point estimates were checked directly.
- A full manual security, privacy, copyright, or redistribution-permission audit.
- Publication, Git initialization, license selection, DOI registration, or edits
  to the manuscript.

These checks establish the tested packaging and reconstruction behavior; they
are not a claim of guaranteed bitwise model reproduction on every environment.

## English documentation update

After localization, all 19 distributed Python files were parsed and inspected:
no Chinese text remains in comments or module/class/function docstrings.
Original-language data identifiers and some diagnostic output strings remain.

AST comparison confirmed that changes were limited to documentation, CLI help,
and report prose. Dynamic report expressions and the executable analysis logic
were preserved. Synthetic report-writer checks, without model training or Excel
file changes, exercised both writers and their optional branches: the English
reports retained the same numerical tables as the original writers. Saved report
tables and case scores were also checked against the pre-translation copies.

The five matching, two differencing, and three training/reproducibility tests were
rerun successfully. The release checksum inventory was refreshed after removing
the internal publication checklist from the distribution.
