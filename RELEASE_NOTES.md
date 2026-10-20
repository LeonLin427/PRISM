# Release preparation notes

This directory was assembled from the current analysis files on 2026-10-20.
No model definition, experimental threshold, input value, or reported numerical
result was changed during packaging.

Packaging changes:

- Selected the current scripts, retained inputs, and final paper results; omitted
  historical runs, internal paper-editing records, temporary patch scripts,
  original-workspace documents, and disposable pickle caches.
- Kept current main-CV predictions, folds, ensemble arrays, summary, and manifest.
  The archived raw pickle is not included; full retraining and CSV/NPZ inspection
  are supported, but `train_gnn.py --report-only` requires a locally generated
  `gnn_oof_raw.pkl`.
- Changed the result-figure destination from a parent-workspace folder to
  `figures/`, and added an installed-font fallback.
- Corrected the label-distribution report description: analysis is event-level,
  with application-level grouping for splits. Label mapping code is unchanged.
- Normalized machine-specific paths in the historical manifest; preserved its
  recorded input hashes. Cleared creator/last-editor metadata in copied Excel
  result files without changing cell values.
- Added a portable pipeline runner, release validation, documentation, checksums,
  and ignore rules.
- Translated 155 comments and 61 docstrings across 14 Python files, plus model
  command-line help, into English. Updated stale descriptive notes to match the
  existing event-level implementation; executable analysis logic is unchanged.
- Translated the attribution and OOD Markdown reports and their rendering
  templates. The working and reference report copies retain identical numerical
  tables, SDK/application identifiers, rankings, and case scores.
- Kept internal publication-approval material outside the release directory.

The historical training manifest retains its original source hashes. Current
source hashes differ after documentation/report translation; `SHA256SUMS.json`
records the distributed files without rewriting historical run provenance.

The original analysis directory was not edited. This packaging task does not
constitute a new full experimental run. See `VALIDATION.md` for checks actually
performed and remaining limitations.
