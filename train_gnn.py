"""Train and evaluate RGCN variants, feature baselines, and graph ablations.

Use application-grouped iterative multilabel stratification with five folds,
--reps split repetitions, and --seeds stochastic-model initializations. Average
seeds within folds, reconstruct OOF predictions, and bootstrap application groups
rather than treating fitted instances as independent observations.

Training graphs contain only training events. Standardization and SDK
co-occurrence are fitted on training events and frozen when unlabeled test events
are attached. Auxiliary flagged/miit fields are excluded. Event differences are
retrospective evidence; repeated events from the same application share a fold.

Variants: gnn_full (features+graph), gnn_nofeat (constant event input), gnn_nosdk
(no SDK co-occurrence), gnn_sdkw2 (co-occurrence w>=2), mlp (features only),
lr_static, lr_staticdiff, and knn_jaccard.

Cache per-event issue-side SDK/permission/API inventories to build fold subgraphs
without repeated signature matching. Feature rows follow graph/app_features.csv.
The --verify option compares full-graph relation tables from both builders.

The prespecified superiority criterion requires paired application-group bootstrap
p<0.05 and a macro-AUC difference greater than 0.03."""
import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from gnn_model import RGCN, MLP
from match_sdks import load_signatures, match_one
from build_graph import load_feats_by_folder, load_label_sets

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.join(HERE, 'graph')
APP_FEATURES = os.path.join(GRAPH_DIR, 'app_features.csv')
CACHE_PKL = os.path.join(GRAPH_DIR, 'app_edge_cache.pkl')
REM_CSV = os.path.join(HERE, 'remediation.csv')
OUT_REPORT = os.path.join(HERE, 'out', 'gnn_report.md')


def split_removed(s):
    """Split the SDK lists using the same delimiter as build_graph.split_removed."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [x.strip() for x in str(s).split('；') if x.strip()]

EDGE_TYPES = ['app-uses-sdk', 'app-requests-perm', 'app-calls-api', 'sdk-sdk',
              'app-removed-sdk', 'app-added-sdk']
CLASSES = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6']
N_CLASSES = len(CLASSES)
META_COLS = {'folder', 'app_group', 'app_name', 'batch', 'has_fix'}

# Prespecified decision thresholds.
SIG_DELTA = 0.03
SIG_ALPHA = 0.05


# --------------------------------------------------------------------------
# Graph construction from cached per-event inventories.
# --------------------------------------------------------------------------

def prepare_cache(sigs, feats, info):
    """Cache each event's issue-side SDK counts, permission union, and API maxima.
    SDK counts increment once per matching issue APK, consistent with app_edges.
    The retained inventories can be reused across folds."""
    cache = {}
    for (folder, batch), d in info.items():
        prob_vers = [v for v in d['problem'] if v in feats.get(folder, {})]
        sdks = defaultdict(int)
        apis, perms = {}, set()
        for f in [feats[folder][v] for v in prob_vers]:
            for s, _ in match_one(f, sigs):
                sdks[s['name']] += 1
            for a, c in (f.get('api_calls') or {}).items():
                apis[a] = max(apis.get(a, 0), c)
            perms.update(f.get('permissions', []))
        cache[f'{folder}__{batch}'] = {'sdks': dict(sdks), 'perms': perms, 'apis': apis}
    return cache


def build_subgraph(folders, cache, sdk_names, api_names, perm_names, sdk_coocc_min_w=1,
                   diff_sdks=None, cooccurrence_folders=None):
    """Build a graph for the requested event subset.
    Event IDs precede the fixed SDK/permission/API vocabulary. By default,
    co-occurrence uses the event subset; cooccurrence_folders can instead freeze
    its estimation to training events when attaching test events.
    diff_sdks maps event IDs to removed/added SDK sets.
    Return n_nodes and (src, dst, relation, weight) edges."""
    n_app = len(folders)
    n_sdk, n_perm, n_api = len(sdk_names), len(perm_names), len(api_names)
    sdk_ids = {n: n_app + i for i, n in enumerate(sdk_names)}
    perm_ids = {n: n_app + n_sdk + i for i, n in enumerate(perm_names)}
    api_ids = {n: n_app + n_sdk + n_perm + i for i, n in enumerate(api_names)}

    edges = []
    cooccurrence_folders = set(folders if cooccurrence_folders is None
                               else cooccurrence_folders)
    sdk_users = defaultdict(set)  # sdk_name -> {app_idx}
    for i, folder in enumerate(folders):
        c = cache[folder]
        for name, cnt in c['sdks'].items():
            if name in sdk_ids:
                edges.append((i, sdk_ids[name], 'app-uses-sdk', math.log1p(cnt)))
                if folder in cooccurrence_folders:
                    sdk_users[name].add(i)
        for p in c['perms']:
            short = p.rsplit('.', 1)[-1]
            if short in perm_ids:
                edges.append((i, perm_ids[short], 'app-requests-perm', 1.0))
        for a, cnt in c['apis'].items():
            if a in api_ids:
                edges.append((i, api_ids[a], 'app-calls-api', math.log1p(cnt)))
        if diff_sdks is not None:
            for sd in diff_sdks.get(folder, {}).get('removed', set()):
                if sd in sdk_ids:
                    edges.append((i, sdk_ids[sd], 'app-removed-sdk', 1.0))
            for sd in diff_sdks.get(folder, {}).get('added', set()):
                if sd in sdk_ids:
                    edges.append((i, sdk_ids[sd], 'app-added-sdk', 1.0))
    for i in range(n_sdk):
        a = sdk_names[i]
        for j in range(i + 1, n_sdk):
            b = sdk_names[j]
            w = len(sdk_users[a] & sdk_users[b])
            if w >= sdk_coocc_min_w:
                ia, ib = sdk_ids[a], sdk_ids[b]
                edges.append((ia, ib, 'sdk-sdk', float(w)))
                edges.append((ib, ia, 'sdk-sdk', float(w)))
    return n_app + n_sdk + n_perm + n_api, edges


def to_adjacency(n_nodes, edges, drop_edges=()):
    """Convert edges to row-normalized weighted tensors, one per relation.
    With A[src,dst]>0 and A @ H, source rows aggregate target representations;
    relation orientation is not forward feature flow.
    drop_edges removes entire relation types; absent relations use zero matrices."""
    adj = [torch.zeros(n_nodes, n_nodes, dtype=torch.float32) for _ in EDGE_TYPES]
    for s, d, t, w in edges:
        if t in drop_edges:
            continue
        adj[EDGE_TYPES.index(t)][s, d] += w
    for i, A in enumerate(adj):
        rs = A.sum(dim=1, keepdim=True)
        adj[i] = A / rs.clamp(min=1e-9)
    return adj


# --------------------------------------------------------------------------
# Iterative stratification and training.
# --------------------------------------------------------------------------

def iterative_strat(labels, k, seed=0):
    """Iteratively stratify binary multilabel rows into k approximately balanced folds.
    Return one integer fold ID per row; follow Sechidis et al. (2011)."""
    N, C = labels.shape
    rng = np.random.RandomState(seed)
    folds = np.full(N, -1, dtype=int)
    remaining = set(range(N))
    need = {}  # Remaining desired positive-label count for each (label, fold).
    for c in range(C):
        n_pos = int(labels[:, c].sum())
        if n_pos == 0:
            continue
        base, rem = divmod(n_pos, k)
        cnts = [base + (1 if i < rem else 0) for i in range(k)]
        rng.shuffle(cnts)
        for f in range(k):
            need[(c, f)] = cnts[f]
    while remaining:
        rem_list = list(remaining)
        cnt = labels[rem_list].sum(axis=0)
        lead = int(np.argmax(cnt))
        if cnt[lead] == 0:
            break  # All remaining rows are negative.
        pos = [i for i in rem_list if labels[i, lead] == 1]
        # Allocate slots by each fold's demand for the selected label, then match positive rows.
        folds_sorted = sorted(range(k), key=lambda f: need.get((lead, f), 0), reverse=True)
        slots = []
        for f in folds_sorted:
            slots += [f] * max(need.get((lead, f), 0), 0)
        for i, f in zip(pos, slots):
            folds[i] = f
            for c in range(C):
                if labels[i, c] and need.get((c, f), 0) > 0:
                    need[(c, f)] -= 1
            remaining.discard(i)
    # Assign remaining all-negative rows to the least-filled fold.
    rest = list(remaining)
    rng.shuffle(rest)
    for i in rest:
        f = int(np.argmin([int((folds == x).sum()) for x in range(k)]))
        folds[i] = f
    return folds


def seeded_model(factory, seed, *args, **kwargs):
    """Seed BEFORE the constructor initializes embeddings, layers and readout."""
    torch.manual_seed(seed)
    return factory(*args, **kwargs)


def train_gnn(model, X_app, adj, train_ids, Y_train, epochs, lr, seed, use_graph=True):
    """Train with fixed hyperparameters and aligned event-feature/adjacency rows.
    use_graph=False selects direct feature-only MLP inference."""
    # Keep the existing dropout stream; constructors must separately use seeded_model.
    torch.manual_seed(seed)
    n_pos = Y_train.sum(0)
    pos_weight = ((len(train_ids) - n_pos) / np.clip(n_pos, 1, None)).clip(1, 50)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    train_ids = torch.tensor(train_ids, dtype=torch.long)
    Yt = torch.tensor(Y_train, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(X_app, adj, train_ids) if use_graph else model(X_app)
        loss = crit(logits, Yt)
        loss.backward()
        opt.step()
    return model


def infer(model, X_app, adj, node_ids):
    model.eval()
    with torch.no_grad():
        logits = model(X_app, adj, torch.tensor(node_ids, dtype=torch.long))
        return torch.sigmoid(logits).numpy()


def auc_pair(y_true, y_score):
    """Return per-category ROC AUC indexed by category IDs 0--5.
    Skip categories missing either label value or having constant predictions."""
    out = {}
    for c in range(N_CLASSES):
        n_p = int(y_true[:, c].sum())
        if 0 < n_p < len(y_true):
            try:
                out[c] = roc_auc_score(y_true[:, c], y_score[:, c])
            except ValueError:
                continue
    return out


def auprc_pair(y_true, y_score):
    """Return per-category average precision, excluding single-label categories."""
    out = {}
    for c in range(N_CLASSES):
        n_p = int(y_true[:, c].sum())
        if 0 < n_p < len(y_true):
            out[c] = average_precision_score(y_true[:, c], y_score[:, c])
    return out


# --------------------------------------------------------------------------
# Main workflow.
# --------------------------------------------------------------------------

VARIANTS = {
    'gnn_full':     {'kind': 'gnn', 'use_feats': True,  'w': 1, 'drop': ()},
    'gnn_nofeat':   {'kind': 'gnn', 'use_feats': False, 'w': 1, 'drop': ()},
    'gnn_nosdk':    {'kind': 'gnn', 'use_feats': True,  'w': 1, 'drop': ('sdk-sdk',)},
    'gnn_sdkw2':    {'kind': 'gnn', 'use_feats': True,  'w': 2, 'drop': ()},
    'mlp':          {'kind': 'mlp', 'use_feats': True},
    'lr_static':    {'kind': 'lr',  'feats': 'static'},
    'lr_staticdiff': {'kind': 'lr', 'feats': 'static+diff'},
    'knn_jaccard':  {'kind': 'knn', 'k': 5},
}


def load_data(rebuild_cache=False):
    """Load event features, feature groups, vocabularies, and the per-event edge cache."""
    if not os.path.exists(APP_FEATURES):
        raise SystemExit(f'缺 {APP_FEATURES}, 先跑 python build_graph.py')
    df = pd.read_csv(APP_FEATURES, encoding='utf-8-sig')
    folders = df['folder'].tolist()
    feat_cols = [c for c in df.columns if c not in META_COLS and not c.startswith('label_')]
    static_cols = [c for c in feat_cols if not c.startswith('d_')]
    diff_cols = [c for c in feat_cols if c.startswith('d_')]
    # Vocabulary order matches build_graph node allocation:
    # SDK/API names are sorted; permissions follow DANGEROUS_PERMS.
    sdk_names = [c[4:] for c in static_cols if c.startswith('sdk_')]
    api_names = [c[4:] for c in static_cols if c.startswith('api_')]
    perm_names = [c[5:] for c in static_cols
                  if c.startswith('perm_') and c not in
                  ('perm_total', 'perm_dangerous', 'perm_custom')]
    Y = df[[f'label_{c}' for c in CLASSES]].to_numpy(float)

    if os.path.exists(CACHE_PKL) and not rebuild_cache:
        with open(CACHE_PKL, 'rb') as f:
            cache = pickle.load(f)
    else:
        t0 = time.time()
        sigs = load_signatures()
        feats = load_feats_by_folder()
        info = load_label_sets()
        cache = prepare_cache(sigs, feats, info)
        with open(CACHE_PKL, 'wb') as f:
            pickle.dump(cache, f)
        print(f'[cache] 预计算 {len(cache)} App 边信息 {time.time()-t0:.0f}s -> '
              f'{os.path.basename(CACHE_PKL)}')

    # SDK removal/addition relations come from the same remediation.csv as build_graph.
    diff_sdks = {}
    rem = pd.read_csv(REM_CSV, encoding='utf-8-sig')
    for _, r in rem.iterrows():
        diff_sdks[f"{r['folder']}__{r.get('batch', '')}"] = {
            'removed': set(split_removed(r.get('removed_sdks'))),
            'added': set(split_removed(r.get('added_sdks'))),
        }
    return (df, folders, static_cols, diff_cols, sdk_names, api_names, perm_names, Y,
            cache, diff_sdks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--reps', type=int, default=3, help='Number of repeated fold assignments')
    ap.add_argument('--seeds', type=int, default=3, help='Number of stochastic-model seeds per fold')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--threads', type=int, default=1,
                    help='PyTorch CPU threads; default 1 avoids overhead on this small dense graph')
    ap.add_argument('--quick', action='store_true', help='Smoke test: one repeat, one seed, 60 epochs')
    ap.add_argument('--rebuild-cache', action='store_true')
    ap.add_argument('--verify', action='store_true', help='Compare cached and full graph edge tables')
    ap.add_argument('--variants', default=None, help='Comma-separated variant identifiers; defaults to all')
    ap.add_argument('--output-dir', default=os.path.join(HERE, 'out'),
                    help='Output directory for reports, per-seed OOF predictions, splits, and run settings')
    ap.add_argument('--report-only', action='store_true',
                    help='Rebuild reports from saved complete OOF predictions without training')
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))

    if args.quick:
        args.reps, args.seeds, args.epochs = 1, 1, 60

    (df, folders, static_cols, diff_cols, sdk_names, api_names, perm_names, Y,
     cache, diff_sdks) = load_data(args.rebuild_cache)
    n_app = len(folders)
    n_nonapp = len(sdk_names) + len(perm_names) + len(api_names)
    print(f'数据: {n_app} App / SDK {len(sdk_names)} 权限 {len(perm_names)} API {len(api_names)}'
          f' / 特征 {len(static_cols)+len(diff_cols)}(static {len(static_cols)}+diff {len(diff_cols)})')

    # Raw features; fit standardization separately on each training fold.
    feat_all = df[static_cols + diff_cols].to_numpy(float)
    sdk_mat = df[[f'sdk_{n}' for n in sdk_names]].to_numpy(float)  # Input to kNN-Jaccard.

    variants = {k: VARIANTS[k] for k in args.variants.split(',')} if args.variants else VARIANTS
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        with (output_dir / 'gnn_oof_raw.pkl').open('rb') as f:
            raw = pickle.load(f)
        report(raw, args, Y, df['app_group'].to_numpy(), output_dir)
        return
    prediction_path = output_dir / 'gnn_oof_seed_predictions.csv'
    if prediction_path.exists():
        raise SystemExit(f'Refusing to overwrite saved predictions: {prediction_path}')
    inputs = [APP_FEATURES, CACHE_PKL, REM_CSV, __file__,
              os.path.join(HERE, 'gnn_model.py')]
    manifest = {
        'status': 'running', 'args': vars(args), 'variants': variants,
        'initialization_and_dropout_seeds': [1000 + s for s in range(args.seeds)],
        'split_seeds': list(range(args.reps)), 'device': 'cpu',
        'seed_protocol': 'seed before construction; reset same seed before training',
        'python': sys.version, 'platform': platform.platform(),
        'versions': {'torch': torch.__version__, 'numpy': np.__version__,
                     'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'sha256': {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                   for p in inputs},
        'classes': CLASSES, 'static_columns': static_cols, 'difference_columns': diff_cols,
        'bootstrap': {'n': 1000, 'ci_seed': 2024, 'comparison_seed': 2025},
    }
    (output_dir / 'gnn_run_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    if args.verify:
        verify_subgraph(folders, cache, sdk_names, api_names, perm_names, diff_sdks)

    # Records: variant -> (repeat, fold, test indices, targets, seed-averaged scores).
    raw = defaultdict(list)
    fold_records = []
    t_start = time.time()
    # Group-stratify event IDs folder__batch so repeated application events share a fold.
    # Union event labels within each application before stratifying groups.
    groups = df['app_group'].to_numpy()
    uniq_groups = sorted(set(groups))
    g_idx = {g: i for i, g in enumerate(uniq_groups)}
    gY = np.zeros((len(uniq_groups), Y.shape[1]))
    for i, g in enumerate(uniq_groups):
        gY[i] = Y[groups == g].max(axis=0)

    for rep in range(args.reps):
        gfolds = iterative_strat(gY, args.folds, seed=rep)
        folds = np.array([gfolds[g_idx[g]] for g in groups])
        for i, f in enumerate(folds):
            fold_records.append({'repeat': rep, 'test_fold': int(f), 'event_index': i,
                                 'event_id': folders[i], 'app_group': groups[i]})
        pd.DataFrame(fold_records).to_csv(output_dir / 'gnn_fold_assignments.csv', index=False)
        for fold in range(args.folds):
            train_idx = np.where(folds != fold)[0]
            test_idx = np.where(folds == fold)[0]
            assert not (set(groups[train_idx]) & set(groups[test_idx]))
            train_folders = [folders[i] for i in train_idx]
            test_folders = [folders[i] for i in test_idx]
            pred_folders = train_folders + test_folders

            mean = feat_all[train_idx].mean(0)
            std = feat_all[train_idx].std(0)
            std[std == 0] = 1.0
            feat_z_train = (feat_all[train_idx] - mean) / std
            feat_z_test = (feat_all[test_idx] - mean) / std
            feat_z_pred = np.vstack([feat_z_train, feat_z_test])

            # Build once per fold and freeze training-derived co-occurrence during inference.
            graphs = {}
            for w in sorted({cfg['w'] for cfg in variants.values() if cfg['kind'] == 'gnn'}):
                graphs[(w, 'tr')] = build_subgraph(train_folders, cache, sdk_names,
                                                   api_names, perm_names, w, diff_sdks)
                graphs[(w, 'te')] = build_subgraph(
                    pred_folders, cache, sdk_names, api_names, perm_names, w,
                    diff_sdks, cooccurrence_folders=train_folders)
            Y_tr = Y[train_idx]
            Y_te = Y[test_idx]

            for vname, cfg in variants.items():
                score_runs = []
                n_runs = args.seeds if cfg['kind'] in ('gnn', 'mlp') else 1
                for s in range(n_runs):
                    kind = cfg['kind']
                    model_seed = 1000 + s
                    t0 = time.time()
                    if kind == 'gnn':
                        n_t, e_t = graphs[(cfg['w'], 'tr')]
                        n_f, e_f = graphs[(cfg['w'], 'te')]
                        adj_t = to_adjacency(n_t, e_t, cfg['drop'])
                        adj_f = to_adjacency(n_f, e_f, cfg['drop'])
                        Xtr = feat_z_train if cfg['use_feats'] else \
                            np.ones((len(train_idx), feat_all.shape[1]))
                        Xfu = feat_z_pred if cfg['use_feats'] else \
                            np.ones((len(pred_folders), feat_all.shape[1]))
                        model = seeded_model(RGCN, model_seed, feat_all.shape[1],
                                             n_nonapp, len(EDGE_TYPES), N_CLASSES)
                        model = train_gnn(model, torch.tensor(Xtr, dtype=torch.float32),
                                          adj_t, np.arange(len(train_idx)), Y_tr,
                                          args.epochs, args.lr, seed=model_seed)
                        scores = infer(model, torch.tensor(Xfu, dtype=torch.float32),
                                       adj_f, len(train_idx) + np.arange(len(test_idx)))
                    elif kind == 'mlp':
                        model = seeded_model(MLP, model_seed, feat_all.shape[1], N_CLASSES)
                        model = train_gnn(model, torch.tensor(feat_z_train, dtype=torch.float32),
                                          [torch.zeros(0)], np.arange(len(train_idx)), Y_tr,
                                          args.epochs, args.lr, seed=model_seed, use_graph=False)
                        with torch.no_grad():
                            model.eval()
                            scores = torch.sigmoid(
                                model(torch.tensor(feat_z_test, dtype=torch.float32))).numpy()
                    elif kind == 'lr':
                        use = static_cols if cfg['feats'] == 'static' else static_cols + diff_cols
                        col_idx = [i for i, c in enumerate(static_cols + diff_cols) if c in use]
                        sc = StandardScaler().fit(feat_all[train_idx][:, col_idx])
                        clf = OneVsRestClassifier(
                            LogisticRegression(C=1.0, max_iter=2000))
                        clf.fit(sc.transform(feat_all[train_idx][:, col_idx]), Y_tr)
                        scores = clf.predict_proba(sc.transform(feat_all[test_idx][:, col_idx]))
                    elif kind == 'knn':
                        scores = knn_jaccard(sdk_mat, train_idx, test_idx, Y, k=cfg['k'])
                    score_runs.append(scores)
                    saved = pd.DataFrame({
                        'variant': vname, 'repeat': rep, 'fold': fold,
                        'seed': model_seed if kind in ('gnn', 'mlp') else -1,
                        'event_index': test_idx, 'event_id': [folders[i] for i in test_idx],
                        'app_group': groups[test_idx],
                    })
                    for c, name in enumerate(CLASSES):
                        saved[f'y_{name}'] = Y_te[:, c].astype(int)
                        saved[f'p_{name}'] = scores[:, c]
                    # Float64 serialization preserves every float32 prediction exactly.
                    saved.to_csv(prediction_path, mode='a', index=False,
                                 header=not prediction_path.exists(), float_format='%.17g')
                    dt = time.time() - t0
                    print(f'  rep{rep} fold{fold} {vname:<12} seed{model_seed if n_runs > 1 or kind in ("gnn", "mlp") else "NA"}  {dt:5.1f}s '
                          f'macro={macro_of(Y_te, scores):.3f}')
                scores = np.mean(score_runs, axis=0)
                raw[vname].append((rep, fold, test_idx, Y_te, scores))
            print(f'  [rep{rep} fold{fold} 完成] {time.time()-t_start:.0f}s')
            with (output_dir / 'gnn_oof_raw.pkl').open('wb') as f:
                pickle.dump(dict(raw), f)
    print(f'\n训练完成 {time.time()-t_start:.0f}s')

    report(raw, args, Y, groups, output_dir)
    manifest['status'] = 'complete'
    manifest['elapsed_seconds'] = time.time() - t_start
    (output_dir / 'gnn_run_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n报告 -> {output_dir / "gnn_report.md"}')


def macro_of(y_true, y_score):
    a = auc_pair(y_true, y_score)
    return float(np.mean(list(a.values()))) if a else float('nan')


def knn_jaccard(sdk_mat, train_idx, test_idx, Y, k=5):
    """Average labels of the top-k neighbors under SDK-set Jaccard similarity."""
    sdk_train = sdk_mat[train_idx].astype(bool)
    sdk_test = sdk_mat[test_idx].astype(bool)
    scores = np.zeros((len(test_idx), Y.shape[1]))
    for i, s in enumerate(sdk_test):
        inter = (sdk_train & s).sum(1).astype(float)
        union = (sdk_train | s).sum(1).astype(float)
        jac = np.where(union > 0, inter / np.clip(union, 1, None), 0.0)
        top = np.argsort(-jac)[:k]
        scores[i] = Y[train_idx[top]].mean(0)
    return scores


def verify_subgraph(folders, cache, sdk_names, api_names, perm_names, diff_sdks):
    """Compare full edge tables from the cached subgraph and full graph builders."""
    from build_graph import build_graph
    t0 = time.time()
    g = build_graph()
    n_nodes, edges = build_subgraph(folders, cache, sdk_names, api_names, perm_names, 1, diff_sdks)
    g_edges = set((s, d, t, round(w, 6)) for s, d, t, w in g['edges'])
    my_edges = set((s, d, t, round(w, 6)) for s, d, t, w in edges)
    print(f'[verify] build_graph {time.time()-t0:.0f}s, 边 {len(g["edges"])} vs 子图 {len(edges)}')
    if g_edges == my_edges:
        print('[verify] 边表完全一致 [OK]')
    else:
        print(f'[verify] 差异: 只在 build_graph {len(g_edges - my_edges)}, '
              f'只在子图 {len(my_edges - g_edges)}')


def grouped_bootstrap(y_true, score_a, groups, score_b=None, n_boot=1000, seed=0):
    """Bootstrap OOF predictions by application group.
    Return a model AUC and CI, or paired delta-AUC, CI, and two-sided p-value
    when score_b is supplied. Repeated events from one app form one resampling block."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, float)
    score_a = np.asarray(score_a, float)
    groups = np.asarray(groups)
    score_b = None if score_b is None else np.asarray(score_b, float)
    uniq = np.array(sorted(set(groups.tolist())), dtype=object)
    members = [np.where(groups == g)[0] for g in uniq]
    point = macro_of(y_true, score_a)
    if score_b is not None:
        point -= macro_of(y_true, score_b)
    vals = []
    attempts = 0
    while len(vals) < n_boot and attempts < n_boot * 2:
        attempts += 1
        sampled = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in sampled])
        va = macro_of(y_true[idx], score_a[idx])
        if not np.isfinite(va):
            continue
        if score_b is None:
            vals.append(va)
        else:
            vb = macro_of(y_true[idx], score_b[idx])
            if np.isfinite(vb):
                vals.append(va - vb)
    vals = np.asarray(vals, float)
    ci = np.percentile(vals, [2.5, 97.5])
    if score_b is None:
        return point, ci, None
    p = min(1.0, 2 * min((np.sum(vals <= 0) + 1) / (len(vals) + 1),
                         (np.sum(vals >= 0) + 1) / (len(vals) + 1)))
    return point, ci, p


def report(raw, args, Y, groups, output_dir=None):
    """Average fold seeds, reconstruct repeated OOF predictions, and report group-bootstrap results."""
    n_app = len(Y)
    repeat_scores = defaultdict(dict)
    for vname, recs in raw.items():
        for rep in range(args.reps):
            scores = np.full((n_app, N_CLASSES), np.nan)
            for r, _fold, test_idx, _y_true, y_score in recs:
                if r == rep:
                    scores[test_idx] = y_score
            if np.isnan(scores).any():
                raise RuntimeError(f'{vname} rep={rep} 的 OOF 预测不完整')
            repeat_scores[vname][rep] = scores

    avg_scores = {v: np.mean([repeat_scores[v][r] for r in range(args.reps)], axis=0)
                  for v in repeat_scores}
    repeat_macro = {v: np.array([macro_of(Y, repeat_scores[v][r])
                                 for r in range(args.reps)])
                    for v in repeat_scores}
    point_ci = {v: grouped_bootstrap(Y, avg_scores[v], groups, seed=2024)
                for v in avg_scores}
    output_dir = Path(output_dir or os.path.dirname(OUT_REPORT))
    summary = {'models': {}, 'comparisons': {}}
    for v in avg_scores:
        point, ci, _ = point_ci[v]
        summary['models'][v] = {
            'oof_auc': point, 'ci': ci.tolist(),
            'repeat_auc': repeat_macro[v].tolist(),
            'repeat_mean': float(repeat_macro[v].mean()),
            'repeat_std': float(repeat_macro[v].std()),
            'macro_auprc': float(np.mean(list(auprc_pair(Y, avg_scores[v]).values()))),
            'class_auc': {CLASSES[c]: a for c, a in auc_pair(Y, avg_scores[v]).items()},
        }
    np.savez_compressed(output_dir / 'gnn_oof_ensemble.npz', y_true=Y,
                        app_group=np.asarray(groups, dtype=str), **avg_scores)

    order = ['gnn_full', 'gnn_nofeat', 'gnn_nosdk', 'gnn_sdkw2', 'mlp',
             'lr_static', 'lr_staticdiff', 'knn_jaccard']
    order = [v for v in order if v in avg_scores]

    def p_fmt(p):
        return '<0.001' if p < 0.001 else f'{p:.3f}'

    def cmp_row(a_, b_, two_sided_label=False):
        dm, ci, p = grouped_bootstrap(Y, avg_scores[a_], groups, avg_scores[b_], seed=2025)
        summary['comparisons'][f'{a_} vs {b_}'] = {
            'delta': dm, 'ci': ci.tolist(), 'p': p,
            'superiority': bool(p < SIG_ALPHA and dm > SIG_DELTA),
        }
        if two_sided_label:
            verdict = '有差异' if (p < SIG_ALPHA and abs(dm) > SIG_DELTA) else 'comparable'
        else:
            verdict = '显著优于' if (p < SIG_ALPHA and dm > SIG_DELTA) else \
                      ('显著劣于' if (p < SIG_ALPHA and dm < -SIG_DELTA) else 'comparable')
        return (f'| {a_} vs {b_} | {dm:+.3f} | [{ci[0]:+.3f},{ci[1]:+.3f}] '
                f'| {p_fmt(p)} | {verdict} |')

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    L = []
    L.append('# M3 结果: RGCN 多标签画像 vs 基线与消融')
    L.append('')
    L.append(f'- CV: {args.folds} 折 × {args.reps} 次重复划分 × {args.seeds} 个随机种子; '
             f'折内先平均随机种子预测; epochs={args.epochs}, lr={args.lr}')
    L.append('- 主指标: 每次重复的全量 OOF macro-AUC；点估计使用重复平均后的 OOF 预测')
    L.append(f'- 初始化和训练种子: {[1000+s for s in range(args.seeds)]}; '
             '构造每个模型之前设种子，训练前重设同一种子控制 dropout')
    L.append(f'- 事先制定的判定: app_group 配对 bootstrap(1000 次) p<{SIG_ALPHA} 且 '
             f'Δmacro-AUC>{SIG_DELTA} 才算"显著优于"; 达不到如实写 comparable')
    L.append('- 泄漏防线: 标准化只拟合训练集；训练图只含 train App；推理图冻结训练集 '
             'sdk-sdk 权重后接入 test App；特征无 flagged/miit')
    L.append('')

    L.append('## 主结果')
    L.append('')
    hdr = ['变体', 'OOF macro-AUC(mean±std across repeats)', 'OOF ensemble',
           'group-bootstrap 95% CI', 'macro-AUPRC'] + \
          [f'{c}(n={int(Y[:, i].sum())})' for i, c in enumerate(CLASSES)]
    L.append('| ' + ' | '.join(hdr) + ' |')
    L.append('|' + '---|' * len(hdr))
    for vname in order:
        s = repeat_macro[vname]
        point, ci, _ = point_ci[vname]
        pr = auprc_pair(Y, avg_scores[vname])
        row = [f'{vname}', f'{s.mean():.3f}±{s.std():.3f}', f'{point:.3f}',
               f'[{ci[0]:.3f},{ci[1]:.3f}]', f'{np.mean(list(pr.values())):.3f}']
        for i, c in enumerate(CLASSES):
            vals = auc_pair(Y, avg_scores[vname])
            row.append(f'{vals[i]:.3f}' if i in vals else '-')
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
    L.append('注: C4/C5/C6 阳性样本较少，per-class AUC 仅作描述；同时报告 macro-AUPRC。')
    L.append('')

    # Three SDK co-occurrence variants.
    L.append('## 消融: sdk-sdk 共现层三格(事先指定对比)')
    L.append('')
    L.append('| 配置 | macro-AUC(mean±std) | 95% CI |')
    L.append('|---|---|---|')
    for v in ['gnn_full', 'gnn_sdkw2', 'gnn_nosdk']:
        s = repeat_macro[v]
        _point, ci, _ = point_ci[v]
        L.append(f'| {v} | {s.mean():.3f}±{s.std():.3f} | [{ci[0]:.3f},{ci[1]:.3f}] |')
    L.append('')
    # Paired comparisons between the three variants.
    L.append('### sdk-sdk 两两配对对比')
    L.append('')
    L.append('| 对比 | Δmacro-AUC | 95% CI | p | 判定 |')
    L.append('|---|---|---|---|---|')
    for a_, b_ in [('gnn_full', 'gnn_sdkw2'), ('gnn_full', 'gnn_nosdk'), ('gnn_sdkw2', 'gnn_nosdk')]:
        L.append(cmp_row(a_, b_, two_sided_label=True))
    L.append('')

    # Feature-only, graph-only, and combined representations.
    L.append('## 消融: 信息源(仅特征 / 仅图 / 特征+图)')
    L.append('')
    L.append('| 配置 | macro-AUC(mean±std) | 95% CI |')
    L.append('|---|---|---|')
    for v in ['mlp', 'gnn_nofeat', 'gnn_full']:
        s = repeat_macro[v]
        _point, ci, _ = point_ci[v]
        L.append(f'| {v} | {s.mean():.3f}±{s.std():.3f} | [{ci[0]:.3f},{ci[1]:.3f}] |')
    L.append('')
    L.append('mlp=仅特征(静态+差分); gnn_nofeat=仅图(App 输入常数, 靠结构传播); gnn_full=特征+图。')
    L.append('')
    L.append('### 信息源两两配对对比')
    L.append('')
    L.append('| 对比 | Δmacro-AUC | 95% CI | p | 判定 |')
    L.append('|---|---|---|---|---|')
    for a_, b_ in [('gnn_nofeat', 'gnn_full'), ('gnn_nofeat', 'mlp'), ('mlp', 'gnn_full')]:
        L.append(cmp_row(a_, b_, two_sided_label=True))
    L.append('')

    # Paired significance comparisons with baselines.
    L.append('## 主模型 vs 基线(事先指定判定)')
    L.append('')
    L.append('| 对比 | Δmacro-AUC | 95% CI | p | 判定 |')
    L.append('|---|---|---|---|---|')
    for b_ in ['lr_static', 'lr_staticdiff', 'knn_jaccard', 'mlp']:
        L.append(cmp_row('gnn_full', b_))
    L.append('')
    L.append('## 说明与限制')
    L.append('')
    L.append('- sdk-sdk 层是否提供独立增益由消融结果决定；若 comparable，不把该层称为核心增量。')
    L.append('- 无负样本(未通报≠干净 App): 结果是"通报池内违规画像"而非检测器。')
    L.append('- 差分特征为回顾性取证; 模型输入与标签同源(整改前后), 泛化见 M5。')

    with (output_dir / 'gnn_report.md').open('w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')
    (output_dir / 'gnn_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
