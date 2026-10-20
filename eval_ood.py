"""Evaluate temporal and SDK-holdout generalization.

Outputs: out/m5_report.md and out/m5_results.xlsx. The corpus contains 217 events
from notice batches within 13--53, covering 2021--2025.

Temporal evaluation uses batch <=41 for training and later events for testing,
with cutoff 45 as a second split. Preserve all later events, but remove earlier
training events from applications also present in the test period. Evaluate
full RGCN, static LR, and static-plus-difference LR. Fit standardization and SDK
co-occurrence only on training events; freeze them at inference.

SDK holdout uses directly matched SDKs from the auxiliary list, excluding Zxing.
If any event in an application group uses a held-out SDK, assign the whole group
to test. Remove training relations touching held-out SDKs, including difference
relations. The matched random control draws the same number of nonlisted SDKs,
uses the maximum frequency in H as its cap, and accepts the first training-size
match within six events using seed 2024. This is one exploratory control draw.

Test events are attached without labels only at inference. Auxiliary list markers
are excluded from inputs, application groups are disjoint, and difference features
represent retrospective evidence."""
import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from gnn_model import RGCN
from train_gnn import load_data, build_subgraph, to_adjacency, train_gnn, infer, \
    auc_pair, macro_of, EDGE_TYPES, CLASSES, N_CLASSES
from match_sdks import load_signatures

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_REPORT = os.path.join(HERE, 'out', 'm5_report.md')
OUT_XLSX = os.path.join(HERE, 'out', 'm5_results.xlsx')

# Fixed analysis protocol; corpus-dependent bounds are calculated from inputs.
PRIMARY_CUTOFF = 41         # Primary temporal cutoff: batches <=41 train, later batches test.
ROBUST_CUTOFF = 45          # Secondary temporal cutoff: batches <=45 train, later batches test.
CROSSBATCH_MODELS = ['gnn_full', 'lr_static', 'lr_staticdiff']
SEEDS = [1000, 2000, 3000]
EPOCHS, LR = 300, 1e-3
W = 1                       # All observed SDK co-occurrence pairs, as in the full RGCN.
LR_C, LR_MAX_ITER = 1.0, 2000

# Hold out auxiliary-list SDKs with direct evidence under conservative matching.
# Exclude the general-purpose Zxing QR-code library from the risk-SDK holdout.
HOLDOUT_H = ['MobTech(ShareSDK/MobSDK)', '神策数据', '七牛云 SDK', 'TalkingData',
             'UBiX Merak', '乐播投屏', '维智地图', '易观方舟分析']
CONTROL_K = len(HOLDOUT_H)  # Match the number of SDKs in H.
CONTROL_SEED = 2024
CONTROL_TRAIN_TOL = 6       # Allowed difference in training-set size relative to H.
CONTROL_MAX_ATTEMPTS = 500
MIN_POS = 12                # Interpret categories with fewer than 12 test positives descriptively.


# --------------------------------------------------------------------------
# Data splits.
# --------------------------------------------------------------------------

def parse_batch_num(df):
    return df['batch'].astype(str).str.extract(r'(\d+)')[0].astype(int).to_numpy()


def split_by_batch(df, cutoff):
    """Split by time and purge training events from test application groups.
    Preserve all later events; omit earlier events belonging to the same
    applications so the held-out period remains unchanged."""
    bn = parse_batch_num(df)
    te = np.where(bn > cutoff)[0]
    groups = df['app_group'].astype(str).to_numpy()
    test_groups = set(groups[te])
    tr = np.where((bn <= cutoff) & ~np.isin(groups, list(test_groups)))[0]
    assert set(tr).isdisjoint(set(te))
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    return tr, te


def split_by_sdk_holdout(sdk_mat, H_idx, groups):
    """Hold out complete application groups that use any SDK in H.
    A single matching event assigns every event in that application to test."""
    used_event = sdk_mat[:, H_idx].sum(1) > 0
    test_groups = set(np.asarray(groups, dtype=str)[used_event])
    is_test_group = np.isin(np.asarray(groups, dtype=str), list(test_groups))
    tr = np.where(~is_test_group)[0]
    te = np.where(is_test_group)[0]
    assert set(np.asarray(groups)[tr]).isdisjoint(set(np.asarray(groups)[te]))
    return tr, te


def filter_diff_sdks(diff_sdks, H_idx, sdk_names):
    """Remove difference relations touching held-out SDKs.
    Otherwise, a training event that adds an SDK only in its remediated version
    could connect that held-out SDK through an app-added-sdk relation."""
    exclude = {sdk_names[i] for i in H_idx}
    return {f: {'removed': d['removed'] - exclude, 'added': d['added'] - exclude}
            for f, d in diff_sdks.items()}


def random_control_H(sdk_names, usage, flagged_set, sdk_mat, groups, control_seed,
                     usage_cap, target_train_n):
    """Draw a size-matched random control from nonlisted SDKs below H's usage cap.
    Accept the first draw within CONTROL_TRAIN_TOL of the target training size."""
    win_lo = max(1, target_train_n - CONTROL_TRAIN_TOL)
    win_hi = min(len(sdk_mat) - 1, target_train_n + CONTROL_TRAIN_TOL)
    pool = [i for i, n in enumerate(sdk_names)
            if n not in flagged_set and usage[i] <= usage_cap]
    assert len(pool) >= CONTROL_K, f'池太小: {len(pool)}'
    rng = np.random.RandomState(control_seed)
    for attempt in range(CONTROL_MAX_ATTEMPTS):
        H_idx = rng.choice(pool, CONTROL_K, replace=False)
        tr, _ = split_by_sdk_holdout(sdk_mat, H_idx, groups)
        if win_lo <= len(tr) <= win_hi:
            return H_idx, len(tr), attempt, win_lo, win_hi
    raise SystemExit(f'控制组 {CONTROL_MAX_ATTEMPTS} 次内未匹配到 train∈'
                     f'[{win_lo},{win_hi}]')


# --------------------------------------------------------------------------
# Model evaluation.
# --------------------------------------------------------------------------

def run_gnn(feat_all, train_idx, test_idx, Y, folders, cache, sdk_names, api_names,
            perm_names, seeds, n_nonapp, epochs, lr,
            diff_sdks=None, diff_sdks_train=None):
    """Train on isolated training events and attach unlabeled test events at inference.
    Standardization and SDK co-occurrence are fitted on training events only.
    diff_sdks_train can exclude held-out-SDK relations from training; None uses
    the same difference mapping as inference."""
    train_folders = [folders[i] for i in train_idx]
    test_folders = [folders[i] for i in test_idx]
    pred_folders = train_folders + test_folders
    ds_train = diff_sdks if diff_sdks_train is None else diff_sdks_train
    n_t, e_t = build_subgraph(train_folders, cache, sdk_names, api_names, perm_names, W,
                              ds_train)
    n_f, e_f = build_subgraph(pred_folders, cache, sdk_names, api_names, perm_names, W,
                              diff_sdks, cooccurrence_folders=train_folders)
    adj_t = to_adjacency(n_t, e_t)
    adj_f = to_adjacency(n_f, e_f)
    Y_tr, Y_te = Y[train_idx], Y[test_idx]
    m = feat_all[train_idx].mean(0)
    s = feat_all[train_idx].std(0)
    s[s == 0] = 1.0
    Xtr_np = (feat_all[train_idx] - m) / s
    Xte_np = (feat_all[test_idx] - m) / s
    Xtr = torch.tensor(Xtr_np, dtype=torch.float32)
    Xfu = torch.tensor(np.vstack([Xtr_np, Xte_np]), dtype=torch.float32)
    macros, per_class = [], {}
    for seed in seeds:
        torch.manual_seed(seed)
        model = RGCN(feat_all.shape[1], n_nonapp, len(EDGE_TYPES), N_CLASSES)
        train_gnn(model, Xtr, adj_t, np.arange(len(train_idx)), Y_tr, epochs, lr, seed)
        scores = infer(model, Xfu, adj_f, len(train_idx) + np.arange(len(test_idx)))
        macros.append(macro_of(Y_te, scores))
        for c, v in auc_pair(Y_te, scores).items():
            per_class.setdefault(c, []).append(v)
    return np.array(macros), per_class, (n_t, len(e_t), n_f, len(e_f))


def run_lr(feat_all, static_cols, diff_cols, feats_use, train_idx, test_idx, Y):
    """Deterministic OvR LR with C=1.0 and train-only StandardScaler, matching main CV."""
    col_idx = [i for i, c in enumerate(static_cols + diff_cols) if c in feats_use]
    sc = StandardScaler().fit(feat_all[train_idx][:, col_idx])
    clf = OneVsRestClassifier(LogisticRegression(C=LR_C, max_iter=LR_MAX_ITER))
    clf.fit(sc.transform(feat_all[train_idx][:, col_idx]), Y[train_idx])
    return clf.predict_proba(sc.transform(feat_all[test_idx][:, col_idx]))


def lr_result(feat_all, static_cols, diff_cols, feats_use, train_idx, test_idx, Y):
    scores = run_lr(feat_all, static_cols, diff_cols, feats_use, train_idx, test_idx, Y)
    return macro_of(Y[test_idx], scores), dict(auc_pair(Y[test_idx], scores))


# --------------------------------------------------------------------------
# Verification (--verify).
# --------------------------------------------------------------------------

def verify_holdout_zero_edges(edges, H_idx, n_app_in_subgraph):
    """Check that no relation touches a held-out SDK in the training subgraph."""
    h_nodes = {n_app_in_subgraph + i for i in H_idx}
    return not any(s in h_nodes or d in h_nodes for s, d, _t, _w in edges)


def run_verify(df, sdk_mat, sdk_names, usage, flagged_set, folders, cache,
               api_names, perm_names, n_nonapp, control_seed, diff_sdks=None):
    n_app = len(folders)
    groups = df['app_group'].astype(str).to_numpy()
    checks = []
    bad = [c for c in df.columns if 'flag' in c.lower() or 'miit' in c.lower()]
    checks.append(('特征无 flagged/miit', len(bad) == 0))
    tr41, te41 = split_by_batch(df, PRIMARY_CUTOFF)
    tr45, te45 = split_by_batch(df, ROBUST_CUTOFF)
    checks.append((f'X=41 切分 ({len(tr41)},{len(te41)}) app_group 隔离',
                   set(tr41).isdisjoint(te41)
                   and set(groups[tr41]).isdisjoint(set(groups[te41]))))
    checks.append((f'X=45 切分 ({len(tr45)},{len(te45)}) app_group 隔离',
                   set(tr45).isdisjoint(te45)
                   and set(groups[tr45]).isdisjoint(set(groups[te45]))))
    missing = [h for h in HOLDOUT_H if h not in sdk_names]
    if missing:
        raise SystemExit(f'严格匹配图中缺少预期留出 SDK: {missing}')
    H_idx = [sdk_names.index(h) for h in HOLDOUT_H]
    ftr, fte = split_by_sdk_holdout(sdk_mat, H_idx, groups)
    usage_cap = int(max(usage[H_idx]))
    checks.append((f'flagged holdout ({len(ftr)},{len(fte)}) 互斥补全',
                   len(ftr) + len(fte) == n_app and set(ftr).isdisjoint(fte)))
    # After filtering difference relations, held-out SDKs must have no training edges.
    ds_filtered = filter_diff_sdks(diff_sdks, H_idx, sdk_names)
    n_t, e_t = build_subgraph([folders[i] for i in ftr], cache, sdk_names, api_names,
                              perm_names, W, ds_filtered)
    n_f, e_f = build_subgraph(folders, cache, sdk_names, api_names, perm_names, W,
                              diff_sdks)
    n_t_full, e_t_full = build_subgraph([folders[i] for i in ftr], cache, sdk_names,
                                        api_names, perm_names, W, diff_sdks)
    checks.append(('训练图 n = len(train)+n_nonapp', n_t == len(ftr) + n_nonapp))
    checks.append(('全图 n = n_app+n_nonapp', n_f == n_app + n_nonapp))
    checks.append(('H 训练图零边(含差分边剔除)', verify_holdout_zero_edges(e_t, H_idx,
                                                                n_t - n_nonapp)))
    h_nodes = {n_t_full - n_nonapp + i for i in H_idx}
    n_h_diff = sum(1 for s, d, t, _w in e_t_full
                   if t in ('app-removed-sdk', 'app-added-sdk')
                   and d in h_nodes and s < n_t_full - n_nonapp)
    print(f'[verify] 差分边剔除: 全量建图触及 H 的差分边 {n_h_diff} 条(剔除后 0)')
    ctrl_H, ctrl_tr_n, attempt, win_lo, win_hi = random_control_H(
        sdk_names, usage, flagged_set, sdk_mat, groups, control_seed, usage_cap, len(ftr))
    ctr, _cte = split_by_sdk_holdout(sdk_mat, ctrl_H, groups)
    ds_ctrl = filter_diff_sdks(diff_sdks, ctrl_H, sdk_names)
    n_c, e_c = build_subgraph([folders[i] for i in ctr], cache, sdk_names, api_names,
                              perm_names, W, ds_ctrl)
    checks.append((f'control usage≤{usage_cap}, train∈[{win_lo},{win_hi}] (={ctrl_tr_n})',
                   win_lo <= len(ctr) <= win_hi
                   and all(usage[i] <= usage_cap for i in ctrl_H)))
    checks.append(('control 训练图零边(含差分边剔除)', verify_holdout_zero_edges(
        e_c, ctrl_H, n_c - n_nonapp)))
    return checks


# --------------------------------------------------------------------------
# Main workflow.
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoffs', default='41,45')
    ap.add_argument('--seeds', default='1000,2000,3000')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--lr', type=float, default=LR)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--holdout-sdks', default=None, help='Comma-separated SDK identifiers; defaults to prespecified H')
    ap.add_argument('--control-seed', type=int, default=CONTROL_SEED)
    ap.add_argument('--quick', action='store_true', help='Smoke test: one seed, 60 epochs')
    ap.add_argument('--verify', action='store_true', help='Run verification checks and exit')
    ap.add_argument('--rebuild-cache', action='store_true')
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))

    seeds = [int(s) for s in args.seeds.split(',')]
    cutoffs = [int(c) for c in args.cutoffs.split(',')]
    if args.quick:
        seeds, args.epochs, cutoffs = seeds[:1], 60, [PRIMARY_CUTOFF]

    (df, folders, static_cols, diff_cols, sdk_names, api_names, perm_names, Y,
     cache, diff_sdks) = load_data(args.rebuild_cache)
    n_app, n_nonapp, n_sdk = len(folders), len(sdk_names) + len(perm_names) + len(api_names), \
        len(sdk_names)
    feat_all = df[static_cols + diff_cols].to_numpy(float)
    sdk_mat = df[[f'sdk_{n}' for n in sdk_names]].to_numpy(float)
    usage = sdk_mat.sum(0)
    groups = df['app_group'].astype(str).to_numpy()
    flagged_set = {s['name'] for s in load_signatures() if s.get('miit_flagged')}
    print(f'数据: {n_app} 事件节点 / 非事件节点 {n_nonapp} / 特征 {feat_all.shape[1]}')
    assert np.isfinite(feat_all).all()

    if args.verify:
        checks = run_verify(df, sdk_mat, sdk_names, usage, flagged_set, folders, cache,
                            api_names, perm_names, n_nonapp, args.control_seed, diff_sdks)
        for name, ok in checks:
            print(f'[verify] {"OK" if ok else "FAIL"} {name}')
        print('全部通过 [OK]' if all(ok for _n, ok in checks) else '存在失败 [FAIL]')
        return

    # Experiment 1: cross-batch temporal evaluation.
    t0 = time.time()
    exp1 = []        # dict: model, cutoff, macros, per_class, train_n, test_n
    for cutoff in cutoffs:
        tr, te = split_by_batch(df, cutoff)
        Y_tr, Y_te = Y[tr], Y[te]
        for model in CROSSBATCH_MODELS:
            if model == 'gnn_full':
                macros, pc, _ = run_gnn(feat_all, tr, te, Y, folders, cache, sdk_names,
                                        api_names, perm_names, seeds, n_nonapp,
                                        args.epochs, args.lr, diff_sdks=diff_sdks)
            else:
                use = static_cols if model == 'lr_static' else static_cols + diff_cols
                m, pc = lr_result(feat_all, static_cols, diff_cols, use, tr, te, Y)
                macros = np.array([m])   # Deterministic LR needs only one fit per split.
            exp1.append({'model': model, 'cutoff': cutoff, 'macros': macros,
                         'per_class': pc, 'train_n': len(tr), 'test_n': len(te)})
            mm = macros.mean()
            print(f'[exp1] cutoff={cutoff} {model:<12} train={len(tr)} test={len(te)} '
                  f'macro={mm:.3f} ({time.time()-t0:.0f}s)')

    # Label-distribution context.
    bn = parse_batch_num(df)
    early = np.where(bn <= 40)[0]
    late = np.where(bn > 40)[0]
    label_shift = {'early_n': len(early), 'late_n': len(late),
                   'early_rate': Y[early].mean(0), 'late_rate': Y[late].mean(0)}

    # ---------------- Exp2: SDK holdout ----------------
    H_names = ([h.strip() for h in args.holdout_sdks.split(',') if h.strip()]
               if args.holdout_sdks else HOLDOUT_H)
    missing = [h for h in H_names if h not in sdk_names]
    if missing:
        raise SystemExit(f'严格匹配图中不存在留出 SDK: {missing}')
    H_idx = [sdk_names.index(h) for h in H_names]
    ftr, fte = split_by_sdk_holdout(sdk_mat, H_idx, groups)
    usage_cap = int(max(usage[H_idx]))
    ctrl_H, ctrl_tr_n, attempt, ctrl_win_lo, ctrl_win_hi = random_control_H(
        sdk_names, usage, flagged_set, sdk_mat, groups, args.control_seed,
        usage_cap, len(ftr))
    ctr, cte = split_by_sdk_holdout(sdk_mat, ctrl_H, groups)

    settings = []
    for sname, tr, te, hd in [('flagged', ftr, fte, H_idx), ('control', ctr, cte, ctrl_H)]:
        ds_train = filter_diff_sdks(diff_sdks, hd, sdk_names)
        gm, gpc, _ = run_gnn(feat_all, tr, te, Y, folders, cache, sdk_names, api_names,
                             perm_names, seeds, n_nonapp, args.epochs, args.lr,
                             diff_sdks=diff_sdks, diff_sdks_train=ds_train)
        lm, lpc = lr_result(feat_all, static_cols, diff_cols, static_cols + diff_cols,
                            tr, te, Y)
        settings.append({'name': sname, 'train_n': len(tr), 'test_n': len(te),
                         'gnn_macros': gm, 'gnn_pc': gpc, 'lr_macro': lm,
                         'H_names': [sdk_names[i] for i in hd]})
        print(f'[exp2] {sname}: train={len(tr)} test={len(te)} '
              f'gnn_macro={gm.mean():.3f} lr_macro={lm:.3f}')

    write_outputs(args, exp1, label_shift, settings, attempt, sdk_names, usage, Y, bn,
                  usage_cap, ctrl_win_lo, ctrl_win_hi)
    print(f'完成: {OUT_REPORT} + {OUT_XLSX}')


# --------------------------------------------------------------------------
# Output generation.
# --------------------------------------------------------------------------

def mean_std(xs):
    xs = np.asarray(xs, dtype=float)
    return f'{xs.mean():.3f}±{xs.std():.3f}'


def pc_str(pc):
    return ' '.join(f'{CLASSES[c]}={np.mean(v):.2f}' for c, v in sorted(pc.items()))


def write_outputs(args, exp1, label_shift, settings, attempt, sdk_names, usage, Y, bn,
                  usage_cap, ctrl_win_lo, ctrl_win_hi):
    os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)

    # ---- xlsx ----
    with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as w:
        rows = []
        for e in exp1:
            n_cls = len(e['per_class'])
            rows.append({'model': e['model'], 'cutoff': e['cutoff'],
                         'train_n': e['train_n'], 'test_n': e['test_n'],
                         'macro_mean': e['macros'].mean(), 'macro_std': e['macros'].std(),
                         'n_classes': n_cls,
                         'per_class': pc_str(e['per_class'])})
        pd.DataFrame(rows).to_excel(w, sheet_name='cross_batch', index=False)
        pd.DataFrame([{'period': 'early(13-40)', 'n': label_shift['early_n'],
                       **{CLASSES[c]: label_shift['early_rate'][c] for c in range(6)}},
                      {'period': 'late(41-53)', 'n': label_shift['late_n'],
                       **{CLASSES[c]: label_shift['late_rate'][c] for c in range(6)}}]
                     ).to_excel(w, sheet_name='label_shift', index=False)
        hd_rows = []
        for s in settings:
            hd_rows.append({'setting': s['name'], 'train_n': s['train_n'],
                            'test_n': s['test_n'], 'gnn_macro_mean': s['gnn_macros'].mean(),
                            'gnn_macro_std': s['gnn_macros'].std(),
                            'lr_macro': s['lr_macro'],
                            'gnn_minus_lr': s['gnn_macros'].mean() - s['lr_macro'],
                            'n_classes': len(s['gnn_pc']),
                            'H_sdks': '；'.join(s['H_names']),
                            'note': '单次探索性敏感性分析；两组测试集不可直接作因果比较'})
        pd.DataFrame(hd_rows).to_excel(w, sheet_name='sdk_holdout', index=False)

    # Markdown report.
    L = []
    A = L.append
    A('# Generalization: cross-batch temporal evaluation and SDK holdout')
    A('')
    holdout_n = len(next(s for s in settings if s['name'] == 'flagged')['H_names'])
    A(f'- Fixed protocol: primary split batch≤{PRIMARY_CUTOFF} for training / >{PRIMARY_CUTOFF} for testing; '
      f'standardization and SDK co-occurrence weights are estimated from training events only; '
      f'H contains {holdout_n} directly matched SDKs from the auxiliary list, excluding Zxing; '
      f'the control draws {CONTROL_K} SDKs from nonlisted candidates with usage≤{usage_cap}, accepting the first '
      f'train∈[{ctrl_win_lo},{ctrl_win_hi}] match (seed {args.control_seed}); '
      'SDK holdout is a single exploratory sensitivity analysis, and all OOD splits isolate app_group.')
    A('- Leakage controls: estimate normalization and SDK co-occurrence only from training events; '
      'attach unlabeled test events at inference with frozen co-occurrence; exclude flagged/miit features; '
      'keep application groups disjoint; purge earlier events of test applications in temporal splits; '
      'assign complete groups in SDK holdout. '
      'Remediation differences are retrospective evidence.')
    A('')
    A('## Experiment 1: cross-batch temporal evaluation')
    A('')
    for cutoff in sorted({e['cutoff'] for e in exp1}):
        rows = [e for e in exp1 if e['cutoff'] == cutoff]
        trn = rows[0]['train_n']
        A(f'### X={cutoff} (training batch≤{cutoff}, n={trn} / test n={rows[0]["test_n"]})')
        A('')
        A('| model | macro-AUC | n_classes | per-class |')
        A('|---|---|---|---|')
        for e in rows:
            A(f"| {e['model']} | {mean_std(e['macros'])} | {len(e['per_class'])} "
              f"| {pc_str(e['per_class'])} |")
        A('')
    A('### Label-distribution context for temporal evaluation')
    A('')
    A('| period | n | C1 | C2 | C3 | C4 | C5 | C6 |')
    A('|---|---|---|---|---|---|---|---|')
    for period, n, rate in (('early(13-40)', label_shift['early_n'], label_shift['early_rate']),
                            ('late(41-53)', label_shift['late_n'], label_shift['late_rate'])):
        A(f'| {period} | {n} | ' + ' | '.join(f'{rate[c]:.2f}' for c in range(6)) + ' |')
    A('')
    A('## Experiment 2: SDK holdout')
    A('')
    A('| setting | train_n | test_n | gnn macro-AUC | lr macro | GNN−LR | H SDKs |')
    A('|---|---|---|---|---|---|---|')
    for s in settings:
        A(f"| {s['name']} | {s['train_n']} | {s['test_n']} | {mean_std(s['gnn_macros'])} "
          f"| {s['lr_macro']:.3f} | {s['gnn_macros'].mean()-s['lr_macro']:+.3f} "
          f"| {'、'.join(s['H_names'])} |")
    A('')
    A('## Interpretation and limitations')
    A('')
    flagged = next(s for s in settings if s['name'] == 'flagged')
    if len(flagged['gnn_pc']) < N_CLASSES:
        A(f'- In the flagged setting, {N_CLASSES-len(flagged["gnn_pc"])} categories were excluded by '
          'auc_pair because of degenerate test labels or predictions; n_classes reports the evaluated count.')
    A('- C4/C5/C6 have <12 test positives and are interpreted descriptively; the standard deviation across 3 seeds measures seed variability only.')
    A('- All z-score statistics and inference-time SDK co-occurrence weights are estimated from training events only.')
    A('- LR is deterministic and fitted once; GNN results are mean±std across 3 seeds. '
      'Held-out SDK columns are zero throughout LR training, so their coefficients cannot learn from SDK-presence variation.')
    A('- Only one training-size-matched random control was drawn. Label composition and difficulty may differ '
      'between test sets, so the comparison does not establish dependence on risk-SDK identity.')
    A('- Generalization is limited by the absence of verified clean controls (unreported does not mean clean), '
      'static features, a single notice-source system, and retrospective differences.')
    with open(OUT_REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
