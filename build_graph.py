"""Build an event-SDK-permission-API heterogeneous graph in graph/.

Event nodes are identified by (folder, batch); app_group links events from the
same application. Other node vocabularies comprise SDKs matched in issue versions,
25 dangerous permissions, and 18 observed sensitive API method names.

Relations:
- app-uses-sdk: event -> SDK; log1p(number of issue APKs matching the SDK).
- app-requests-perm: event -> permission; unit weight for requested permissions.
- app-calls-api: event -> API; log1p(maximum issue-side occurrence count).
- sdk-sdk: both directions; number of events whose issue versions share the SDKs.
  At least one common event creates a co-occurrence pair. The default retains
  all observed pairs; sdk_coocc_min_w supports the co-occurrence ablation.
- app-removed-sdk: event -> SDK; unit weight for issue-only SDKs.
- app-added-sdk: event -> SDK; unit weight for remediated-only SDKs.

With A[src,dst] and A @ H, an event aggregates its target-node representations.
Relation arrows are storage orientations, not forward feature-flow directions.
Removal/addition edges encode retrospective remediation evidence and use distinct
relation transformations. Removals are therefore not independent ground truth
for validating attribution rankings.

Event features concatenate issue-side static features and remediation differences.
Binary static features use unions, inventory counts are recomputed from unions,
and API counts use issue-side maxima. Missing differences become zero, with
has_fix recording availability. C1--C6 labels are stored separately from inputs.

Auxiliary flagged/miit indicators never enter model features or relations.
The app_subset option retains only the specified events and recomputes SDK
co-occurrence on that subset. Application-group isolation is enforced by callers.

Outputs: nodes.csv, edges.csv, app_features.csv, graph_stats.txt."""
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict

import pandas as pd

from build_dataset import DANGEROUS_PERMS
from match_sdks import load_signatures, match_one

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(HERE, 'features')
LABELS_PATH = os.path.join(HERE, 'labels.csv')
REM_CSV = os.path.join(HERE, 'remediation.csv')
SIG_PATH = os.path.join(HERE, 'sdk_signatures.json')
GRAPH_DIR = os.path.join(HERE, 'graph')

# Node types.
N_APP, N_SDK, N_PERM, N_API = 'app', 'sdk', 'perm', 'api'


def split_removed(s):
    """Split the semicolon-delimited SDK-name lists stored in remediation.csv."""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [x.strip() for x in str(s).split('；') if x.strip()]


def norm_ver(s):
    s = str(s or '').strip().lower()
    return re.sub(r'^v', '', s)


def apk_version(apk_file):
    base = os.path.splitext(apk_file)[0]
    if '_v' not in base:
        return '?'
    return norm_ver(base.split('_v', 1)[1].split('_wdj')[0])


def load_label_sets():
    """Map (folder, batch) to issue/remediated version sets, classes, and names.
    Repeated notices for one application remain separate events."""
    info = defaultdict(lambda: {'problem': set(), 'fix': set(),
                                'classes': set(), 'names': set()})
    with open(LABELS_PATH, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            d = info[(r['folder'], r.get('batch', ''))]
            d['problem'].add(norm_ver(r.get('problem_version_on_disk') or r['problem_version']))
            d['fix'].update(norm_ver(v) for v in r['fix_versions'].split('；') if v)
            d['classes'].update(c for c in r['violation_classes'].split('；') if c)
            d['names'].add(r['app_name'])
    return info


def load_feats_by_folder():
    """folder -> {version: feat}"""
    feats = defaultdict(dict)
    for fp in sorted(glob.glob(os.path.join(FEAT_DIR, '*.json'))):
        with open(fp, encoding='utf-8') as f:
            feat = json.load(f)
        feats[feat['folder']][apk_version(feat['apk_file'])] = feat
    return feats


def app_static_feats(prob_feats, sigs, sdk_names, api_names):
    """Aggregate issue-version features for one event.
    Use unions for binary inventories and inventory counts, and maxima for API
    occurrences. Auxiliary flagged/miit annotations are excluded."""
    perms = set()
    api_count = defaultdict(int)
    hit_names, pack = set(), False
    pref3, own, covered = set(), None, set()
    for f in prob_feats:
        perms.update(f.get('permissions', []))
        for a, c in (f.get('api_calls') or {}).items():
            api_count[a] = max(api_count[a], c)
        for s, _ in match_one(f, sigs):
            hit_names.add(s['name'])
            if (s.get('category') or '') == '加固':
                pack = True
            covered.update(s.get('packages', []))
        pref3.update(f.get('prefix3', {}))
        own = own or f.get('package')

    short = {p.rsplit('.', 1)[-1] for p in perms}
    row = {
        'perm_total': len(short),
        'perm_dangerous': sum(1 for p in short if p in DANGEROUS_PERMS),
        'perm_custom': sum(1 for p in perms if not p.startswith('android.permission.')),
    }
    for p in DANGEROUS_PERMS:
        row[f'perm_{p}'] = int(p in short)
    for a in api_names:
        row[f'api_{a}'] = math.log1p(api_count.get(a, 0))
    for name in sdk_names:
        row[f'sdk_{name}'] = int(name in hit_names)
    # Unmatched-prefix counts use the same issue-side union logic as build_dataset.
    n_unknown = 0
    for pref in pref3:
        if own and (pref == own or pref.startswith(own + '.') or own.startswith(pref + '.')):
            continue
        if not any(pref == c or pref.startswith(c + '.') or c.startswith(pref + '.')
                   for c in covered):
            n_unknown += 1
    row.update({
        'struct_packed': int(pack),
        'struct_unknown_prefix3': n_unknown,
        'struct_n_so': max(len(f.get('so_libs', [])) for f in prob_feats),
        'struct_n_components': max(sum(len(f.get(k, []) or []) for k in
                                       ('activities', 'services', 'receivers', 'providers'))
                                   for f in prob_feats),
        'struct_n_domains': max(len(f.get('domains', [])) for f in prob_feats),
        'struct_n_meta': max(len(f.get('meta_keys', [])) for f in prob_feats),
    })
    return row


def app_edges(prob_feats, sigs, app_idx, sdk_ids, perm_ids, api_ids):
    """Build use-SDK, request-permission, and API relations from issue features.
    Return edges and per-SDK matching-APK counts for this event."""
    edges, sdk_hit_cnt = [], defaultdict(int)
    api_count = defaultdict(int)
    perms = set()
    for f in prob_feats:
        for s, _ in match_one(f, sigs):
            sdk_hit_cnt[s['name']] += 1
        for a, c in (f.get('api_calls') or {}).items():
            api_count[a] = max(api_count[a], c)
        perms.update(f.get('permissions', []))
    # Weight use-SDK edges by log1p of the number of matching issue APKs.
    for name, cnt in sdk_hit_cnt.items():
        edges.append((app_idx, sdk_ids[name], 'app-uses-sdk', math.log1p(cnt)))
    for p in sorted(perms):
        short = p.rsplit('.', 1)[-1]
        if short in perm_ids:
            edges.append((app_idx, perm_ids[short], 'app-requests-perm', 1.0))
    for a, c in api_count.items():
        if a in api_ids:
            edges.append((app_idx, api_ids[a], 'app-calls-api', math.log1p(c)))
    return edges, sdk_hit_cnt


def build_graph(app_subset=None, sdk_coocc_min_w=1):
    """Build the full graph or the event subset specified by app_subset.
    sdk_coocc_min_w is the minimum shared-event count for a co-occurrence edge;
    the default of 1 retains all observed pairs.
    Return nodes (id, type, name), edges (src, dst, relation, weight), and app_rows."""
    sigs = load_signatures()
    info = load_label_sets()
    feats = load_feats_by_folder()

    # Fixed vocabularies; subset mode reduces event nodes, not SDK/permission/API vocabularies.
    api_names = sorted({a for folder in feats for f in feats[folder].values()
                        for a in (f.get('api_calls') or {})})
    sdk_names = set()
    for (folder, batch), d in info.items():
        prob_vers = [v for v in d['problem'] if v in feats.get(folder, {})]
        for f in [feats[folder][v] for v in prob_vers]:
            for s, _ in match_one(f, sigs):
                sdk_names.add(s['name'])
    sdk_names = sorted(sdk_names)
    perm_names = list(DANGEROUS_PERMS)

    events = sorted(info.keys()) if app_subset is None else sorted(app_subset)
    # Event IDs are folder__batch; app_group=folder binds repeated events to one CV fold.
    n_app = len(events)
    n_sdk, n_perm, n_api = len(sdk_names), len(perm_names), len(api_names)
    app_ids = {ev: i for i, ev in enumerate(events)}
    sdk_ids = {n: n_app + i for i, n in enumerate(sdk_names)}
    perm_ids = {n: n_app + n_sdk + i for i, n in enumerate(perm_names)}
    api_ids = {n: n_app + n_sdk + n_perm + i for i, n in enumerate(api_names)}
    nodes = ([(i, N_APP, f'{folder}__{batch}') for i, (folder, batch) in enumerate(events)] +
             [(n_app + i, N_SDK, n) for i, n in enumerate(sdk_names)] +
             [(n_app + n_sdk + i, N_PERM, n) for i, n in enumerate(perm_names)] +
             [(n_app + n_sdk + n_perm + i, N_API, n) for i, n in enumerate(api_names)])

    # Read differences and align them by (folder, batch).
    rem = pd.read_csv(REM_CSV, encoding='utf-8-sig')
    rem_idx = {(r['folder'], str(r.get('batch', ''))): i for i, r in rem.iterrows()}
    diff_cols = [c for c in rem.columns if c.startswith(('n_removed', 'n_added', 'api_'))]

    edges, app_rows = [], []
    for (folder, batch) in events:
        d = info[(folder, batch)]
        prob_vers = [v for v in d['problem'] if v in feats.get(folder, {})]
        prob_feats = [feats[folder][v] for v in prob_vers]

        row = {'folder': f'{folder}__{batch}', 'app_group': folder, 'batch': batch,
               'app_name': '/'.join(sorted(d['names'])),
               'has_fix': int((folder, batch) in rem_idx and bool(rem.at[rem_idx[(folder, batch)], 'has_fix']))}
        # Multilabel targets are stored in separate columns.
        for c in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6'):
            row[f'label_{c}'] = int(any(x.startswith(c) for x in d['classes']))

        if prob_feats:
            row.update(app_static_feats(prob_feats, sigs, sdk_names, api_names))
            e, _ = app_edges(prob_feats, sigs, app_ids[(folder, batch)], sdk_ids, perm_ids, api_ids)
            edges.extend(e)
        else:
            # If issue features are unavailable, use zero features and no event relations.
            row.update({f'perm_{p}': 0 for p in DANGEROUS_PERMS})
            row.update({'perm_total': 0, 'perm_dangerous': 0, 'perm_custom': 0})
            row.update({f'api_{a}': 0.0 for a in api_names})
            row.update({f'sdk_{n}': 0 for n in sdk_names})
            row.update({'struct_packed': 0, 'struct_unknown_prefix3': 0, 'struct_n_so': 0,
                        'struct_n_components': 0, 'struct_n_domains': 0, 'struct_n_meta': 0})

        # Missing differences become zero; has_fix records whether paired evidence is available.
        if (folder, batch) in rem_idx:
            r = rem.iloc[rem_idx[(folder, batch)]]
            for c in diff_cols:
                v = r[c]
                row[f'd_{c}'] = 0.0 if pd.isna(v) else float(v)
            # Retrospective SDK removal/addition relations have unit weights.
            for sd in split_removed(r.get('removed_sdks')):
                if sd in sdk_ids:
                    edges.append((app_ids[(folder, batch)], sdk_ids[sd], 'app-removed-sdk', 1.0))
            for sd in split_removed(r.get('added_sdks')):
                if sd in sdk_ids:
                    edges.append((app_ids[(folder, batch)], sdk_ids[sd], 'app-added-sdk', 1.0))
        else:
            for c in diff_cols:
                row[f'd_{c}'] = 0.0
        app_rows.append(row)

    # SDK co-occurrence uses only issue versions from the current event subset.
    sdk_in_app = defaultdict(set)  # sdk_name -> {app_idx}
    for (folder, batch) in events:
        for v in info[(folder, batch)]['problem']:
            if v in feats.get(folder, {}):
                for s, _ in match_one(feats[folder][v], sigs):
                    sdk_in_app[s['name']].add(app_ids[(folder, batch)])
    sdk_list = list(sdk_names)
    for i in range(len(sdk_list)):
        a = sdk_list[i]
        if a not in sdk_ids:
            continue
        for b in sdk_list[i + 1:]:
            if b not in sdk_ids:
                continue
            w = len(sdk_in_app.get(a, set()) & sdk_in_app.get(b, set()))
            if w >= sdk_coocc_min_w:
                ia, ib = sdk_ids[a], sdk_ids[b]
                edges.append((ia, ib, 'sdk-sdk', float(w)))
                edges.append((ib, ia, 'sdk-sdk', float(w)))

    return {'nodes': nodes, 'edges': edges, 'app_rows': app_rows,
            'sdk_names': sdk_names, 'api_names': api_names}


def write_full():
    os.makedirs(GRAPH_DIR, exist_ok=True)
    g = build_graph()
    nodes, edges, app_rows = g['nodes'], g['edges'], g['app_rows']

    # nodes.csv
    with open(os.path.join(GRAPH_DIR, 'nodes.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['node_id', 'type', 'name'])
        w.writerows(nodes)
    # edges.csv
    with open(os.path.join(GRAPH_DIR, 'edges.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['src', 'dst', 'etype', 'weight'])
        w.writerows(edges)
    # app_features.csv contains metadata, targets, static features, and d_* differences.
    adf = pd.DataFrame(app_rows)
    adf.to_csv(os.path.join(GRAPH_DIR, 'app_features.csv'), index=False, encoding='utf-8-sig')

    # Graph statistics.
    n_app = sum(1 for _, t, _ in nodes if t == N_APP)
    n_sdk = sum(1 for _, t, _ in nodes if t == N_SDK)
    n_perm = sum(1 for _, t, _ in nodes if t == N_PERM)
    n_api = sum(1 for _, t, _ in nodes if t == N_API)
    from collections import Counter
    etype_cnt = Counter(e[2] for e in edges)
    lines = [
        'M2 异构图统计(全量)',
        f'节点: App {n_app} / SDK {n_sdk} / 权限 {n_perm} / API {n_api} = {len(nodes)}',
        f'边: ' + ', '.join(f'{k} {v}' for k, v in sorted(etype_cnt.items())) + f' = {len(edges)}',
        '',
    ]
    # Check whether any event has no incident relations.
    app_ids_ = {n for n, t, _ in nodes if t == N_APP}
    connected_apps = {s for s, _, _, _ in edges} | {d for _, d, _, _ in edges}
    iso_apps = app_ids_ - connected_apps
    lines.append(f'孤立 App(无边): {len(iso_apps)} {sorted(iso_apps) if len(iso_apps) <= 10 else ""}')
    # Average event degree, counting relations originating at event nodes.
    from collections import defaultdict as dd
    deg = dd(int)
    for s, d, t, _ in edges:
        if t in ('app-uses-sdk', 'app-requests-perm', 'app-calls-api',
                 'app-removed-sdk', 'app-added-sdk'):
            deg[s] += 1
    app_deg = [deg[i] for i in range(n_app)]
    lines.append(f'App 平均入/出度(use_sdk+req_perm+call_api+removed_sdk): 均值 {sum(app_deg)/n_app:.1f}, '
                 f'min {min(app_deg)}, max {max(app_deg)}')
    # SDK co-occurrence density as coverage of possible undirected pairs.
    sdk_edges = [e for e in edges if e[2] == 'sdk-sdk']
    undir = {tuple(sorted([e[0], e[1]])) for e in sdk_edges}
    total_pairs = n_sdk * (n_sdk - 1) // 2
    dens = len(undir) / total_pairs * 100 if total_pairs else 0
    lines.append(f'sdk-sdk 共现无向对 {len(undir)}/{total_pairs} = {dens:.1f}%'
                 f'(本域 SDK 构成同质化, 共现层天然稠密; M3 消融验证其价值)')
    stats = '\n'.join(lines)
    with open(os.path.join(GRAPH_DIR, 'graph_stats.txt'), 'w', encoding='utf-8') as f:
        f.write(stats)
    print(stats)
    # Feature-family summary and auxiliary-marker exclusion check.
    feat_cols = [c for c in adf.columns if not c.startswith(('label_', 'folder', 'app_name', 'batch', 'has_fix'))]
    n_perm = sum(c.startswith('perm_') for c in feat_cols)
    n_api = sum(c.startswith('api_') for c in feat_cols)
    n_sdk = sum(c.startswith('sdk_') for c in feat_cols)
    n_struct = sum(c.startswith('struct_') for c in feat_cols)
    n_diff = sum(c.startswith('d_') for c in feat_cols)
    leak_cols = [c for c in feat_cols if 'flagged' in c or 'miit' in c.lower()]
    print(f'\nApp 特征列 {len(feat_cols)}: perm_* {n_perm}, api_* {n_api}, sdk_* {n_sdk}, '
          f'struct_* {n_struct}, d_* 差分 {n_diff}')
    print(f'泄漏检查: 特征里含 flagged/miit 的列 -> {leak_cols if leak_cols else "无(通过)"}')
    print(f'\n输出 -> {GRAPH_DIR}/')


if __name__ == '__main__':
    write_full()
