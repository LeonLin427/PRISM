"""Compute event-level remediation differences and write remediation.csv.

Each (folder, batch) pair is a separate event. Compare issue and remediated
versions for permissions, sensitive method-name occurrences, SDKs, domains,
metadata keys, and components. Repeated events from one application are grouped
by train_gnn during cross-validation.

Events without a remediated version receive has_fix=False and NaN differences,
not zero changes. packing_changed marks pairs whose visible inventories may be
affected by packing. Generated component stubs are conservatively filtered when
both package and class segments have the specified random-name patterns.

Inputs: retained features/*.json, labels.csv, and match_sdks signature matching."""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import pandas as pd

from match_sdks import load_signatures, match_one

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(HERE, 'features')
LABELS_PATH = os.path.join(HERE, 'labels.csv')
SIG_PATH = os.path.join(HERE, 'sdk_signatures.json')
OUT_CSV = os.path.join(HERE, 'remediation.csv')


def norm_ver(s):
    s = str(s or '').strip().lower()
    return re.sub(r'^v', '', s)


def apk_version(apk_file):
    base = os.path.splitext(apk_file)[0]
    if '_v' not in base:
        return '?'
    rest = base.split('_v', 1)[1]
    return norm_ver(rest.split('_wdj')[0])


def set_diff(problem, fix):
    """Return the removed and added sets."""
    return problem - fix, fix - problem


def api_deltas(prob_feats, fix_feats, api_names):
    """For each API, subtract the remediated-side maximum from the issue-side maximum.
    Output names use the api_ prefix expected by stat_verify.py."""
    d = {}
    for api in api_names:
        p = max(((f.get('api_calls') or {}).get(api, 0) for f in prob_feats), default=0)
        x = max(((f.get('api_calls') or {}).get(api, 0) for f in fix_feats), default=0)
        d[f'api_{api}'] = p - x
    return d


RANDOM_PACKAGE_SEGMENT = re.compile(r'^[a-z]{5,}\d{1,3}$')
RANDOM_COMPONENT_CLASS = re.compile(
    r'^[A-Z][a-z]{4,}\d{1,3}(?:Activity|Service|Receiver|Provider)$')


def is_generated_component_stub(name):
    """Identify generated Manifest stubs with both random package and class segments."""
    parts = str(name or '').split('.')
    return (len(parts) >= 2 and
            any(RANDOM_PACKAGE_SEGMENT.fullmatch(part) for part in parts[:-1]) and
            bool(RANDOM_COMPONENT_CLASS.fullmatch(parts[-1])))


def collect_sets(feats):
    """Aggregate permission, SDK, domain, metadata-key, and component sets."""
    perms, sdks, domains, metas = set(), set(), set(), set()
    comps = set()
    filtered_stubs = set()
    for f in feats:
        perms.update(f.get('permissions', []))
        domains.update(f.get('domains', []))
        metas.update(f.get('meta_keys', []))
        for key in ('activities', 'services', 'receivers', 'providers'):
            for component in f.get(key, []) or []:
                if is_generated_component_stub(component):
                    filtered_stubs.add(component)
                else:
                    comps.add(component)
    return perms, sdks, domains, metas, comps, filtered_stubs


def main():
    sigs = load_signatures()
    packing_sdks = {s['name'] for s in sigs if s.get('category') == '加固'}

    # folder -> {version: feat}
    feats = defaultdict(dict)
    for fp in sorted(glob.glob(os.path.join(FEAT_DIR, '*.json'))):
        with open(fp, encoding='utf-8') as f:
            feat = json.load(f)
        feats[feat['folder']][apk_version(feat['apk_file'])] = feat

    # One row per (folder, batch) event; repeated notices remain distinct.
    info = {}
    with open(LABELS_PATH, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            key = (row['folder'], row.get('batch', ''))
            d = info.setdefault(key, {'problem': set(), 'fix': set(), 'names': set(),
                                      'classes': set()})
            d['problem'].add(norm_ver(row.get('problem_version_on_disk') or row['problem_version']))
            d['fix'].update(norm_ver(v) for v in row['fix_versions'].split('；') if v)
            d['names'].add(row['app_name'])
            d['classes'].update(c for c in row['violation_classes'].split('；') if c)

    # Union of observed sensitive API names ensures a complete set of columns.
    api_names = sorted({a for f in feats.values() for ff in f.values()
                        for a in (ff.get('api_calls') or {})})

    rows = []
    for (folder, batch), d in sorted(info.items()):
        prob_vers = [v for v in d['problem'] if v in feats.get(folder, {})]
        fix_vers = [v for v in d['fix'] if v in feats.get(folder, {})]
        has_fix = len(fix_vers) > 0
        row = {
            'folder': folder,
            'batch': batch,
            'app_name': '/'.join(sorted(d['names'])),
            'violation_classes': '；'.join(sorted(d['classes'])),
            'has_fix': has_fix,
            'n_problem_vers': len(prob_vers),
            'n_fix_vers': len(fix_vers),
        }

        if not prob_vers or not has_fix:
            # Without a remediated version, represent missing differences with NaN.
            for col in ('n_removed_perms', 'n_added_perms', 'n_removed_sdks', 'n_added_sdks',
                        'n_removed_domains', 'n_added_domains', 'n_removed_meta', 'n_added_meta',
                        'n_removed_components', 'n_added_components'):
                row[col] = float('nan')
            for api in api_names:
                row[f'api_{api}'] = float('nan')
            row['packing_changed'] = float('nan')
            row['component_stubs_problem'] = float('nan')
            row['component_stubs_fix'] = float('nan')
            row['component_stub_filter_applied'] = False
            row['removed_perms'] = row['added_perms'] = row['removed_sdks'] = ''
            row['added_sdks'] = row['removed_domains'] = row['added_domains'] = ''
            rows.append(row)
            continue

        pf = [feats[folder][v] for v in prob_vers]
        xf = [feats[folder][v] for v in fix_vers]

        p_perms, p_sdks, p_doms, p_metas, p_comps, p_stubs = collect_sets(pf)
        x_perms, x_sdks, x_doms, x_metas, x_comps, x_stubs = collect_sets(xf)

        # SDK-name sets obtained by signature matching.
        p_sdks = {s['name'] for f in pf for s, _ in match_one(f, sigs)}
        x_sdks = {s['name'] for f in xf for s, _ in match_one(f, sigs)}

        rm_perms, add_perms = set_diff(p_perms, x_perms)
        rm_sdks, add_sdks = set_diff(p_sdks, x_sdks)
        rm_doms, add_doms = set_diff(p_doms, x_doms)
        rm_metas, add_metas = set_diff(p_metas, x_metas)
        rm_comps, add_comps = set_diff(p_comps, x_comps)

        pack_problem = bool(p_sdks & packing_sdks)
        pack_fix = bool(x_sdks & packing_sdks)

        row.update({
            'n_removed_perms': len(rm_perms), 'n_added_perms': len(add_perms),
            'n_removed_sdks': len(rm_sdks), 'n_added_sdks': len(add_sdks),
            'n_removed_domains': len(rm_doms), 'n_added_domains': len(add_doms),
            'n_removed_meta': len(rm_metas), 'n_added_meta': len(add_metas),
            'n_removed_components': len(rm_comps), 'n_added_components': len(add_comps),
            'packing_changed': pack_problem != pack_fix,
            'component_stubs_problem': len(p_stubs),
            'component_stubs_fix': len(x_stubs),
            'component_stub_filter_applied': bool(p_stubs or x_stubs),
            'removed_perms': '；'.join(sorted(rm_perms))[:500],
            'added_perms': '；'.join(sorted(add_perms))[:500],
            'removed_sdks': '；'.join(sorted(rm_sdks))[:500],
            'added_sdks': '；'.join(sorted(add_sdks))[:500],
            'removed_domains': '；'.join(sorted(rm_doms))[:500],
            'added_domains': '；'.join(sorted(add_doms))[:500],
        })
        row.update(api_deltas(pf, xf, api_names))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    n_fix = int(df['has_fix'].sum())
    n_miss = int((~df['has_fix']).sum())
    n_pack = int(df[df['has_fix']]['packing_changed'].sum()) if n_fix else 0
    print(f'remediation.csv: {len(df)} 个事件 (有修复版 {n_fix}, 缺失掩码 {n_miss}, 加固状态变化 {n_pack})')
    print('差分特征列(数值):')
    print('  ', ', '.join([c for c in df.columns if c.startswith(('n_', 'api_'))]))
    # Print three illustrative examples.
    sample = df[df['has_fix']].head(3)
    for _, r in sample.iterrows():
        print(f"  {r['app_name'][:12]:<14} 删SDK {r['n_removed_sdks']} 加SDK {r['n_added_sdks']} "
              f"删权限 {r['n_removed_perms']} 加权限 {r['n_added_perms']} "
              f"删域名 {r['n_removed_domains']} 加域名 {r['n_added_domains']}")


if __name__ == '__main__':
    main()
