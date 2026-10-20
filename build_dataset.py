"""Build dataset.csv from retained features, labels, and SDK signatures.

Each row represents one APK. label=1 denotes an issue version and label=0 a
remediated version, as specified by problem_version_on_disk and fix_versions.
The group field identifies the application for grouped cross-validation.

Feature groups (excluding version number, file size, and timestamps):
- perm_*: dangerous-permission indicators and total/dangerous/custom counts.
- api_*: log1p-transformed sensitive method-name occurrence counts.
- sdk_*: signature indicators retained when present in at least five APKs.
- struct_*: native-library, component, domain, unmatched-prefix, and packing features.

Auxiliary miit_flagged annotations are for auditing/descriptive analysis only;
they must not enter issue/remediated classification features."""
import csv
import glob
import json
import math
import os
import re

import pandas as pd

from match_sdks import load_signatures, match_one

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(HERE, 'features')
LABELS_PATH = os.path.join(HERE, 'labels.csv')
OUT_CSV = os.path.join(HERE, 'dataset.csv')

APK_RE = re.compile(r'^(?P<name>.+?)_v(?P<ver>.+?)(?:_wdj\d+)?(?:\.json)?$')

# Dangerous-permission indicators relevant to privacy violations in the notices.
DANGEROUS_PERMS = [
    'READ_PHONE_STATE', 'READ_PHONE_NUMBERS', 'READ_CALL_LOG', 'CALL_PHONE',
    'READ_CONTACTS', 'WRITE_CONTACTS', 'GET_ACCOUNTS',
    'READ_SMS', 'RECEIVE_SMS', 'SEND_SMS',
    'RECORD_AUDIO', 'CAMERA',
    'ACCESS_FINE_LOCATION', 'ACCESS_COARSE_LOCATION', 'ACCESS_BACKGROUND_LOCATION',
    'READ_EXTERNAL_STORAGE', 'WRITE_EXTERNAL_STORAGE', 'MANAGE_EXTERNAL_STORAGE',
    'QUERY_ALL_PACKAGES', 'REQUEST_INSTALL_PACKAGES',
    'SYSTEM_ALERT_WINDOW', 'WRITE_SETTINGS',
    'BIND_ACCESSIBILITY_SERVICE', 'BODY_SENSORS', 'ACTIVITY_RECOGNITION',
]


def norm_ver(s):
    s = str(s or '').strip().lower()
    return re.sub(r'^v', '', s)


def load_label_sets():
    """Map each application folder to its issue-version and remediated-version sets."""
    probs, fixes = {}, {}
    with open(LABELS_PATH, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            folder = row['folder']
            probs.setdefault(folder, set()).add(norm_ver(row['problem_version_on_disk']))
            fixes.setdefault(folder, set()).update(
                norm_ver(v) for v in row['fix_versions'].split('；') if v)
    return probs, fixes


def short_perm(p):
    return p.rsplit('.', 1)[-1]


def feat_row(feat, sigs):
    perms = [short_perm(p) for p in feat.get('permissions', [])]
    perm_set = set(perms)
    row = {
        'perm_total': len(perms),
        'perm_dangerous': sum(1 for p in perms if p in DANGEROUS_PERMS),
        'perm_custom': sum(1 for p in feat.get('permissions', [])
                           if not p.startswith('android.permission.')),
    }
    for p in DANGEROUS_PERMS:
        row[f'perm_{p}'] = int(p in perm_set)

    for api, cnt in (feat.get('api_calls') or {}).items():
        row[f'api_{api}'] = math.log1p(cnt)

    hits = match_one(feat, sigs)
    hit_names = set()
    packed = 0
    for sig, _ in hits:
        hit_names.add(sig['name'])
        if '加固' in (sig.get('category') or ''):
            packed = 1

    # Unmatched prefixes may indicate SDKs outside the signature library.
    own = feat.get('package') or ''
    covered = {p for sig, _ in hits for p in sig.get('packages', [])}
    n_unknown = 0
    for pref in set(feat.get('prefix3', {})):
        if own and (pref == own or pref.startswith(own + '.') or own.startswith(pref + '.')):
            continue
        if not any(pref == c or pref.startswith(c + '.') or c.startswith(pref + '.')
                   for c in covered):
            n_unknown += 1

    row.update({
        'sdk_count': len(hit_names),
        'struct_packed': packed,
        'struct_unknown_prefix3': n_unknown,
        'struct_n_so': len(feat.get('so_libs', [])),
        'struct_n_components': sum(len(feat.get(k, []) or []) for k in
                                   ('activities', 'services', 'receivers', 'providers')),
        'struct_n_domains': len(feat.get('domains', [])),
        'struct_n_meta': len(feat.get('meta_keys', [])),
    })
    return row, hit_names


def main():
    sigs = load_signatures()
    probs, fixes = load_label_sets()

    rows = []
    skipped = []
    for fp in sorted(glob.glob(os.path.join(FEAT_DIR, '*.json'))):
        with open(fp, encoding='utf-8') as f:
            feat = json.load(f)
        folder = feat['folder']
        apk_base = os.path.splitext(feat['apk_file'])[0]
        ver = norm_ver(apk_base.split('_v', 1)[1].split('_wdj')[0] if '_v' in apk_base else '?')
        if folder in probs and ver in probs[folder]:
            label = 1
        elif folder in fixes and ver in fixes[folder]:
            label = 0
        else:
            skipped.append(f'{folder}/{feat["apk_file"]}')
            continue
        row, hit_names = feat_row(feat, sigs)
        row.update({'apk_file': feat['apk_file'], 'folder': folder,
                    'version': ver, 'label': label})
        row['_hits'] = hit_names  # Temporary SDK-name set, expanded into sdk_* columns below.
        rows.append(row)

    # SDK indicators: retain signatures observed in at least five APKs.
    from collections import Counter
    freq = Counter()
    for r in rows:
        freq.update(r['_hits'])
    keep_sdks = {name for name, c in freq.items() if c >= 5}
    for r in rows:
        for name in keep_sdks:
            r[f'sdk_{name}'] = int(name in r['_hits'])
        del r['_hits']

    df = pd.DataFrame(rows).fillna(0)
    id_cols = ['apk_file', 'folder', 'version', 'label']
    feat_cols = sorted(c for c in df.columns if c not in id_cols)
    df = df[id_cols + feat_cols]
    df.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    print(f'dataset.csv: {len(df)} 行 (问题版 {int(df.label.sum())}, 修复版 {int((1-df.label).sum())})')
    print(f'特征列 {len(feat_cols)}: perm_* {sum(c.startswith("perm_") for c in feat_cols)}, '
          f'api_* {sum(c.startswith("api_") for c in feat_cols)}, '
          f'sdk_* {sum(c.startswith("sdk_") for c in feat_cols)}, '
          f'struct_* {sum(c.startswith("struct_") for c in feat_cols)}')
    if skipped:
        print(f'跳过(版本不在标签里) {len(skipped)} 个: {skipped[:5]}')


if __name__ == '__main__':
    main()
