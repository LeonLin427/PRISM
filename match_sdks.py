"""Match retained APK features against SDK signatures.

Outputs: out/sdk_inventory.xlsx and out/unmatched_prefixes.csv.
A signature matches when any supported evidence rule matches; retain the rules
that support each identification.
- packages: an observed DEX prefix or full Manifest component equals the
  signature package or is a descendant; never infer a deeper SDK from a vendor root.
- so_libs: exact native-library basename match.
- meta_keys: exact Manifest metadata-key match.
- domains: exact domain or subdomain match.
- assets: asset-name evidence as defined by the signature matcher.

Only two-/three-level DEX prefixes remain in the retained corpus. Deeper SDK
packages therefore require full component names or other independent evidence.
This conservative strategy can miss SDKs without sufficiently specific evidence."""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR = os.path.join(HERE, 'features')
OUT_DIR = os.path.join(HERE, 'out')
SIG_PATH = os.path.join(HERE, 'sdk_signatures.json')
LABELS_PATH = os.path.join(HERE, 'labels.csv')

# Exclude framework, open-source, and system-noise prefixes from unmatched listings.
NOISE_PREFIX = re.compile(
    r'^(L|a\.a$|android\.|androidx\.|java\.|javax\.|kotlin|dalvik\.|'
    r'org\.(apache|jetbrains|json|xmlpull|w3c|xml|ietf|kxml|conscrypt|bouncycastle|'
    r'intellij|checkerframework|springframework|repackage|osgi|eclipse|objectweb|slf4j)|'
    r'com\.google\.(gson|zxing|android|common|mlkit|protobuf|errorprone|firebase|gms|'
    r'material|devtools|auto)|'
    r'io\.(flutter|grpc|reactivex|netty|opencensus)|okhttp3|okio|retrofit2|rx\.|'
    r'com\.(facebook|bumptech|blankj|squareup|jakewharton|github|android)|'
    r'com\.alibaba\.(fastjson|android)|org\.greenrobot|de\.greenrobot|kotlinx\.|sun\.|com\.sun\.|'
    r'com\.asus(?!\.msa)|com\.samsung|com\.huawei\.(android|secure|libcore)|pl\.droidsonroids|dagger\.|'
    r'com\.vivo\.vms|jp\.wasabeef|com\.chad|com\.yalantis|org\.(reactivestreams|glassfish|joda))')


def load_signatures():
    with open(SIG_PATH, encoding='utf-8') as f:
        sigs = json.load(f)
    return [s for s in sigs if s.get('name') and not s['name'].startswith('__')]


def comp_prefixes(feat):
    """Derive two-/three-level package prefixes from Manifest component names."""
    p2, p3 = set(), set()
    for key in ('activities', 'services', 'receivers', 'providers'):
        for cls in feat.get(key, []) or []:
            parts = cls.split('.')
            if len(parts) >= 2:
                p2.add('.'.join(parts[:2]))
            if len(parts) >= 3:
                p3.add('.'.join(parts[:3]))
    return p2, p3


def component_classes(feat):
    """Return full Manifest component names as specific evidence for deep packages."""
    classes = set()
    for key in ('activities', 'services', 'receivers', 'providers'):
        for cls in feat.get(key, []) or []:
            cls = str(cls).strip().strip('.')
            if cls:
                classes.add(cls)
    return classes


def match_one(feat, sigs):
    """Return matching signatures with their supporting evidence rules."""
    cp2, cp3 = comp_prefixes(feat)
    p2 = set(feat.get('prefix2', {})) | cp2
    p3 = set(feat.get('prefix3', {})) | cp3
    full_components = component_classes(feat)
    sos = set(feat.get('so_libs', []))
    metas = set(feat.get('meta_keys', []))
    domains = set(feat.get('domains', []))
    assets = set(feat.get('assets', []))
    hits = []
    for sig in sigs:
        rules = []
        for pkg in sig.get('packages', []):
            pkg = str(pkg).strip().strip('.')
            if not pkg:
                continue
            observed = next((pref for pref in p2 | p3
                             if pref == pkg or pref.startswith(pkg + '.')), None)
            if observed is not None:
                rules.append(f'pkg:{pkg}<=prefix:{observed}')
                continue
            observed = next((cls for cls in full_components
                             if cls == pkg or cls.startswith(pkg + '.')), None)
            if observed is not None:
                rules.append(f'pkg:{pkg}<=component:{observed}')
        for so in sig.get('so_libs', []):
            if so in sos:
                rules.append(f'so:{so}')
        for mk in sig.get('meta_keys', []):
            if mk in metas:
                rules.append(f'meta:{mk}')
        for dom in sig.get('domains', []):
            for d in domains:
                if d == dom or d.endswith('.' + dom):
                    rules.append(f'domain:{dom}')
                    break
        for a in sig.get('assets', []):
            if a in assets:
                rules.append(f'asset:{a}')
        if rules:
            hits.append((sig, rules))
    return hits


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sigs = load_signatures()
    labels = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                key = (row['folder'], row['problem_version'])
                labels.setdefault(row['folder'], []).append(row)

    rows = []          # Detailed APK-by-SDK inventory.
    app_summary = {}   # folder -> {ver: [sdk names]}
    unmatched = defaultdict(lambda: {'count': 0, 'apps': set()})  # Prefix-level occurrence statistics.

    feat_files = sorted(glob.glob(os.path.join(FEAT_DIR, '*.json')))
    print(f'{len(feat_files)} 个特征文件')
    for fp in feat_files:
        with open(fp, encoding='utf-8') as f:
            feat = json.load(f)
        folder = feat['folder']
        apk_base = os.path.splitext(feat['apk_file'])[0]
        # Parse the version between _v and _wdj in the filename.
        ver = apk_base.split('_v', 1)[1].split('_wdj')[0] if '_v' in apk_base else '?'

        hits = match_one(feat, sigs)
        hit_pkgs = set()
        for sig, rules in hits:
            hit_pkgs.update(sig.get('packages', []))
            rows.append({
                'app_folder': folder,
                'apk': feat['apk_file'],
                'version': ver,
                'sdk_name': sig['name'],
                'vendor': sig.get('vendor', ''),
                'category': sig.get('category', ''),
                'miit_flagged': sig.get('miit_flagged', False),
                'matched_rules': '; '.join(rules),
            })
        app_summary.setdefault(folder, {})[ver] = sorted({s['name'] for s, _ in hits})

        # Unmatched prefixes need at least two classes, no signature match, and no obvious app-package match.
        own_pkg = feat.get('package') or ''
        for pref, cnt in {**feat.get('prefix2', {}), **feat.get('prefix3', {})}.items():
            if cnt < 2 or NOISE_PREFIX.match(pref):
                continue
            if own_pkg and (pref.startswith(own_pkg) or own_pkg.startswith(pref)):
                continue
            covered = any(
                pref == p or pref.startswith(p + '.')
                for p in hit_pkgs
            )
            if not covered:
                unmatched[pref]['count'] += 1
                unmatched[pref]['apps'].add(folder)

    df = pd.DataFrame(rows)
    out_xlsx = os.path.join(OUT_DIR, 'sdk_inventory.xlsx')
    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='APK×SDK明细', index=False)
        # Summarize SDK sets by application and version.
        sum_rows = []
        for folder, vers in sorted(app_summary.items()):
            for ver, sdk_names in sorted(vers.items()):
                sum_rows.append({'app_folder': folder, 'version': ver,
                                 'sdk_count': len(sdk_names), 'sdks': ', '.join(sdk_names)})
        pd.DataFrame(sum_rows).to_excel(w, sheet_name='App版本汇总', index=False)
    print(f'sdk_inventory.xlsx: {len(rows)} 条命中, {len(app_summary)} 个 App')

    # Frequently observed unmatched prefixes.
    um_rows = [{'prefix': p, 'apk_count': v['count'],
                'example_apps': ', '.join(sorted(v['apps'])[:5])}
               for p, v in unmatched.items() if v['count'] >= 5]
    um_rows.sort(key=lambda x: -x['apk_count'])
    out_csv = os.path.join(OUT_DIR, 'unmatched_prefixes.csv')
    with open(out_csv, 'w', encoding='utf-8-sig', newline='') as f:
        wtr = csv.DictWriter(f, fieldnames=['prefix', 'apk_count', 'example_apps'])
        wtr.writeheader()
        wtr.writerows(um_rows)
    print(f'unmatched_prefixes.csv: {len(um_rows)} 个未识别高频前缀(≥5 APK)')
    for r in um_rows[:20]:
        print(f"  {r['prefix']:<40} {r['apk_count']:>4}  {r['example_apps'][:60]}")


if __name__ == '__main__':
    main()
