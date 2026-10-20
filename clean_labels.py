"""Normalize violation_types in labels.csv and populate violation_classes.

Source notices contain whitespace variants, split phrases, and free-text suffixes.
Back up the input once, normalize and deduplicate descriptions, apply the ordered
violation_taxonomy rules, and write the category column. Matching uses normalized
strings without whitespace; source-language identifiers remain unchanged."""
import csv
import os
import re
import shutil
from collections import Counter
from datetime import date

from violation_taxonomy import CLASS_NAMES, classify

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS_PATH = os.path.join(HERE, 'labels.csv')
BACKUP_PATH = os.path.join(HERE, 'labels_raw_backup.csv')
DIST_MD_PATH = os.path.join(HERE, 'out', 'violation_class_distribution.md')


def write_distribution(rows):
    """Write category distributions and the description-to-category mapping."""
    os.makedirs(os.path.dirname(DIST_MD_PATH), exist_ok=True)
    cnt = Counter()
    type_cnt = Counter()
    for r in rows:
        for c in r['violation_classes'].split('；'):
            if c:
                cnt[c] += 1
        for t in r['violation_types'].split('；'):
            if t:
                type_cnt[t] += 1
    n_app = len({r['folder'] for r in rows})
    # Application-level counts deduplicate folders and union their event labels.
    app_classes = {}
    for r in rows:
        app_classes.setdefault(r['folder'], set()).update(
            c for c in r['violation_classes'].split('；') if c)
    app_cnt = Counter(c for s in app_classes.values() for c in s)

    lines = [
        '# 违规类型归并分布(清洗后)',
        '',
        f'- 生成时间: {date.today().isoformat()}, 来源: labels.csv({len(rows)} 行, {n_app} 个唯一 App)',
        '- 归并规则见 `violation_taxonomy.py`; 清洗逻辑见 `clean_labels.py`',
        '- **口径**: 行级 = 按通报条目计(同一 App 出现在多个批次会重复计数); '
        'App 级 = 按唯一 App 计(违规类取并集)。主要分析单位为通报事件；交叉验证按唯一 App 分组',
        '',
        '## 6 粗类分布(行级, 按通报条目计, 多标签)',
        '',
        '| 类 | 含义 | 正样本数 |',
        '|---|---|---|',
    ]
    for c in CLASS_NAMES:
        lines.append(f'| {c.split()[0]} | {c.split(" ", 1)[1]} | {cnt.get(c, 0)} |')
    lines += [
        '',
        '## 6 粗类分布(App 级, 唯一 App 计, 多标签)',
        '',
        '| 类 | 含义 | 正样本数 |',
        '|---|---|---|',
    ]
    for c in CLASS_NAMES:
        lines.append(f'| {c.split()[0]} | {c.split(" ", 1)[1]} | {app_cnt.get(c, 0)} |')
    lines += [
        '',
        f'## {len(type_cnt)} 种清洗后表述 → 6 类归并明细',
        '',
        '| 原始违规(清洗后) | 归入类 | 出现次数 |',
        '|---|---|---|',
    ]
    for t, c_ in sorted(type_cnt.items(), key=lambda kv: -kv[1]):
        lines.append(f'| {t} | {classify(t)} | {c_} |')
    with open(DIST_MD_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'分布表已保存 -> {os.path.basename(DIST_MD_PATH)}')


def clean_one(raw):
    """Return normalized violation descriptions and their category labels."""
    raw = str(raw or '')
    # Rejoin the over-requesting-permissions phrase split across source cells
    # before splitting descriptions, so its suffix is not treated as a new violation.
    raw = re.sub(r'(?i)APP\s*强制、频繁、过\s*；\s*度索取权限',
                 'APP强制、频繁、过度索取权限', raw)
    types, classes = [], []
    for part in raw.split('；'):
        p = re.sub(r'\s+', '', part).strip('。')
        p = re.sub(r'(?i)^app', 'APP', p)
        if not p:
            continue
        cls = classify(p)
        if p not in types:
            types.append(p)
        if cls != 'UNKNOWN' and cls not in classes:
            classes.append(cls)
    return types, classes


def main():
    if not os.path.exists(LABELS_PATH):
        raise SystemExit(f'找不到 {LABELS_PATH}')
    # Create the backup once; never overwrite an existing backup.
    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(LABELS_PATH, BACKUP_PATH)
        print(f'已备份原文件 -> {os.path.basename(BACKUP_PATH)}')

    with open(LABELS_PATH, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        if 'violation_classes' not in fields:
            fields = fields + ['violation_classes']
        rows = list(reader)

    unknown = set()
    n_changed = 0
    for r in rows:
        types, classes = clean_one(r.get('violation_types', ''))
        cleaned = '；'.join(types)
        if cleaned != (r.get('violation_types') or ''):
            n_changed += 1
        r['violation_types'] = cleaned
        r['violation_classes'] = '；'.join(classes)
        if r.get('violation_classes') == '' and r.get('violation_types'):
            unknown.add(r['violation_types'])

    with open(LABELS_PATH, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    write_distribution(rows)

    # Check category distributions.
    from collections import Counter
    cnt = Counter()
    for r in rows:
        for c in r['violation_classes'].split('；'):
            if c:
                cnt[c] += 1
    print(f'\nlabels.csv: {len(rows)} 行, 清洗改动 {n_changed} 行, 含违规类型 {sum(1 for r in rows if r["violation_types"])} 行')
    print('归并后违规类分布:')
    for c in CLASS_NAMES:
        print(f'  {cnt.get(c, 0):>4}  {c}')
    if unknown:
        print(f'\n[警告] 未能分类的违规串 {len(unknown)} 个: {sorted(unknown)[:10]}')
    else:
        print('\n全部违规串分类成功, 无 UNKNOWN')


if __name__ == '__main__':
    main()
