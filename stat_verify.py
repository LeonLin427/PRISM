"""Test associations between remediation differences and violation categories.

For each nonconstant numeric feature/category pair, compare events with and
without the category using two-sided Mann-Whitney U, apply BH-FDR correction,
and report Cliff's delta. Effect sizes complement p-values; neither alone
establishes a stable or causal association.

Plot six-panel density comparisons for features with large absolute delta.
Outputs: out/stats/remediation_stats.csv and out/stats/*.png."""
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from violation_taxonomy import CLASS_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
REM_CSV = os.path.join(HERE, 'remediation.csv')
STATS_DIR = os.path.join(HERE, 'out', 'stats')

# Fonts for source-language diagnostic plots on Windows.
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Numeric remediation-difference columns.
NUM_FEATS = [f'n_removed_perms', 'n_added_perms', 'n_removed_sdks', 'n_added_sdks',
             'n_removed_domains', 'n_added_domains', 'n_removed_meta', 'n_added_meta',
             'n_removed_components', 'n_added_components']

# Source-language diagnostic titles; keep stable English column IDs in filenames.
NUM_FEATS_CN = {
    'n_removed_perms': '删除权限数', 'n_added_perms': '新增权限数',
    'n_removed_sdks': '删除SDK数', 'n_added_sdks': '新增SDK数',
    'n_removed_domains': '删除域名数', 'n_added_domains': '新增域名数',
    'n_removed_meta': '删除meta键数', 'n_added_meta': '新增meta键数',
    'n_removed_components': '删除组件数', 'n_added_components': '新增组件数',
}
API_CN = {
    'DexClassLoader': '动态加载类', 'addJavascriptInterface': 'JS桥接注入',
    'getCellLocation': '获取基站位置', 'getDeviceId': '获取设备ID',
    'getImei': '获取IMEI', 'getInstalledApplications': '枚举已装应用',
    'getInstalledPackages': '枚举已装应用包', 'getLastKnownLocation': '获取最后已知位置',
    'getLine1Number': '获取本机号码', 'getMacAddress': '获取MAC地址',
    'getMeid': '获取MEID', 'getPrimaryClip': '读取剪贴板',
    'getRunningAppProcesses': '获取运行进程', 'getRunningTasks': '获取运行任务',
    'getSerial': '获取设备序列号', 'getSimSerialNumber': '获取SIM卡序列号',
    'getSubscriberId': '获取运营商用户ID', 'setPrimaryClip': '写入剪贴板',
}


def feat_cn(feat):
    """Map feature identifiers to source-language diagnostic plot titles.
    Unknown identifiers are displayed unchanged."""
    if feat.startswith('api_'):
        name = API_CN.get(feat[4:], feat[4:])
        return f'敏感API引用数变化({name})'
    return NUM_FEATS_CN.get(feat, feat)


def bh_fdr(ps):
    """Apply Benjamini-Hochberg false-discovery-rate adjustment."""
    ps = np.asarray(ps, dtype=float)
    n = len(ps)
    order = np.argsort(ps)
    ranked = ps[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # Enforce monotonic adjusted p-values in ascending raw-p order.
    for i in range(n - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    out = np.empty_like(adj)
    out[order] = adj
    return np.clip(out, 0, 1)


def cliffs_delta(x, y):
    """Compute Cliff's delta as (number of x>y pairs minus number of x<y pairs)/(nx*ny).
    Positive values indicate larger values among events with the category."""
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return float('nan')
    x, y = np.asarray(x, float), np.asarray(y, float)
    gt = sum(xi > yj for xi in x for yj in y)
    lt = sum(xi < yj for xi in x for yj in y)
    return (gt - lt) / (nx * ny)


def main():
    os.makedirs(STATS_DIR, exist_ok=True)
    # Clear previously generated KDE PNGs because selected top features can change.
    # This prevents obsolete feature plots from being mixed with a new run.
    for old in glob.glob(os.path.join(STATS_DIR, 'kde_*.png')):
        os.remove(old)
    df = pd.read_csv(REM_CSV, encoding='utf-8-sig')
    df = df[df['has_fix'].fillna(False).astype(bool)].copy()  # Restrict comparisons to events with remediated versions.
    print(f'统计样本: {len(df)} 个通报事件(有修复版)')

    # Numeric columns, including api_* differences.
    api_cols = sorted(c for c in df.columns if c.startswith('api_'))
    feats = NUM_FEATS + api_cols

    # Binary membership indicator for each category.
    class_col = defaultdict(lambda: pd.Series(False, index=df.index))
    for i, row in df.iterrows():
        for c in str(row.get('violation_classes', '')).split('；'):
            if c:
                class_col[c].loc[i] = True

    test_rows = []
    for feat in feats:
        vals = df[feat]
        if vals.isna().all() or vals.nunique() <= 1:
            continue
        for cls in CLASS_NAMES:
            mask = class_col[cls]
            x = vals[mask].dropna()
            y = vals[~mask].dropna()
            if len(x) < 3 or len(y) < 3:
                continue
            try:
                u, p = stats.mannwhitneyu(x, y, alternative='two-sided')
            except ValueError:
                continue
            test_rows.append({
                'feature': feat, 'violation_class': cls,
                'n_with_class': len(x), 'n_without_class': len(y),
                'mean_with': x.mean(), 'mean_without': y.mean(),
                'mw_u': u, 'p_value': p,
                'cliffs_delta': cliffs_delta(x, y),
            })

    tdf = pd.DataFrame(test_rows)
    if tdf.empty:
        print('没有可检验的特征-类对!')
        return
    tdf['p_fdr'] = bh_fdr(tdf['p_value'].values)
    tdf = tdf.sort_values(['cliffs_delta'], key=abs, ascending=False)
    tdf.to_csv(os.path.join(STATS_DIR, 'remediation_stats.csv'), index=False,
               encoding='utf-8-sig')

    # Statistical summary.
    print(f'\n检验 {len(tdf)} 个 特征×类 对, 按 |Cliff δ| 排序 TOP 12:')
    print(f'{"特征":<26}{"类":<12}{"δ":>6}{"p":>8}{"p_fdr":>8}{"含类均值":>10}{"不含均值":>10}')
    for _, r in tdf.head(12).iterrows():
        print(f"{r['feature']:<26}{r['violation_class'][:8]:<12}{r['cliffs_delta']:>6.2f}"
              f"{r['p_value']:>8.3f}{r['p_fdr']:>8.3f}{r['mean_with']:>10.1f}{r['mean_without']:>10.1f}")

    # Significant feature/category pairs after correction.
    sig = tdf[tdf['p_fdr'] < 0.05]
    print(f'\nFDR<0.05 的显著对: {len(sig)} 个')
    for _, r in sig.head(10).iterrows():
        print(f"  {r['feature']:<26} {r['violation_class'][:8]:<10} δ={r['cliffs_delta']:.2f}")

    # Six-panel histograms and KDEs: one figure per feature and one panel per category.
    # Compare category-positive versus other events; clip display axes at the 99th percentile.
    top_feats = []
    for cls in CLASS_NAMES:
        sub = tdf[tdf['violation_class'] == cls]
        if not sub.empty:
            top_feats.append((cls, sub.iloc[0]['feature'], sub.iloc[0]['cliffs_delta'],
                              sub.iloc[0]['p_value']))
    seen = set()
    n_plot = 0
    for cls0, feat, delta, p0 in top_feats:
        if feat in seen:
            continue
        seen.add(feat)
        vals = df[feat].dropna()
        if vals.nunique() < 3:
            continue
        clip = max(np.percentile(vals, 99), 1)
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        for ax, c in zip(axes.flat, CLASS_NAMES):
            x_in = df[class_col[c]][feat].dropna()
            x_out = df[~class_col[c]][feat].dropna()
            if len(x_in) < 3 or len(x_out) < 3:
                ax.set_visible(False)
                continue
            bins = np.linspace(0, clip, 25)
            for x, lab, colr in ((x_out, '不含该类', '#999999'), (x_in, '含该类', '#d62728')):
                ax.hist(x.clip(upper=clip), bins=bins, density=True, alpha=0.4,
                        color=colr, label=lab)
                if x.clip(upper=clip).nunique() >= 3:
                    try:
                        xx = np.linspace(0, clip, 200)
                        ax.plot(xx, stats.gaussian_kde(x.clip(upper=clip))(xx),
                                color=colr, lw=1.5)
                    except Exception:
                        pass
            row = tdf[(tdf['feature'] == feat) & (tdf['violation_class'] == c)]
            d = row.iloc[0]['cliffs_delta'] if not row.empty else float('nan')
            ax.set_title(f'{c}\nn={int(class_col[c].sum())} δ={d:.2f}', fontsize=9)
            ax.legend(fontsize=7)
        fig.suptitle(f'{feat_cn(feat)}: 各类违规 App 的整改量分布(直方图+KDE, x 截断 p99={clip:.0f})',
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_png = os.path.join(STATS_DIR, f'kde_{feat}.png')
        fig.savefig(out_png, dpi=130)
        plt.close(fig)
        n_plot += 1
    print(f'\n直方图+KDE 小倍图 {n_plot} 张 -> {STATS_DIR}/')


if __name__ == '__main__':
    main()
