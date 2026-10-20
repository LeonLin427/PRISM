# -*- coding: utf-8 -*-
"""Generate the English result figures 3--7.

Read remediation and attribution artifacts directly and use the model/OOD
summaries in out/ for comparisons. Figures use 300 dpi, a serif font with
Times New Roman preferred, and blue/orange/neutral-gray colors.

Output: figures/fig{3..7}_*.png."""
import argparse
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from violation_taxonomy import CLASS_NAMES

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'axes.unicode_minus': False,
    'font.size': 8.5,
    'axes.labelsize': 9.5,
    'axes.titlesize': 9.5,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333',
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'legend.fontsize': 8, 'legend.frameon': False,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

BLUE, GRAY, LIGHT_GRAY, ORANGE = '#4C72B0', '#8C8C8C', '#C7C7C7', '#DD8452'
SHORT = {'C1': 'C1 Collection', 'C2': 'C2 Permission', 'C3': 'C3 Misleading',
         'C4': 'C4 Auto-start', 'C5': 'C5 Misuse', 'C6': 'C6 Other'}
KEY = lambda c: c[:2]  # CLASS_NAMES contains the category code followed by its original-language description.
SDK_EN = {  # English display names for SDKs appearing in the top-three rankings.
    '阿里云 SDK': 'Aliyun SDK', '魅族联运 SDK': 'Meizu Joint-Op SDK',
    '魅族推送': 'Meizu Push', '高德地图/定位': 'AMap/Location',
    '360 手机卫士/组件': '360 Mobile Security', '优量汇(广点通)': 'Youlianghui (GDT)',
    '抖音开放 SDK': 'Douyin Open SDK', '字节 JProtect 加固': 'ByteDance JProtect',
    '微博 SDK': 'Weibo SDK', 'vivo 推送': 'vivo Push', 'vivo 广告': 'vivo Ads',
    '友盟 U-Verify 一键登录': 'Umeng U-Verify', '友盟 U-Share/分享': 'Umeng U-Share',
    '友盟+ U-App/组件': 'Umeng U-App', 'OPPO 广告': 'OPPO Ads',
    'ButterKnife 注解库': 'ButterKnife', 'Lottie 动画库': 'Lottie',
    'Firebase': 'Firebase', '个推': 'GeTui', '热云 TrackingIO': 'TrackingIO',
    'AndroidUtilCode': 'AndroidUtilCode', '高德地图/定位': 'AMap/Location',
    '小米推送 MiPush': 'MiPush', '头条/巨量日志': 'Toutiao/BCLog',
    '微博 SDK': 'Weibo SDK', '极光 JPush/JAnalytics': 'JPush/JAnalytics',
    'PushSDK': 'PushSDK', '电信天翼账号': 'ChinaTelecom SSO',
    '荣耀推送': 'Honor Push', '优量汇(广点通)': 'Youlianghui (GDT)',
    '字节 JProtect 加固': 'ByteDance JProtect',
    'UCrop 图片裁剪': 'UCrop', '穿山甲/巨量引擎': 'Pangle/ByteDance Ads',
    '字节直播 SDK': 'ByteDance Live SDK', 'Guava': 'Guava',
    '阿里百川/mtop': 'Alibaba Baichuan/mtop',
}

DISPLAY = {
    'gnn_full': 'GNN (feat+graph)', 'gnn_nofeat': 'GNN (graph only)',
    'gnn_nosdk': 'GNN (no sdk-sdk)', 'gnn_sdkw2': 'GNN (w≥2)',
    'mlp': 'MLP (feat only)', 'lr_static': 'LR (static)',
    'lr_staticdiff': 'LR (static+diff)', 'knn_jaccard': 'kNN (Jaccard)',
}
COLORS = {
    'gnn_full': BLUE, 'gnn_nofeat': BLUE, 'gnn_nosdk': BLUE,
    'gnn_sdkw2': BLUE, 'mlp': GRAY, 'lr_static': ORANGE,
    'lr_staticdiff': ORANGE, 'knn_jaccard': LIGHT_GRAY,
}


def load_gnn_main_results():
    """Prefer full-precision persisted results; support older Markdown reports."""
    summary_path = os.path.join(BASE, 'out', 'gnn_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path, encoding='utf-8') as f:
            models = json.load(f)['models']
        return [{'key': key, 'name': DISPLAY[key], 'point': models[key]['oof_auc'],
                 'lo': models[key]['ci'][0], 'hi': models[key]['ci'][1],
                 'perclass': [models[key]['class_auc'][f'C{i}'] for i in range(1, 7)]}
                for key in DISPLAY]
    path = os.path.join(BASE, 'out', 'gnn_report.md')
    text = open(path, encoding='utf-8').read()
    block = text.split('## 主结果', 1)[1].split('## 消融', 1)[0]
    out = []
    for line in block.splitlines():
        if not line.startswith('|') or line.startswith('|---') or '变体' in line:
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) < 11 or cells[0] not in DISPLAY:
            continue
        ci = [float(x) for x in re.findall(r'[-+]?\d+\.\d+', cells[3])]
        out.append({
            'key': cells[0], 'name': DISPLAY[cells[0]],
            'point': float(cells[2]), 'lo': ci[0], 'hi': ci[1],
            'perclass': [float(x) for x in cells[5:11]],
        })
    if len(out) != 8:
        raise ValueError(f'expected 8 model rows in {path}, got {len(out)}')
    return out


# Figure 3: six-panel KDE comparison of remediation differences.
def fig3_kde():
    df = pd.read_csv(os.path.join(BASE, 'remediation.csv'), encoding='utf-8-sig')
    feat = 'n_added_components'
    from collections import defaultdict
    class_col = defaultdict(lambda: pd.Series(False, index=df.index))
    for i, row in df.iterrows():
        for c in str(row.get('violation_classes', '')).split('；'):
            if c in CLASS_NAMES:
                class_col[c].loc[i] = True
    vals = df[feat].dropna()
    clip = max(np.percentile(vals, 99), 1)
    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.2))
    for ax, c in zip(axes.flat, CLASS_NAMES):
        x_in = df[class_col[c]][feat].dropna().clip(upper=clip)
        x_out = df[~class_col[c]][feat].dropna().clip(upper=clip)
        bins = np.linspace(0, clip, 25)
        ax.hist(x_out, bins=bins, density=True, alpha=0.42, color=GRAY,
                edgecolor='white', linewidth=0.35, label='Other events')
        ax.hist(x_in, bins=bins, density=True, alpha=0.55, color=BLUE,
                edgecolor='white', linewidth=0.35, label='Events with class')
        for x, colr in ((x_out, GRAY), (x_in, BLUE)):
            if x.nunique() >= 3:
                xx = np.linspace(0, clip, 200)
                ax.plot(xx, stats.gaussian_kde(x)(xx), color=colr, lw=1.2)
        ax.set_title(f'{SHORT[KEY(c)]}  (n={len(x_in)})')
        if ax is axes.flat[-1]:
            ax.legend()
    for ax in axes.flat:
        ax.set_xlabel('Added components')
        ax.set_ylabel('Density')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_kde_added_components.png'))
    plt.close(fig)


# Figure 4: strict OOF macro-AUC for eight model variants.
# Bars show AUC of repeat-averaged OOF predictions; intervals are application-group
# bootstrap 95% CIs. The 45 fitted instances are not independent observations.


def fig4_macro_auc():
    variants = load_gnn_main_results()
    # A near-square single-column aspect keeps Figure 4 visually balanced with
    # the adjacent heatmap and avoids a large blank band below the shorter float.
    fig, ax = plt.subplots(figsize=(5.8, 3.75))
    x = np.arange(len(variants))
    for i, row in enumerate(variants):
        name, point, lo, hi = row['name'], row['point'], row['lo'], row['hi']
        colr = COLORS[row['key']]
        yerr = np.array([[point - lo], [hi - point]])
        ax.bar(i, point, yerr=yerr, capsize=2.8, color=colr,
               edgecolor='#FFFFFF', linewidth=0.7,
               error_kw={'lw': 0.9, 'ecolor': '#333333'})
        ax.text(i, hi + 0.006, f'{point:.3f}', ha='center', va='bottom',
                fontsize=7.2, color='#333333')
    ax.axhline(0.5, ls='--', lw=0.8, color='#555555')
    ax.text(1.005, 0.5, 'Chance', ha='left', va='center', color='#555555',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels([v['name'] for v in variants], rotation=32, ha='right')
    ax.set_ylabel('macro-AUC')
    ax.set_ylim(0.47, 0.69)
    ax.set_title('Strict grouped OOF macro-AUC with app-group bootstrap')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_macro_auc_bars.png'))
    plt.close(fig)


# Figure 5: model-by-category OOF AUC heatmap.
def fig5_heatmap():
    results = load_gnn_main_results()
    names = [r['name'] for r in results]
    M = np.array([r['perclass'] for r in results])
    # A taller single-column canvas keeps the 8x6 matrix readable after the
    # journal template scales it to column width and avoids a flattened heatmap.
    fig, ax = plt.subplots(figsize=(5.6, 5.8))
    im = ax.imshow(M, cmap='cividis', vmin=0.35, vmax=0.90, aspect='auto')
    ax.set_xticks(range(6))
    ax.set_xticklabels([SHORT[KEY(c)] for c in CLASS_NAMES], rotation=20, ha='right')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(6):
            lum = (M[i, j] - 0.35) / (0.90 - 0.35)
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                    color='white' if lum < 0.55 else 'black', fontsize=7.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label('per-class AUC')
    ax.set_title('Per-class AUC of model variants on six violation categories')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig5_perclass_heatmap.png'))
    plt.close(fig)


# Figure 6: top-three SDKs in each category-specific ranking.
def fig6_risk_ranking():
    rk = pd.read_excel(os.path.join(BASE, 'out', 'sdk_attribution.xlsx'),
                       sheet_name='per_class_risk')
    rk = rk[rk['rank'] <= 3]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    yticks, ylabs, y = [], [], 0.0
    for c in CLASS_NAMES:
        sub = rk[rk['class'] == KEY(c)].sort_values('rank')  # The workbook uses short category codes.
        yticks.append(y); ylabs.append(SHORT[KEY(c)])  # Category-heading row without a bar.
        y += 0.9
        for _, r in sub.iterrows():
            colr = ORANGE if r['method'] == 'freqdiff' else BLUE
            ax.barh(y, r['score'], color=colr, edgecolor='white', height=0.75)
            ax.text(r['score'] + 0.008, y, f"{r['score']:.2f}",
                    va='center', fontsize=7.5)
            yticks.append(y); ylabs.append('  ' + SDK_EN.get(r['sdk'], r['sdk']))
            y += 1.0
        y += 0.55
    ax.invert_yaxis()
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabs)
    for t in ax.get_yticklabels():
        if t.get_text().strip() in SHORT.values():
            t.set_fontweight('bold')
    ax.set_xlabel('Risk score (gradient or freq-diff per stability rule)')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label='Gradient'),
                       Patch(color=ORANGE, label='Deterministic freq-diff')],
              loc='lower right')
    max_score = float(rk['score'].max())
    ax.set_xlim(0, max_score * 1.08)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig6_risk_ranking_top3.png'))
    plt.close(fig)


# Figure 7: temporal and SDK-holdout sensitivity panels.
# Both panels use grouped bars; compare models only within each split.
def fig7_generalization():
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.9, 3.25), sharey=True)
    xlsx = os.path.join(BASE, 'out', 'm5_results.xlsx')
    cross = pd.read_excel(xlsx, sheet_name='cross_batch')
    hold = pd.read_excel(xlsx, sheet_name='sdk_holdout')

    # (a) Temporal splits, with post-purging application-disjoint counts from OOD outputs.
    cutoffs = sorted(cross['cutoff'].unique())
    splits = []
    for c in cutoffs:
        row = cross[cross.cutoff == c].iloc[0]
        splits.append(f'X={c}\n{int(row.train_n)} / {int(row.test_n)}')
    x = np.arange(len(splits))
    width = 0.24
    def vals(model, col):
        return [float(cross[(cross.model == model) & (cross.cutoff == c)].iloc[0][col])
                for c in cutoffs]
    temporal = [
        ('GNN', vals('gnn_full', 'macro_mean'), vals('gnn_full', 'macro_std'), BLUE),
        ('LR (static)', vals('lr_static', 'macro_mean'), None, ORANGE),
        ('LR (static+diff)', vals('lr_staticdiff', 'macro_mean'), None, GRAY),
    ]
    for offset, (label, vals, errs, colr) in zip((-width, 0, width), temporal):
        bars = a.bar(x + offset, vals, width, yerr=errs, capsize=2.3,
                     color=colr, edgecolor='white', linewidth=0.6,
                     label=label, error_kw={'lw': 0.8, 'ecolor': '#333333'})
        label_errs = errs if errs is not None else [0.0] * len(vals)
        for bar, val, err in zip(bars, vals, label_errs):
            a.text(bar.get_x() + bar.get_width() / 2, val + err + 0.007,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=6.8)
    a.set_xticks(x)
    a.set_xticklabels(splits)
    a.set_ylabel('macro-AUC')
    a.set_title('(a) Cross-batch splits')
    a.legend(loc='upper left', fontsize=7.2, ncol=1)

    # (b) Listed-SDK and matched random holdouts use different test sets.
    holds = [
        f'Flagged H\n{int(hold.iloc[0].train_n)} / {int(hold.iloc[0].test_n)}',
        f'Matched control\n{int(hold.iloc[1].train_n)} / {int(hold.iloc[1].test_n)}',
    ]
    x2 = np.arange(len(holds))
    holdout = [
        ('GNN', hold['gnn_macro_mean'].astype(float).tolist(),
         hold['gnn_macro_std'].astype(float).tolist(), BLUE),
        ('LR (static+diff)', hold['lr_macro'].astype(float).tolist(), None, ORANGE),
    ]
    for offset, (label, vals, errs, colr) in zip((-width / 1.7, width / 1.7), holdout):
        bars = b.bar(x2 + offset, vals, width * 1.15, yerr=errs, capsize=2.3,
                     color=colr, edgecolor='white', linewidth=0.6,
                     label=label, error_kw={'lw': 0.8, 'ecolor': '#333333'})
        label_errs = errs if errs is not None else [0.0] * len(vals)
        for bar, val, err in zip(bars, vals, label_errs):
            b.text(bar.get_x() + bar.get_width() / 2, val + err + 0.007,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=6.8)
    b.set_xticks(x2)
    b.set_xticklabels(holds)
    b.set_title('(b) SDK holdout sensitivity')
    b.legend(loc='upper left', fontsize=7.2)

    for ax in (a, b):
        ax.axhline(0.5, ls='--', lw=0.8, color='#555555')
        ax.set_ylim(0.48, 0.72)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=1.3)
    fig.savefig(os.path.join(OUT, 'fig7_generalization.png'))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures', default='3,4,5,6,7',
                        help='Comma-separated figure numbers; e.g. 4,5 for CV updates')
    args = parser.parse_args()
    builders = {'3': fig3_kde, '4': fig4_macro_auc, '5': fig5_heatmap,
                '6': fig6_risk_ranking, '7': fig7_generalization}
    for key in args.figures.split(','):
        builders[key.strip()]()
    print('saved to', os.path.abspath(OUT))
