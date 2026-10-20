"""Evaluate issue-versus-remediated binary classification.

Use logistic regression and histogram-based gradient boosting with five-fold
GroupKFold by application. Versions of the same application never cross the
training/test boundary. Outputs are out/ml_report.md and out/sdk_risk_ranking.xlsx.

Near-chance grouped-CV AUC indicates limited discrimination under these features
and this sample, not proof that the two version populations are identical."""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, 'dataset.csv')
OUT_DIR = os.path.join(HERE, 'out')

ID_COLS = ['apk_file', 'folder', 'version', 'label']
RANDOM_STATE = 42


def markdown_table(frame, index=False, floatfmt='.3f'):
    """Format a simple Markdown table without the optional tabulate dependency."""
    df = frame.copy()
    if index:
        df.insert(0, df.index.name or 'feature', df.index)
    headers = [str(c) for c in df.columns]
    rows = []
    for values in df.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                cells.append(format(float(value), floatfmt))
            else:
                cells.append(str(value))
        rows.append(cells)
    return '\n'.join([
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
        *('| ' + ' | '.join(row) + ' |' for row in rows),
    ])


def eval_model(name, model, X, y, groups, cv):
    proba = cross_val_predict(model, X, y, groups=groups, cv=cv, method='predict_proba')[:, 1]
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        'model': name,
        'AUC': roc_auc_score(y, proba),
        'F1': f1_score(y, pred),
        'precision': precision_score(y, pred),
        'recall': recall_score(y, pred),
        'accuracy': accuracy_score(y, pred),
        '混淆矩阵': f'TP={tp} FP={fp} FN={fn} TN={tn}',
    }, proba


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_CSV, encoding='utf-8-sig')
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    # Exclude regulatory-list annotations and label-derived fields from classification inputs.
    forbidden = [c for c in feat_cols if 'flagged' in c.lower() or 'miit' in c.lower()]
    if forbidden:
        raise ValueError(f'发现禁止的标签派生特征: {forbidden}')
    # Drop all-zero columns.
    feat_cols = [c for c in feat_cols if df[c].sum() != 0]
    X = df[feat_cols]
    y = df['label'].astype(int)
    groups = df['folder']
    cv = GroupKFold(n_splits=5)

    print(f'样本 {len(df)} (正 {y.sum()}/负 {(1-y).sum()}), 特征 {len(feat_cols)}, 分组 {groups.nunique()}')

    models = {
        '逻辑回归': make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=5000, C=0.1,
                                                 random_state=RANDOM_STATE)),
        '梯度提升树': HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=RANDOM_STATE),
    }

    metrics = []
    for name, m in models.items():
        row, _ = eval_model(name, m, X, y, groups, cv)
        metrics.append(row)
        print(f"  {name}: AUC={row['AUC']:.3f} F1={row['F1']:.3f} "
              f"P={row['precision']:.3f} R={row['recall']:.3f}")

    # Full-corpus permutation importance is descriptive, in-sample ranking information.
    gbt = models['梯度提升树'].fit(X, y)
    imp = permutation_importance(gbt, X, y, n_repeats=10,
                                 random_state=RANDOM_STATE, scoring='roc_auc')
    imp_df = pd.DataFrame({'feature': feat_cols,
                           'importance': imp.importances_mean,
                           'std': imp.importances_std}
                          ).sort_values('importance', ascending=False)

    lr = models['逻辑回归'].fit(X, y)
    lr_coef = pd.Series(lr.named_steps['logisticregression'].coef_[0], index=feat_cols)

    # SDK frequency/risk summary.
    sdk_cols = [c for c in feat_cols if c.startswith('sdk_')]
    imp_map = dict(zip(imp_df['feature'], imp_df['importance']))
    n_prob, n_fix = int(y.sum()), int((1 - y).sum())
    rank_rows = []
    for c in sdk_cols:
        present = df[c] == 1
        p_cnt = int((present & (y == 1)).sum())
        f_cnt = int((present & (y == 0)).sum())
        p_rate = p_cnt / n_prob if n_prob else 0
        f_rate = f_cnt / n_fix if n_fix else 0
        rank_rows.append({
            'sdk': c[4:],
            'importance': imp_map.get(c, 0),
            'lr_coef': lr_coef.get(c, 0),
            '问题版出现': p_cnt, '修复版出现': f_cnt,
            '问题版占比': round(p_rate, 3), '修复版占比': round(f_rate, 3),
            'lift': round((p_rate + 0.005) / (f_rate + 0.005), 2),
        })
    rank_df = pd.DataFrame(rank_rows).sort_values('importance', ascending=False)
    rank_path = os.path.join(OUT_DIR, 'sdk_risk_ranking.xlsx')
    rank_df.to_excel(rank_path, index=False)

    # In-sample fit AUC and single-feature correlations are diagnostic only.
    insample = {}
    for name, m in models.items():
        m.fit(X, y)
        insample[name] = roc_auc_score(y, m.predict_proba(X)[:, 1])
    corr = pd.Series({c: np.corrcoef(df[c], y)[0, 1] for c in feat_cols}
                     ).dropna().sort_values(key=abs, ascending=False)

    # Human-readable report.
    lines = [
        '# ML 基线报告: 问题版 vs 修复版',
        '',
        '## 结论(先说重要的)',
        '',
        '**在当前样本与特征集合下，模型未获得稳定高于随机水平的版本区分能力。**',
        f'- 分组 CV AUC 接近 0.5, 而树模型样本内拟合 AUC = {insample["梯度提升树"]:.3f}，提示明显过拟合，未观察到可泛化信号',
        f'- 单特征与标签相关性的最大绝对值为 |r| = {abs(corr.iloc[0]):.3f}；该结果未支持存在稳定的单一静态标志',
        '- 可能解释是问题版与修复版在 SDK、权限和敏感 API 等静态表征上高度相似，',
        '  而部分整改发生在收集时机、同意流程、配置或披露文本等当前特征未覆盖的层面。',
        '',
        '因此，本实验支持将这些特征用于**差分排查**，但不支持把当前模型作为端到端版本判别器。',
        '要做真正的风险预测, 需要: ①阴性样本(未被通报的干净 App) ②更接近行为的特征',
        '(隐私政策文本、API 调用上下文/时序、动态沙箱日志)。',
        '',
        '---',
        '',
        f'- 样本: {len(df)} (问题版 {n_prob}, 修复版 {n_fix}), 来自 {groups.nunique()} 个 App',
        f'- 特征: {len(feat_cols)} 维 (perm/api/sdk/struct), 无版本号/时间类特征(防泄漏)',
        '- 验证: GroupKFold(5) 按 App 分组, 同一 App 不跨训练/测试',
        '',
        '## 交叉验证指标',
        '',
        markdown_table(pd.DataFrame(metrics), index=False, floatfmt='.3f'),
        '',
        '## 诊断: 信号存在性检验',
        '',
        f'- 样本内拟合 AUC（描述性，不是泛化上界）: ' + ', '.join(f'{k}={v:.3f}' for k, v in insample.items()),
        '- 单特征相关性 TOP 10:',
        '',
        markdown_table(corr.head(10).to_frame('corr'), index=True, floatfmt='.3f'),
        '',
        '## 特征重要性 TOP 30 (参考, 无信号时排名无实际意义)',
        '',
        markdown_table(imp_df.head(30), index=False, floatfmt='.4f'),
        '',
        'SDK 出现频次对比表: `out/sdk_risk_ranking.xlsx`',
    ]
    report_path = os.path.join(OUT_DIR, 'ml_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n报告: {report_path}')
    print(f'风险榜: {rank_path}')


if __name__ == '__main__':
    main()
