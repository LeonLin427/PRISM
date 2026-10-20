"""Map normalized notice descriptions to the paper's six violation categories.

Use ordered keyword matching. Specific phrases must precede broader keywords;
for example, targeted-push wording takes precedence over a generic coercion
keyword so it maps to the intended category. The category strings are retained
in their source language as stable dataset identifiers.

clean_labels.py derives the observed distribution from labels.csv."""
import re

# Remove whitespace before matching source-language descriptions.
def norm_vio(s):
    return re.sub(r'\s+', '', s or '')

# Ordered (regex, category) rules; the first matching rule wins.
RULES = [
    (r'应用分发平台上的App信息明示不到位|应用分发平台管理责任落实不到位',
     'C6 其他(未明示/定向推送/默认同意)'),
    (r'定向推送', 'C6 其他(未明示/定向推送/默认同意)'),
    (r'索取权限|强制', 'C2 强制频繁过度索取权限'),
    (r'自启动', 'C4 频繁自启动关联启动'),
    (r'超范围收集|违规收集', 'C1 违规/超范围收集个人信息'),
    (r'违规使用', 'C5 违规使用个人信息'),
    (r'未明示|默认同意', 'C6 其他(未明示/定向推送/默认同意)'),
    (r'跳转|摇一摇|无法关闭|欺骗误导|弹窗|关闭|退出|误导', 'C3 弹窗跳转/欺骗误导'),
]

CLASS_NAMES = [
    'C1 违规/超范围收集个人信息',
    'C2 强制频繁过度索取权限',
    'C3 弹窗跳转/欺骗误导',
    'C4 频繁自启动关联启动',
    'C5 违规使用个人信息',
    'C6 其他(未明示/定向推送/默认同意)',
]


def classify(raw):
    """Map a notice description to its category, or return UNKNOWN when unmatched."""
    v = norm_vio(raw)
    if not v:
        return 'UNKNOWN'
    for pat, cls in RULES:
        if re.search(pat, v):
            return cls
    return 'UNKNOWN'


if __name__ == '__main__':
    # Print mappings for known wording variants for manual inspection.
    samples = [
        '违规收集个人信息',
        'APP 强制、频繁、过度索取权限',
        '超范围收集个人信息',
        '信息窗口乱跳转',
        'APP 频繁自启动和关联启动',
        '违规使用个人信息',
        'APP强制、频繁、过度索取权限',
        '未明示收集个人信息清单',
        '信息窗口“摇一摇”乱跳转',
        '强制用户使用定向推送功能',
        '信息窗口点击乱跳转',
        '信息窗口无法关闭',
        '欺骗误导强迫用户',
        '开屏信息窗口“乱跳转”，误导用户',
        '隐私政策默认同意',
        'APP 强制、频繁、过',
        '度索取权限',
        '信息窗口“摇一摇”乱跳转，误导用户',
        '开屏信息窗口“摇一摇”乱跳转',
        '开屏信息窗口“摇一摇”或误导用户点击乱跳转',
        '信息窗口未提供关闭或退出标识',
        '信息窗口“摇一摇”乱跳转或自动乱跳转',
        '信息窗口关闭按钮未显著展示',
        '弹窗关不掉、乱跳转',
        '“摇一摇”乱跳转',
    ]
    for s in samples:
        print(f'  {classify(s):<22}  <-  {s}')
