# ML 基线报告: 问题版 vs 修复版

## 结论(先说重要的)

**在当前样本与特征集合下，模型未获得稳定高于随机水平的版本区分能力。**
- 分组 CV AUC 接近 0.5, 而树模型样本内拟合 AUC = 0.948，提示明显过拟合，未观察到可泛化信号
- 单特征与标签相关性的最大绝对值为 |r| = 0.048；该结果未支持存在稳定的单一静态标志
- 可能解释是问题版与修复版在 SDK、权限和敏感 API 等静态表征上高度相似，
  而部分整改发生在收集时机、同意流程、配置或披露文本等当前特征未覆盖的层面。

因此，本实验支持将这些特征用于**差分排查**，但不支持把当前模型作为端到端版本判别器。
要做真正的风险预测, 需要: ①阴性样本(未被通报的干净 App) ②更接近行为的特征
(隐私政策文本、API 调用上下文/时序、动态沙箱日志)。

---

- 样本: 433 (问题版 217, 修复版 216), 来自 212 个 App
- 特征: 176 维 (perm/api/sdk/struct), 无版本号/时间类特征(防泄漏)
- 验证: GroupKFold(5) 按 App 分组, 同一 App 不跨训练/测试

## 交叉验证指标

| model | AUC | F1 | precision | recall | accuracy | 混淆矩阵 |
| --- | --- | --- | --- | --- | --- | --- |
| 逻辑回归 | 0.506 | 0.508 | 0.496 | 0.521 | 0.494 | TP=113 FP=115 FN=104 TN=101 |
| 梯度提升树 | 0.510 | 0.517 | 0.519 | 0.516 | 0.517 | TP=112 FP=104 FN=105 TN=112 |

## 诊断: 信号存在性检验

- 样本内拟合 AUC（描述性，不是泛化上界）: 逻辑回归=0.646, 梯度提升树=0.948
- 单特征相关性 TOP 10:

| feature | corr |
| --- | --- |
| sdk_360 手机卫士/组件 | 0.048 |
| sdk_头条/巨量日志 | -0.045 |
| perm_MANAGE_EXTERNAL_STORAGE | -0.042 |
| sdk_火山引擎/字节云 | -0.040 |
| sdk_Gson | -0.039 |
| sdk_vivo 广告 | 0.034 |
| sdk_FastJSON | -0.032 |
| sdk_Guava | -0.030 |
| sdk_360 加固保 | 0.027 |
| sdk_Mintegral 广告 | -0.027 |

## 特征重要性 TOP 30 (参考, 无信号时排名无实际意义)

| feature | importance | std |
| --- | --- | --- |
| struct_unknown_prefix3 | 0.1333 | 0.0125 |
| sdk_count | 0.0905 | 0.0107 |
| struct_n_so | 0.0891 | 0.0115 |
| struct_n_components | 0.0756 | 0.0102 |
| struct_n_domains | 0.0538 | 0.0120 |
| api_getInstalledApplications | 0.0442 | 0.0072 |
| perm_total | 0.0431 | 0.0044 |
| api_getDeviceId | 0.0405 | 0.0078 |
| sdk_优量汇(广点通) | 0.0273 | 0.0038 |
| api_getSimSerialNumber | 0.0243 | 0.0060 |
| api_getSerial | 0.0239 | 0.0047 |
| api_getRunningTasks | 0.0142 | 0.0022 |
| api_getRunningAppProcesses | 0.0142 | 0.0038 |
| perm_custom | 0.0127 | 0.0046 |
| api_addJavascriptInterface | 0.0113 | 0.0032 |
| perm_WRITE_SETTINGS | 0.0111 | 0.0032 |
| api_getMacAddress | 0.0110 | 0.0022 |
| sdk_Gson | 0.0109 | 0.0045 |
| perm_dangerous | 0.0104 | 0.0024 |
| sdk_腾讯 MMKV | 0.0083 | 0.0042 |
| api_getImei | 0.0079 | 0.0018 |
| api_DexClassLoader | 0.0076 | 0.0030 |
| sdk_友盟+ U-App/组件 | 0.0069 | 0.0026 |
| sdk_极光 JPush/JAnalytics | 0.0068 | 0.0024 |
| perm_MANAGE_EXTERNAL_STORAGE | 0.0052 | 0.0038 |
| sdk_华为 HMS/推送 | 0.0050 | 0.0016 |
| sdk_Luban 图片压缩 | 0.0045 | 0.0015 |
| sdk_UCrop 图片裁剪 | 0.0043 | 0.0016 |
| api_getCellLocation | 0.0042 | 0.0024 |
| sdk_头条/巨量日志 | 0.0040 | 0.0021 |

SDK 出现频次对比表: `out/sdk_risk_ranking.xlsx`