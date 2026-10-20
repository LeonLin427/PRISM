# SDK attribution and descriptive case studies

- Model: full-graph RGCN fitted to all corpus events, with the same hyperparameters as gnn_full: lr=0.001, epochs=300, seeds [1000, 2000, 3000].
- Difference-aware graph: app-removed-sdk/app-added-sdk relations encode retrospective remediated-version information; rankings reflect both SDK use and remediation evidence.
- Attribution: SDK embedding gradient times input, S_c[i]=(∂T_c/∂E_i)·E_i; T_c is the mean logit among category-positive events.
- Prespecified gate: use gradient rankings when category-level ρ_c, the mean pairwise Spearman correlation across 3 seeds, is >= 0.6; otherwise use deterministic frequency difference. Both score types are retained.
- External validity and recall@K are not evaluated: application and SDK notices have different scopes and timing, and remediation removals are not independent ground truth when removal relations are already graph inputs.

## Cross-seed stability (3 seeds)

| class | rho | method |
|---|---|---|
| C1 | 0.758 | gradient |
| C2 | 0.745 | gradient |
| C3 | 0.595 | freqdiff |
| C4 | 0.632 | gradient |
| C5 | 0.711 | gradient |
| C6 | 0.702 | gradient |

## Top-10 SDKs per category, ranked by the selected method

### C1 (method=gradient)

| rank | sdk | score | freqdiff | flagged |
|---|---|---|---|---|
| 1 | 友盟+ U-App/组件 | 0.1125 | -0.040 |  |
| 2 | 360 手机卫士/组件 | 0.1035 | 0.031 |  |
| 3 | 阿里云 SDK | 0.0784 | 0.059 |  |
| 4 | PushSDK | 0.0738 | 0.009 |  |
| 5 | 极光 JPush/JAnalytics | 0.0718 | 0.123 |  |
| 6 | 微博 SDK | 0.0628 | 0.201 |  |
| 7 | QQ 互联 SDK | 0.0544 | 0.044 |  |
| 8 | vivo 广告 | 0.0542 | 0.004 |  |
| 9 | 小米推送 MiPush | 0.0462 | 0.078 |  |
| 10 | 腾讯云 SDK | 0.0446 | 0.076 |  |

### C2 (method=gradient)

| rank | sdk | score | freqdiff | flagged |
|---|---|---|---|---|
| 1 | 友盟+ U-App/组件 | 0.1525 | 0.017 |  |
| 2 | Lottie 动画库 | 0.1069 | -0.103 |  |
| 3 | UCrop 图片裁剪 | 0.0890 | 0.108 |  |
| 4 | Glide | 0.0818 | 0.060 |  |
| 5 | 360 加固保 | 0.0786 | 0.044 |  |
| 6 | 支付宝 SDK | 0.0710 | 0.019 |  |
| 7 | 360 手机卫士/组件 | 0.0636 | 0.140 |  |
| 8 | 魅族推送 | 0.0619 | 0.162 |  |
| 9 | 百度定位/地图 | 0.0575 | 0.158 |  |
| 10 | Gson | 0.0532 | 0.042 |  |

### C3 (method=freqdiff)

| rank | sdk | score | freqdiff | flagged |
|---|---|---|---|---|
| 1 | 优量汇(广点通) | 0.7155 | 0.715 |  |
| 2 | 穿山甲/巨量引擎 | 0.5907 | 0.591 |  |
| 3 | 字节直播 SDK | 0.4977 | 0.498 |  |
| 4 | 快手广告联盟 | 0.3492 | 0.349 |  |
| 5 | 百度联盟广告 | 0.3431 | 0.343 |  |
| 6 | 字节 JProtect 加固 | 0.2922 | 0.292 |  |
| 7 | 移动安全联盟 OAID(MSA) | 0.2548 | 0.255 |  |
| 8 | 腾讯 Matrix | 0.1731 | 0.173 |  |
| 9 | 华为广告 | 0.1548 | 0.155 |  |
| 10 | Sigmob 广告 | 0.1520 | 0.152 |  |

### C4 (method=gradient)

| rank | sdk | score | freqdiff | flagged |
|---|---|---|---|---|
| 1 | 360 手机卫士/组件 | 0.8386 | 0.063 |  |
| 2 | Guava | 0.4389 | -0.219 |  |
| 3 | 电信天翼账号 | 0.2686 | -0.156 |  |
| 4 | 支付宝 SDK | 0.1813 | 0.062 |  |
| 5 | 荣耀推送 | 0.1472 | 0.056 |  |
| 6 | QQ 互联 SDK | 0.1450 | 0.179 |  |
| 7 | 神策数据 | 0.1182 | 0.021 | ✓ |
| 8 | 友盟+ U-App/组件 | 0.0951 | -0.023 |  |
| 9 | 火山引擎/字节云 | 0.0921 | -0.006 |  |
| 10 | 阿里百川/mtop | 0.0828 | -0.060 |  |

### C5 (method=gradient)

| rank | sdk | score | freqdiff | flagged |
|---|---|---|---|---|
| 1 | 热云 TrackingIO | 0.4722 | 0.078 |  |
| 2 | 个推 | 0.3709 | -0.151 |  |
| 3 | 阿里百川/mtop | 0.1619 | 0.076 |  |
| 4 | Agoo Push | 0.1281 | 0.119 |  |
| 5 | 支付宝 SDK | 0.1024 | 0.048 |  |
| 6 | 阿里云 SDK | 0.0988 | 0.159 |  |
| 7 | 腾讯 Bugly | 0.0861 | -0.028 |  |
| 8 | 百度联盟广告 | 0.0733 | 0.128 |  |
| 9 | 友盟+ U-App/组件 | 0.0729 | 0.126 |  |
| 10 | QQ 互联 SDK | 0.0620 | -0.011 |  |

### C6 (method=gradient)

| rank | sdk | score | freqdiff | flagged |
|---|---|---|---|---|
| 1 | 小米推送 MiPush | 0.2741 | 0.289 |  |
| 2 | 极光 JPush/JAnalytics | 0.2420 | 0.113 |  |
| 3 | Firebase | 0.2031 | 0.070 |  |
| 4 | 友盟 U-Share/分享 | 0.1758 | 0.119 |  |
| 5 | 百度定位/地图 | 0.1614 | -0.099 |  |
| 6 | 支付宝 SDK | 0.1274 | 0.140 |  |
| 7 | 高德地图/定位 | 0.0953 | 0.034 |  |
| 8 | OPPO 推送 | 0.0800 | 0.229 |  |
| 9 | vivo 推送 | 0.0750 | 0.356 |  |
| 10 | 华为 HMS/推送 | 0.0731 | 0.264 |  |

## Descriptive cases: top-10 rankings alongside remediation removals

### 吴试纸 (C1, method=per-app gradient; in-sample descriptive score=0.995)
- This score comes from a full-corpus fit and describes the case; it is not an OOF prediction or a generalization estimate.

- Top-10 SDKs for this category: Guava；极光 JPush/JAnalytics；PushSDK；微博 SDK；小米推送 MiPush；QQ 互联 SDK；腾讯 MMKV；荣耀推送；SmartRefreshLayout；vivo 推送
- SDKs removed during remediation: BRVAH 列表适配器；MobTech(ShareSDK/MobSDK)；OPPO 推送；OkHttp；Picasso 图片库；PushSDK；QQ 互联 SDK；RxJava/RxAndroid；SmartRefreshLayout；vivo 推送；小米推送 MiPush；微博 SDK；极光 JPush/JAnalytics；移动安全联盟 OAID(MSA)；腾讯 MMKV；荣耀推送；魅族推送
- Number of removed components: 136

### 吴试纸 (C4, method=per-app gradient; in-sample descriptive score=1.000)
- This score comes from a full-corpus fit and describes the case; it is not an OOF prediction or a generalization estimate.

- Top-10 SDKs for this category: Guava；微博 SDK；UniversalImageLoader；MobTech(ShareSDK/MobSDK)；Gson；移动安全联盟 OAID(MSA)；BRVAH 列表适配器；AdMob；QQ 互联 SDK；SmartRefreshLayout
- SDKs removed during remediation: BRVAH 列表适配器；MobTech(ShareSDK/MobSDK)；OPPO 推送；OkHttp；Picasso 图片库；PushSDK；QQ 互联 SDK；RxJava/RxAndroid；SmartRefreshLayout；vivo 推送；小米推送 MiPush；微博 SDK；极光 JPush/JAnalytics；移动安全联盟 OAID(MSA)；腾讯 MMKV；荣耀推送；魅族推送
- Number of removed components: 136

### 备忘录记事本 (C3, method=freqdiff; in-sample descriptive score=1.000)
- This score comes from a full-corpus fit and describes the case; it is not an OOF prediction or a generalization estimate.

- Top-10 SDKs for this category: 优量汇(广点通)；穿山甲/巨量引擎；字节直播 SDK；快手广告联盟；百度联盟广告；字节 JProtect 加固；移动安全联盟 OAID(MSA)；腾讯 Matrix；华为广告；Sigmob 广告
- SDKs removed during remediation: OPPO 推送；vivo 广告；优量汇(广点通)；字节直播 SDK；百度联盟广告；穿山甲/巨量引擎
- Number of removed components: 127

### 健康界 (C1, method=per-app gradient; in-sample descriptive score=0.998)
- This score comes from a full-corpus fit and describes the case; it is not an OOF prediction or a generalization estimate.

- Top-10 SDKs for this category: PushSDK；友盟+ U-App/组件；神策数据；微博 SDK；Zxing 二维码；QQ 互联 SDK；华为 HMS/推送；Lottie 动画库；阿里百川/mtop；阿里云 SDK
- SDKs removed during remediation: PushSDK；神策数据
- Number of removed components: 2

## Interpretation and limitations

- Attribution depends on the full graph through 2 layers of relational aggregation, including SDK co-occurrence. It provides rankings, not per-edge causal effects.
- Gradient times input is a local first-order measure assessed for cross-seed stability; the rankings are descriptive profiles, not detectors.
- Removal/addition relations directly encode retrospective remediation evidence. Frequency difference compares category-positive and other events, independently of auxiliary SDK-list membership.
- The differences in remediation.csv are retrospective evidence, distinct from prospective prediction.
- Cases are illustrative examples selected after analysis. Categories below the stability gate use the global frequency-difference ranking; case overlap is not external validation or causal evidence.
