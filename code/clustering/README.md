# 层次聚类模块

## 简介

本目录包含对GPM DPR雷达观测数据进行层次聚类分析的代码。**仅使用GPM特征（0-46维），不包含ERA5环境场变量**。

## 文件说明

| 文件 | 说明 |
|------|------|
| `cluster_hierarchical_kmeans_447.py` | 主聚类脚本（GPM-only版本） |
| `README.md` | 本说明文档 |

## 特征子集配置

参考 `config/cluster_gpm_features_only.yaml` 文件，包含以下预设：

| 预设名称 | 维度 | 主要内容 | 说明 |
|----------|------|----------|------|
| `S1_radar_precipitation` | 30 | 纯雷达特征 + 降水参数 | 基础雷达分析 |
| `S2_radar_microphysics_full` | 36 | 雷达 + 高度 + 雨滴谱 + 融化层 | **默认推荐** |
| `S3_radar_precip_dsd` | 16 | 雷达 + 降水 + 雨滴谱（精简） | 快速测试 |
| `S4_full_gpm` | 47 | 全GPM特征 | 完整分析 |
| `default` | 36 | 同S2 | 默认配置 |

### S2详细参数 (36维，推荐配置)

| 参数组 | 全局索引 | 物理含义 |
|--------|----------|----------|
| 降水基础 | 0-3 | 降水率、风暴顶、0℃层、自由底 |
| 衰减后雷达 | 4-15 | 近地面、柱内最大、斜率、高度 |
| 相态层 | 16-17 | 冰相层、液相层厚度 |
| 衰减前雷达 | 18-29 | 衰减前近地面、柱内最大等 |
| 融化层 | 32-33 | 融化层高度、厚度 |
| 雨滴谱 | 34-37 | Dm、Nw（2.5km和柱内最大） |

详见文档：`docs/GPM特征子集参数表.md`

## 使用方法

### 基本用法

```bash
# 使用默认S2配置（36维，推荐）
cd d:\qingwen\pycharm\learnpython1
python run/hierarchical_clustering/cluster_hierarchical_kmeans_447.py

# 使用S1配置（30维，纯雷达）
python run/hierarchical_clustering/cluster_hierarchical_kmeans_447.py --feature-preset S1_radar_precipitation

# 使用S3配置（16维，精简版）
python run/hierarchical_clustering/cluster_hierarchical_kmeans_447.py --feature-preset S3_radar_precip_dsd

# 使用S4配置（47维，全GPM）
python run/hierarchical_clustering/cluster_hierarchical_kmeans_447.py --feature-preset S4_full_gpm

# 指定类别数范围
python run/hierarchical_clustering/cluster_hierarchical_kmeans_447.py --n-clusters 3 4 5 6 7 8
```

### 输出目录

- 聚类结果：`results/hierarchical_clustering/cluster_results_{preset}_K{best_k}/`
- 包含：聚类标签、树状图、评估曲线、配置文件

## 输入数据

- 输入：`results/matched_nearest/matched_nearest_YYYY_MM.npz`
- 使用：前47维（GPM特征），忽略ERA5部分（47-446维）

## 修改特征配置

编辑 `config/cluster_gpm_features_only.yaml` 文件，修改 `feature_presets` 中的 `indices` 列表即可自定义特征子集。
