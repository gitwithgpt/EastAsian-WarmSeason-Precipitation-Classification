# 论文图表与附件完整清单（修订版 v3 — 含投稿系统归类）

> 目标期刊：Atmospheric Research (Elsevier)  
> 论文编号：基于当前结果分析章节中的引用  
> 修订日期：2024年6月16日  
> 状态：🟢 已确认 / 🔴 需处理 / ⚠️ 已知问题

---

## 修订记录

| 日期 | 修订内容 |
|------|---------|
| 2024-06-16 | 重命名 `figure_s4_bootstrap_stability` → `figure_s11_bootstrap_stability`（解决与S4的编号冲突） |
| 2024-06-16 | 文稿修正：3.4节 "图3b/3c" → "图S4/S5"；3.6节 "图10" → "图6" |
| 2024-06-16 | 补充所有图表的生成代码来源 |
| 2024-06-16 | 标注S8/S9数据密度问题 |
| 2024-06-16 | **新增**：补充投稿系统归类与上传文件类型标注（Main / Supplementary / e-component） |
| 2024-06-16 | **新增**：Supplementary Data 建议清单与内容说明 |

---

## 一、Elsevier Editorial Manager 投稿材料清单

### 1. 必传文件（System Mandatory）

| 编号 | 文件名称 | 文件类型 | 上传至系统时选择的文件类型 | 备注 |
|------|---------|---------|------------------------|------|
| A | Manuscript | `.docx` 或 `.tex` | **Manuscript** | 主文稿；含标题、摘要、正文、参考文献、表注 |
| B | Cover Letter | `.doc` / `.docx` | **Cover Letter** | 说明投稿动机、科学贡献、期刊适配性 |
| C | Highlights | `.doc` / `.docx` | **Highlights** | 3–5条，每条 ≤85 字符（含空格），英文 |
| D | Graphical Abstract | `.tif` / `.eps` / `.jpg` | **Graphical Abstract** | 建议用论文架构图（图3）或分类示意缩略图；≥531×1328 px |

### 2. 图表文件（Figures & Tables）

**⚠️ 核心原则：Elsevier 系统要求每个图单独作为独立文件上传，禁止将多图合并为一张上传。**

| 编号 | 内容 | 文件类型 | 系统文件类型 | 命名格式 |
|------|------|---------|-------------|---------|
| F1–F6 | 主图 Figure 1–6 | `.tif` / `.eps`（首选）或 `.png` / `.jpg` | **Figure** | `Fig1.tif`, `Fig2.tif`, ... `Fig6.tif` |
| S1–S11 | 补充图 Figure S1–S11 | `.tif` / `.eps` | **Supplementary Material** | `FigS1.tif`, `FigS2.tif`, ... `FigS11.tif` |
| T1–T7 | 正文表格（嵌入文稿中） | 嵌入 Manuscript | — | 在文稿中以可编辑表格呈现，无需单独上传 |
| TS1–TS5 | 补充表格 | `.csv` / `.xlsx` | **Supplementary Material** | `TableS1.csv`, `TableS2.csv`, ... `TableS5.csv` |

### 3. 声明文件（Declarations）

| 编号 | 内容 | 文件类型 | 系统文件类型 | 备注 |
|------|------|---------|-------------|------|
| D1 | Declaration of Competing Interest | `.doc` / 系统内勾选 | **Declaration** | Elsevier 系统内置模板；也可上传独立文件 |
| D2 | CRediT Author Contributions | 嵌入文稿或独立文件 | **CRediT** | 建议嵌入文稿末尾，并同时在系统填写 |
| D3 | Data Availability Statement | 嵌入文稿 | — | 嵌入文稿末尾 |
| D4 | Funding Statement | 嵌入文稿 | — | 嵌入文稿末尾 |
| D5 | Ethics Statement | 如适用 | — | 本研究不涉及人类/动物，可声明 "Not applicable" |

### 4. 可选/建议文件（Optional but Recommended）

| 编号 | 内容 | 文件类型 | 系统文件类型 | 备注 |
|------|------|---------|-------------|------|
| O1 | Suggested Reviewers | 系统内填写 | — | 建议3–5位，提供姓名、单位、邮箱；避免近期合作者 |
| O2 | Excluded Reviewers | 系统内填写 | — | 如有利益冲突者，可列出 |
| O3 | Supplementary Data（代码/原始数据） | `.zip` / `.py` / `.npz` | **Supplementary Material** | 见下文「Supplementary Data 建议」 |
| O4 | Response to Reviewers（仅修回时） | `.docx` | **Response to Reviewers** | 初次投稿无需 |

---

## 二、主图（Main Figures, 正文内引用）

**📁 投稿系统归类：上传时选择文件类型为 `Figure`，按编号单独上传。**

| 论文编号 | 图题（建议） | 当前文件名 | 论文引用位置 | 格式 | 生成代码来源 | 状态 | 备注 |
|---------|------------|-----------|------------|------|------------|------|------|
| **Figure 1** | 四类降水关键微物理参数箱线图 | `figure_1_microphysics_boxplots` | 3.2.1 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**主图** |
| **Figure 2** | 四类降水关键环境参数箱线图 | `figure_2_environment_boxplots` | 3.5 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**主图** |
| **Figure 3** | 研究区域有效降水样本空间频率分布 | `figure_3_spatial_distribution` | 3.4 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 单张图，总样本密度；**主图** |
| **Figure 4** | 四类降水多频准CFAD（Ku/Ka/DFR） | `figure_4_multifrequency_cfad` | 3.3 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**主图** |
| **Figure 5** | 四类环境参数标准化z-score比较 | `figure_5_environment_zscore_profile` | 3.5 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**主图** |
| **Figure 6** | 四类降水强度频率分布与分位数 | `figure_6_precip_intensity_distribution` | 3.6 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**主图** |

> **主图总数**：6 张。符合 Atmospheric Research 无严格上限但建议 6–10 张的惯例。

---

## 三、补充图（Supplementary Figures, SI引用）

**📁 投稿系统归类：上传时选择文件类型为 `Supplementary Material`，命名格式 `FigS1.tif`, `FigS2.tif` 等。**

| 论文编号 | 图题（建议） | 当前文件名 | 论文引用位置 | 格式 | 生成代码来源 | 状态 | 备注 |
|---------|------------|-----------|------------|------|------------|------|------|
| **Figure S1** | k值选择指标（DB指数、轮廓系数、CH指数） | `figure_s1_k_selection` | 3.1 | png/pdf/svg/tif | `generate_supplementary_figures_v2.py` (Jun 16) | 🟢 | K=3–8，已修正标题位置；**SI 图** |
| **Figure S2** | 层次聚类树状图（Ward连接） | `figure_s2_dendrogram` | 3.1 | png/pdf/svg/tif | `generate_supplementary_figures_v2.py` (Jun 16) | 🟢 | 从Z_matrix.npy生成；**SI 图** |
| **Figure S3** | 微物理参数小提琴分布图 | `figure_s3_microphysics_violin` | 3.2.1 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**SI 图** |
| **Figure S4** | 四类降水各自空间分布 | `figure_s4_spatial_each_cluster` | 3.4 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 2×2子图，各簇独立分布；**SI 图** |
| **Figure S5** | 各网格主导降水类型地图 | `figure_s5_dominant_cluster_map` | 3.4 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**SI 图** |
| **Figure S6** | 冰相参数箱线图（IPL/LPL/亮带高度/厚度） | `figure_s6_ice_phase_boxplots` | 3.2.1 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**SI 图** |
| **Figure S7** | 亮带出现率统计 | `figure_s7_brightband_occurrence` | 3.2.1 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**SI 图** |
| **Figure S8** | DSD参数与亮带高度联合分布（准CFAD） | `figure_s8_quasi_cfad_dsd_brightband` | 3.3 | png/pdf/svg/tif | `generate_supplementary_figures_v2.py` (Jun 16) | ⚠️ | **数据密度极低，颜色无法区分**；需增大bin或改用log计数；**SI 图** |
| **Figure S9** | DSD参数与风暴顶高联合分布（准CFAD） | `figure_s9_quasi_cfad_dsd_stormtop` | 3.3 | png/pdf/svg/tif | `generate_supplementary_figures_v2.py` (Jun 16) | ⚠️ | **数据密度极低，颜色无法区分**；同S8问题；**SI 图** |
| **Figure S10** | 环境参数效应量（Effect Size） | `figure_s10_environment_effect_size` | 3.5 | png/pdf/svg/tif | `generate_all_figures.py` (Jun 15) | 🟢 | 已确认；**SI 图** |
| **Figure S11**（可选） | Bootstrap稳定性分析（ARI） | `figure_s11_bootstrap_stability` | 方法章节 | png/pdf/svg/tif | `generate_supplementary_figures_v2.py` (Jun 16) | 🟢 | **已更新**：100次Bootstrap迭代，200k子样本。K=4 ARI=0.886±0.041，最优且稳健；**SI 图（推荐上传）** |
| **Figure S12**（可选） | 暖季各月降水样本分布 | **待确认** | 不再正文引用 | — | — | 🔴 | 原`figure_4_seasonal`，已移出主线，可作为SI保留；**SI 图（可选）** |

> **SI 图总数**：11 张（S1–S11），S12 为可选。建议全部上传作为 Supplementary Material。

---

## 四、论文表格（正文内，嵌入 Manuscript）

**📁 投稿系统归类：嵌入 Manuscript 文件内，无需单独上传。**

| 论文编号 | 表题 | 内容来源 | 位置 | 状态 | 备注 |
|---------|------|---------|------|------|------|
| **Table 1** | 研究数据来源 | 手动整理 | 2.1 | 🟢 | 已完整；嵌入文稿 |
| **Table 2** | 22维微物理特征参数 | 手动整理 | 2.2 | 🟢 | 已完整；嵌入文稿 |
| **Table 3** | ERA5环境场参数 | 手动整理 | 2.3 | 🟢 | 已完整；嵌入文稿 |
| **Table 4** | 聚类数K=3–8评估指标汇总 | 手动整理（K值选择结果） | 3.1 | 🟢 | 已完整；嵌入文稿 |
| **Table 5** | 四类关键微物理参数统计特征（z-score） | 基于 `cluster4_microphysics_stats.csv` | 3.2.2 | 🟢 | 已完整，节选10个参数；嵌入文稿 |
| **Table 6** | 关键微物理参数统计显著性检验（节选） | 基于 `cluster4_significance_tests.csv` | 3.7 | 🟢 | 正文为节选；嵌入文稿 |
| **Table 7** | 关键环境参数统计显著性检验（节选） | 基于 `cluster4_significance_tests.csv` | 3.7 | 🟢 | 正文为节选；嵌入文稿 |

> **注**：原清单中Table 1–4的编号与文稿不一致。本清单已按文稿实际引用顺序重新编号（Table 1=数据来源，Table 2=22维参数，Table 3=ERA5参数，Table 4=K评估，Table 5=z-score，Table 6=微物理检验，Table 7=环境检验）。

---

## 五、附件/补充材料表格（Supplementary Tables）

**📁 投稿系统归类：上传时选择文件类型为 `Supplementary Material`，命名格式 `TableS1.csv` 等。**

| 附件编号 | 文件名 | 内容描述 | 来源文件 | 状态 | 格式 | 投稿归类 |
|---------|--------|---------|---------|------|------|---------|
| **Table S1** | `table_s1_cluster_evaluation_metrics.csv` | 完整聚类评估指标（K=3–8，含DB/轮廓/CH/最小样本量） | 手动整理K值选择数据 | 🟢 已生成 | CSV | **Supplementary Material** |
| **Table S2** | `cluster4_microphysics_stats.csv` | 四类全部微物理参数统计特征（均值、中位数、标准差、四分位数、z-score） | 已有 | 🟢 | CSV | **Supplementary Material** |
| **Table S3** | `cluster4_environment_stats.csv` | 四类全部环境参数统计特征（均值、中位数、标准差、四分位数、z-score） | 已有 | 🟢 | CSV | **Supplementary Material** |
| **Table S4** | `cluster4_significance_tests.csv` | 全部90组配对检验结果（6对×15参数，含Mann-Whitney U、KS、Cliff's Delta） | 已有 | 🟢 | CSV | **Supplementary Material** |
| **Table S5** | `table_s5_cluster_sample_composition.csv` | 各类别样本量、占比、暖季各月分布 | 基于story_dataset生成 | 🟢 已生成 | CSV | **Supplementary Material** |

> **Supplementary Table 总数**：5 张。建议全部上传作为 Supplementary Material。

---

## 六、Supplementary Data 建议清单

除了 Supplementary Figures 和 Supplementary Tables 之外，Elsevier 系统允许上传 **e-component / Supplementary Material** 类型的额外文件。以下是你研究中有价值且符合期刊规范的建议内容：

### 1. 强烈推荐上传（Recommended）

| 编号 | 内容 | 文件类型 | 理由 | 建议文件名 |
|------|------|---------|------|---------|
| **SD1** | 图表生成代码 | `.py` / `.zip` | 确保结果可复现； reviewers 可验证作图方法 | `supplementary_code_figure_generation.zip` |
| **SD2** | 数据分析与统计检验代码 | `.py` / `.ipynb` | 聚类、检验、Bootstrap 等核心分析的可复现脚本 | `supplementary_code_analysis.zip` |
| **SD3** | GPM-ERA5 数据匹配框架代码 | `.py` | 多GPU并行匹配方法的技术实现细节 | `supplementary_code_data_matching.zip` |
| **SD4** | 分类结果数据集（元数据） | `.csv` / `.npz` | 提供 960 万样本分类标签的抽样（如 1% 随机抽样）供读者验证 | `supplementary_sample_classification_labels.csv` |

### 2. 建议上传（Suggested）

| 编号 | 内容 | 文件类型 | 理由 | 建议文件名 |
|------|------|---------|------|---------|
| **SD5** | Bootstrap 稳定性完整结果 | `.csv` | 100次迭代完整ARI结果（含每次迭代的子样本索引） | `supplementary_bootstrap_ari_results.csv` |
| **SD6** | 子集敏感性分析结果 | `.csv` / `.png` | 不同随机子样本量（50k/100k/200k/500k）下的聚类稳定性对比 | `supplementary_sensitivity_analysis.zip` |
| **SD7** | 全部配对检验完整结果（已含在Table S4） | `.csv` | Table S4 已覆盖，无需重复；但可附加上效应量等级判定说明 | — |
| **SD8** | 各月降水类型频率分布数据 | `.csv` | 支撑 S12（如上传）和 Table S5 的原始月度统计数据 | `supplementary_monthly_frequency.csv` |

### 3. 可选上传（Optional）

| 编号 | 内容 | 文件类型 | 理由 | 备注 |
|------|------|---------|------|------|
| **SD9** | 数据匹配流程文档 | `.pdf` / `.docx` | 详细描述 GPM-ERA5 匹配算法的步骤和参数 | 如方法章节已足够详细，可不上传 |
| **SD10** | 参数选择说明文档 | `.pdf` | 解释 22 维参数选取的依据和文献支撑 | 如引言已充分论述，可不上传 |
| **SD11** | 聚类算法对比（k-means vs GMM vs HDBSCAN） | `.csv` / `.png` | 展示为何选择 k-means 而非其他算法的对比实验 | 如有预实验结果，建议上传 |

### 4. 不建议上传的内容

| 内容 | 理由 |
|------|------|
| 原始 GPM DPR L2 文件（~TB 级） | 体积过大；期刊和读者无法直接使用；已通过数据可用性声明指引获取途径 |
| 原始 ERA5 再分析数据（~GB 级） | 同上；ERA5 为公开数据集，无需重复上传 |
| 完整 960 万样本分类结果（~GB 级） | 体积过大；建议上传 1% 随机抽样即可 |
| 中间过程的临时 `.npz` 文件 | 非最终成果，无直接参考价值 |

---

## 七、文件完整性总览（environment_story/ 目录）

```
environment_story/
├── 主图（6张，各4格式 = 24文件）→ 上传为 Figure
│   ├── figure_1_microphysics_boxplots [png/pdf/svg/tif] 🟢 → Fig1.tif
│   ├── figure_2_environment_boxplots [png/pdf/svg/tif] 🟢 → Fig2.tif
│   ├── figure_3_spatial_distribution [png/pdf/svg/tif] 🟢 → Fig3.tif
│   ├── figure_4_multifrequency_cfad [png/pdf/svg/tif] 🟢 → Fig4.tif
│   ├── figure_5_environment_zscore_profile [png/pdf/svg/tif] 🟢 → Fig5.tif
│   └── figure_6_precip_intensity_distribution [png/pdf/svg/tif] 🟢 → Fig6.tif
│
├── 补充图（11张，各4格式 = 44文件）→ 上传为 Supplementary Material
│   ├── figure_s1_k_selection → FigS1.tif 🟢
│   ├── figure_s2_dendrogram → FigS2.tif 🟢
│   ├── figure_s3_microphysics_violin → FigS3.tif 🟢
│   ├── figure_s4_spatial_each_cluster → FigS4.tif 🟢
│   ├── figure_s5_dominant_cluster_map → FigS5.tif 🟢
│   ├── figure_s6_ice_phase_boxplots → FigS6.tif 🟢
│   ├── figure_s7_brightband_occurrence → FigS7.tif 🟢
│   ├── figure_s8_quasi_cfad_dsd_brightband → FigS8.tif ⚠️（需修复后上传）
│   ├── figure_s9_quasi_cfad_dsd_stormtop → FigS9.tif ⚠️（需修复后上传）
│   ├── figure_s10_environment_effect_size → FigS10.tif 🟢
│   ├── figure_s11_bootstrap_stability → FigS11.tif 🟢（推荐上传）
│   └── [可选] figure_4_seasonal → FigS12.tif? 🔴（可选）
│
├── 数据表格（5张CSV）→ 上传为 Supplementary Material
│   ├── table_s1_cluster_evaluation_metrics.csv → TableS1.csv 🟢
│   ├── cluster4_microphysics_stats.csv → TableS2.csv 🟢
│   ├── cluster4_environment_stats.csv → TableS3.csv 🟢
│   ├── cluster4_significance_tests.csv → TableS4.csv 🟢
│   └── table_s5_cluster_sample_composition.csv → TableS5.csv 🟢
│
├── 代码与补充数据（建议打包上传）→ 上传为 Supplementary Material
│   ├── generate_all_figures.py (Jun 15) → 主图1-6 + SI图S3/S4/S5/S6/S7/S10
│   ├── generate_supplementary_figures_v2.py (Jun 16) → SI图S1/S2/S8/S9/S11
│   └── [建议] supplementary_code_analysis.zip → 聚类/检验/Bootstrap代码
│
└── 投稿文稿（1份）→ 上传为 Manuscript
    └── 论文完整稿_AtmosphericResearch投稿版.docx
```

---

## 八、已知问题与待处理项

### 1. S8/S9 数据密度问题（⚠️）

**现象**：S8（DSD-亮带）和S9（DSD-风暴顶）二维直方图几乎全为深紫色，概率密度值极低（0–0.01），颜色无法区分数据分布。

**原因**：9.6M样本在二维参数空间（Dm 0.5–3.0 mm × 高度 0–14 km）分布极为稀疏，归一化后每个bin的概率密度 < 0.01。

**建议修复方案**：
- 方案A：增大2D直方图的bin尺寸（如从50×50改为20×20），提升每个bin的样本数
- 方案B：使用原始计数而非概率密度，并采用对数色标（`LogNorm`）
- 方案C：对数据进行分位数自适应分bin（不等距），确保高密度区域有足够的颜色区分度
- 方案D：如果数据本身确实如此稀疏，可考虑在图注中明确说明"本图为概率密度分布，颜色深浅反映相对密度"

> **修复后再上传为 Supplementary Material**，否则 reviewers 可能质疑图示有效性。

### 2. S11 Bootstrap稳定性（已解决 ✅）

**更新后数据**（100次迭代，200k子样本）：

| K | ARI Mean | ARI Std | ARI Min | ARI Max |
|---|----------|---------|---------|---------|
| 3 | 0.866 | 0.040 | 0.470 | 0.872 |
| 4 | **0.886** | **0.041** | 0.751 | 0.950 |
| 5 | 0.814 | 0.043 | 0.671 | 0.886 |
| 6 | 0.800 | 0.043 | 0.683 | 0.869 |
| 7 | 0.671 | 0.036 | 0.601 | 0.778 |
| 8 | 0.604 | 0.043 | 0.504 | 0.722 |

K=4的ARI均值最高（0.886）且标准差合理（0.041），100次迭代结果稳健。图已重新生成，样式与现有SI图统一（无大标题，子图标注(a)(b)）。**推荐上传为 FigS11.tif**。

### 3. 代码来源说明

| 脚本 | 修改时间 | 生成内容 | 依赖数据 |
|------|---------|---------|---------|
| `generate_all_figures.py` | 2024-06-15 | 主图1-6，S3/S4/S5/S6/S7/S10 | `cluster4_story_dataset.npz`, `cluster4_microphysics_stats.csv`, `cluster4_environment_stats.csv`, `cluster4_significance_tests.csv` |
| `generate_supplementary_figures_v2.py` | 2024-06-16 | S1, S2, S8, S9, S11 | `Z_matrix.npy`, `stability_bootstrap_results.csv`, `cluster4_story_dataset_plus_dsd25.npz` |

---

## 九、投稿前最终检查清单（Checklist）

### 文件上传检查
- [ ] Manuscript（`.docx`）已最终定稿，含所有表格、参考文献、声明
- [ ] Cover Letter（`.docx`）已撰写，说明科学贡献与期刊适配性
- [ ] Highlights（`.docx`）3–5 条，每条 ≤85 字符，以 bullet point 格式
- [ ] Graphical Abstract（`.tif` 或 `.jpg`）≥531×1328 px，无图注，字体大而清晰
- [ ] Figure 1–6 各一张独立文件（`.tif` 或 `.eps`），编号命名
- [ ] Supplementary Figure S1–S11 各一张独立文件（`.tif` 或 `.eps`），编号命名
- [ ] Supplementary Table S1–S5 各一张独立文件（`.csv` 或 `.xlsx`），编号命名
- [ ] 代码包（`.zip`）已准备（可选但强烈推荐）

### 系统声明检查
- [ ] Declaration of Competing Interest（系统内完成）
- [ ] CRediT Author Contributions（系统内或文稿内完成）
- [ ] Data Availability Statement（已嵌入文稿末尾）
- [ ] Funding Statement（已嵌入文稿末尾，或待补充）
- [ ] Ethics Statement（本研究不涉及人类/动物，可声明 "Not applicable"）
- [ ] ORCID 账号已关联所有作者
- [ ] Suggested Reviewers（3–5 位，已准备姓名、单位、邮箱）

### 格式检查
- [ ] 所有图片分辨率为 300 DPI 以上（印刷标准）
- [ ] 所有图片字体 ≥8 pt，线条 ≥0.5 pt
- [ ] 正文中图引用顺序与上传编号一致
- [ ] 补充材料文件名与文稿中引用一致（如 "see Supplementary Fig. S1"）
- [ ] 参考文献格式符合 Atmospheric Research 要求（Harvard 风格）

---

> 说明：结果分析文件中的图引用编号已同步更新（S3小提琴、S4各簇空间、S5主导类型、S6冰相、S7亮带、S8 DSD亮带、S9 DSD风暴顶、S10效应量、S11 Bootstrap）。
