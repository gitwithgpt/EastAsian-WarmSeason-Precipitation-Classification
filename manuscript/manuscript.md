# 基于GPM DPR的东亚暖季降水客观分类：两阶段层次聚类方法

**作者** 学号

国防科技大学 气象海洋学院, 湖南 长沙 410073

---

## Highlights

• Objective classification of East Asian warm-season precipitation into four types using GPM DPR.

• Two-stage hierarchical k-means clustering with 22-dimensional microphysical parameters.

• Independent physical validation via ERA5 reanalysis environmental fields.

• Spatiotemporal patterns of precipitation types revealed across East Asian monsoon region.

---

## 摘 要

降水系统从纯层状到深对流呈现连续谱特征，其垂直结构蕴含大量微物理信息。准确分类是理解云-降水物理机制及验证数值天气预报方案的必要前提。本研究利用2014–2024年共十余年GPM DPR（Global Precipitation Measurement Dual-frequency Precipitation Radar，全球降水测量双频降水雷达）暖季（4–9月）东亚观测资料，采用两阶段层次k-means聚类方法，选取包含反射率廓线、雨滴大小分布参数、回波顶高及降水类型占比在内的22维雷达参数，结合ERA5再分析资料中的对流有效位能、整层可降水量和垂直风切变等进行分类验证。研究识别了东亚暖季主要四类降水类型，刻画了其垂直结构和微物理特征，评估了分类稳健性，并揭示了各类降水在东亚季风区的时空分布规律。

**关键词：** 降水分类；GPM DPR；K-means聚类；东亚季风；雨滴大小分布

---

## Abstract

Precipitation systems exhibit a continuous spectrum from pure stratiform to deep convection, with their vertical structures containing substantial microphysical information. Accurate classification is essential for understanding cloud-precipitation physics and validating numerical weather prediction schemes. This study utilizes over ten years (2014–2024) of GPM DPR warm-season (April–September) observations over East Asia, applying a two-stage hierarchical k-means clustering method with 22-dimensional radar parameters including reflectivity profiles, drop size distribution parameters, echo top height, and precipitation type fractions. ERA5 reanalysis data including convective available potential energy, total column water vapor, and vertical wind shear are employed for independent physical validation. Four major precipitation types are identified, their vertical structures and microphysical characteristics are characterized, classification robustness is assessed, and the spatiotemporal distributions across the East Asian monsoon region are revealed.

**Keywords:** Precipitation classification; GPM DPR; K-means clustering; East Asian monsoon; Drop size distribution

---

## 1. 引言

降水是全球水循环的关键组成部分，源于多尺度动力过程和微物理过程之间的复杂相互作用。不同降水系统的空间分布和出现频率直接反映大气的热力学和动力学状态：对流降水（convective precipitation）增加通常指示对流活动扩展和大气不稳定性增强，层状降水（stratiform precipitation）变化则与季风推进、锋面系统和大尺度垂直运动密切相关（Houze, 1997; Pendergrass and Hartmann, 2014; Varga-Balogh et al., 2025）。准确识别降水类型并刻画其物理属性，是理解云-降水机制、改进定量降水估计以及优化数值天气预报模式微物理参数化方案的基础。降水类型分类在气候监测方面也有实用价值，对流与层状降水相对频率的长期变化为诊断大尺度环流转变和水循环对气候强迫的响应提供依据。对流降水和层状降水在形成机制、垂直结构和微物理特征上存在显著差异（Houze, 1989; Rasmussen and Houze, 2016）。对流降水由强烈局地垂直运动驱动，具有高耸云塔、活跃冰相过程和显著水平反射率不均匀性；层状降水由大尺度斜压强迫主导，表现为水平范围广阔但垂直发展受限、结构相对均匀的云系。两类降水在地表水文影响方面也有差异：强烈局地对流可通过快速产流引发山洪和城市内涝，连续性层状降水则通过缓慢入渗成为区域水资源补给的重要来源。大气中降水系统并不遵循简单二元划分，而是呈现从浅对流到深对流、从孤立单体到有组织对流系统、从纯层状到对流嵌入层状降水的连续谱特征（Mapes and Houze, 1993）。这种内在连续性增加了基于观测数据的客观分类难度，需要能够捕捉降水微物理完整多样性而不施加人为分类边界的方法。

降水分类方法长期沿着两条路径平行发展：基于雨滴大小分布（drop size distribution, DSD）的方法和基于雷达反射率阈值的分类方法。早期DSD研究在Nw-Dm参数空间中建立海洋-大陆二分框架，其中海洋性对流以质量加权平均直径（mass-weighted mean diameter, Dm）适中、归一化截距参数（normalized intercept parameter, Nw）较高为特征，反映暖雨过程主导；大陆性对流则表现为更大Dm和更低Nw，与活跃冰相过程相关（Bringi et al., 2003; Ulbrich and Atlas, 2007）。随后在不同气候区开展的全球观测揭示大量介于两个理想化类别之间的过渡性DSD状态（Thurai et al., 2017a, 2017b; Tokay and Short, 1996; Tokay et al., 2008），表明真实大气中微物理变率远比离散分类方案能充分刻画的更为复杂。针对这一挑战，Dolan et al. (2018)运用主成分分析（principal component analysis, PCA）从全球雨滴谱仪观测中识别出六种不同DSD模态，将分类粒度从二元系统推进到六类系统。从卫星遥感视角，Schumacher and Funk (2023)分析GPM DPR全球观测资料，揭示对流降水三模态结构，包括海洋性深对流、大陆性深对流和浅对流，三者回波顶高、降水强度和垂直结构均存在差异。基于反射率阈值的分类方法从简单二维水平判据逐步发展到精细化三维结构方法。Churchill and Houze (1984)率先利用反射率梯度和峰值检测识别对流降水，Steiner et al. (1995)随后开发被广泛采用的SHY95算法，成为气候学研究标准参考。这些早期方法存在共同局限：仅利用二维水平信息，忽视垂直结构，而垂直结构是区分对流与层状降水的关键维度。固定反射率阈值对区域依赖性构成另一挑战，Liu and Zipser (2015)研究表明最优对流降水阈值从热带海洋区约20 dBZ到热带大陆和亚热带区35–40 dBZ不等，反映不同气候区对流系统发展强度和微物理过程差异。针对硬分类局限，Dixon and Romatschke (2022)提出ECCO（Enhanced Convective-Stratiform Classification and Organization）算法，基于三维反射率纹理将对流性量化为0–1连续变量，与GPM DPR官方分类达到84–88%一致性，同时揭示10–15%降水事件难以明确归类为纯对流或纯层状。

2014年发射的GPM（Global Precipitation Measurement，全球降水测量）核心卫星搭载DPR（Dual-frequency Precipitation Radar，双频降水雷达），为客观降水分类研究提供了长期观测数据。与其前身TRMM PR相比，DPR在多项技术指标上实现提升，尤其适用于微物理分类研究：Ku和Ka双波段协同观测改善弱降水事件探测能力，探测能力约为PR的1.7倍；云-噪声信号误判率从10.14%大幅降低至0.51%；2018年升级为全扫描（Full Scan）模式后，双频观测刈幅宽度从约120 km扩展至245 km（Awaka et al., 2021）。随着V6至V7/V8产品版本算法改进以及截至2024年已积累的十余年连续全球观测，DPR为大样本客观降水分类研究提供了数据基础。这些数据优势推动基于多维DPR参数的聚类分类研究。K-means聚类因其简洁性和计算效率被广泛采用：Luo et al. (2017)将k-means应用于TRMM、CloudSat和CALIPSO近同步观测，识别出有组织深对流、对流抑制厚砧降水和对流抑制系统三种主要模态；Zhang et al. (2026)结合主成分分析降维与k-means算法对全球GPM DPR观测分类，识别出高纬浅层、副热带浅层、中等、深对流、极端深对流、强降水、极端强降水和海洋性极端降水共八类降水系统，揭示全球不同气候区降水微物理属性系统差异。高斯混合模型（Gaussian Mixture Models, GMM）提供互补的软分类能力：Ryu et al. (2021)利用GMM基于雨滴大小分布将全球大雨划分为大陆性、海洋性深对流和海洋性浅对流三种类型；Ryu et al. (2025)进一步采用GMM基于风暴高度将温带地区大滴大雨区分为高风暴高度（high storm height, HSH）和低风暴高度（low storm height, LSH）两类，揭示温带气旋中独特降水微物理结构。在东亚季风区，研究主要聚焦特定天气系统或子区域：Yamaji and Takahashi (2023)识别亚洲季风区降水微物理显著季节变化，Dm和重冰相降水频率在季风前期与盛期之间差异明显；Aoki et al. (2026)利用EarthCARE/CPR和GPM/DPR联合观测数据集，基于回波顶高度将降水分为深层层状、中等对流、高大对流和浅层四种类型，揭示对流和层状降水在垂直运动和微物理特征上的差异，体现多传感器联合观测在降水分类中的潜力。这些研究仍局限于个例分析或特定传感器组合，缺乏在整个东亚季风区覆盖完整暖季、仅基于GPM DPR单源数据的系统客观分类统一方法框架。

上述多种方法虽取得进展，当前降水分类研究中仍存在若干系统性不足，制约区域降水多样性的认识。现有研究对可用特征维度利用多不充分。Gatlin and Stough (2023)从DPR中提取完整垂直反射率信息，但在PCA降维后低维空间中执行聚类，重要垂直结构细节可能在降维过程中丢失。现有方法大多缺乏层次结构和过渡状态识别能力。传统k-means算法强制将每个样本归入单一离散类别（Lloyd, 1982），这与全球DSD综合分析揭示的降水微物理连续性观测事实不符（Wen et al., 2026）。GMM提供概率性软分类，但计算复杂度可能限制面向区域气候学研究所需大样本卫星数据集的实际可扩展性；层次聚类方法（Murtagh and Contreras, 2012）在处理数百万DPR廓线样本时面临计算效率挑战。针对东亚季风区的系统性分类仍然缺失。该气候复杂区域涵盖南海海洋性对流、华南前汛期降水、江淮梅雨、华北对流降水和青藏高原地形降水等多种降水体系，均受多时空尺度复杂相互作用调制。虽有研究刻画特定区域或季节降水微物理特征（Yamaji and Takahashi, 2023; Wen et al., 2023），现有研究多聚焦特定时段而未覆盖4至9月完整暖季。验证方法主要局限于轮廓系数、Davies-Bouldin指数等内部统计指标（Davies and Bouldin, 1979）或与GPM DPR官方分类产品的外部比对，本质上属于自洽性验证而非独立物理验证。利用完全独立于DPR观测的大气环境参数对分类结果进行系统性交叉验证在现有文献中罕见，制约对所识别降水类别物理可解释性的信心。

针对上述研究空白，本研究利用2014–2024年共十余年GPM DPR暖季（4–9月）东亚季风区连续观测资料，采用两阶段层次k-means聚类方法，选取包含多层Ku和Ka波段反射率廓线、DSD参数（Dm和Nw）、回波顶高、亮带特征和降水类型占比在内的22维雷达微物理特征参数。分类完成后，利用ERA5再分析资料中的对流有效位能（convective available potential energy, CAPE）、整层可降水量、垂直风切变和对流抑制（convective inhibition, CIN）等环境场参数，检验各识别降水类型对应的气象条件，提供不依赖于DPR反演量的独立物理验证。本研究追求三个主要目标：（1）基于22维微物理特征空间识别东亚暖季主要降水类型，刻画其三维垂直结构和微物理特征；（2）通过两阶段层次聚类框架结合多种验证指标，系统评估分类稳健性和稳定性；（3）揭示各降水类型对应环境场特征，定量刻画其在东亚季风区4至9月的空间分布格局和季节演变。本研究创新之处包括四个方面。数据基础上，完整十余年GPM DPR观测覆盖整个东亚暖季且空间采样密集，样本量充分，时空覆盖好。方法论上，两阶段层次k-means聚类框架通过22维参数空间充分挖掘DPR垂直廓线信息，并引入ERA5环境场验证，建立分类结果物理可信度。科学贡献上，在东亚季风区内识别并刻画包括过渡态和混合态在内的多种暖季降水类型，记录其微物理特征、环境控制因子和时空演变规律。应用价值上，所建立的降水类型分类体系及其微物理气候学特征为评估和改进东亚地区数值天气预报模式微物理参数化方案提供观测约束。

---

## 2. 数据与方法

### 2.1 数据来源

本研究采用卫星雷达观测资料和环境场再分析资料。数据基本信息汇总于表1。

**表1 研究数据来源**

| 数据类型 | 具体产品 | 时间范围 | 空间范围 | 空间分辨率 |
|---------|---------|---------|---------|----------|
| 卫星雷达 | GPM DPR Ku/Ka双频雷达V07产品 | 2014–2024年暖季（4–9月） | 东亚地区（15°–54°N, 105°–135°E） | 约5 km（水平）/ 250 m（垂直） |
| 环境场 | ERA5再分析资料 | 同期 | 匹配降水样本 | 0.25° × 0.25° |

GPM DPR数据由NASA戈达德航天中心和日本宇宙航空研究开发机构（JAXA）联合提供（Hou et al., 2014; Skofronick-Jackson et al., 2017）。DPR搭载Ku波段（13.6 GHz）和Ka波段（35.5 GHz）两部雷达，采用双频协同观测以获取降水粒子微物理信息（Iguchi et al., 2012; Meneghini et al., 2015）。本研究采用DPR二级（L2）产品中的反射率因子（Z）、雨滴大小分布参数（Dm、Nw）、降水率（precipRate）及降水类型标记等核心变量（Seto et al., 2021）。

差分频率比（dual-frequency ratio, DFR）定义为：

$$DFR = Z_{Ku} - Z_{Ka} \tag{1}$$

式中，$Z_{Ku}$和$Z_{Ka}$分别为Ku波段和Ka波段反射率因子（单位dBZ）。DFR可反映降水粒子尺寸信息。2018年升级为全扫描（Full Scan）模式后，DPR双频观测刈幅宽度扩展至约245 km。

ERA5再分析数据由欧洲中期天气预报中心（ECMWF）通过哥白尼气候变化服务（C3S）门户提供（Hersbach et al., 2020）。该数据集同化卫星观测、探空资料、地面观测等多源数据，通过四维变分数据同化系统生成，时空一致性和物理协调性较高。本研究提取与降水微物理特征相关的环境参数，包括热力参数、水汽参数、动力参数、地形参数和云参数。

### 2.2 微物理特征参数

本研究从GPM DPR L2产品中提取22维微物理特征参数，涵盖反射率特征、垂直结构特征、相态结构特征、粒径分布特征和降水强度五个维度（表2）。参数选择原则如下：（1）充分利用DPR双频观测优势，兼顾Ku和Ka波段互补信息；（2）包含垂直廓线形态信息，避免仅用单一高度参数；（3）纳入DSD参数以刻画微物理本质属性（Testud et al., 2001; Bringi et al., 2003）；（4）包含相态结构参数以区分冰相和液相过程。

**表2 22维微物理特征参数**

| 类别 | 参数 | 符号 | 单位 | 物理意义 |
|------|------|------|------|---------|
| 反射率特征 | 近地表Ku反射率 | zku_ns | dBZ | 地表附近降水粒子散射强度 |
| | 近地表Ka反射率 | zka_ns | dBZ | 地表附近降水粒子散射强度 |
| | 峰值Ku反射率 | zku_max | dBZ | 垂直廓线中最大反射率强度 |
| | 峰值Ka反射率 | zka_max | dBZ | 垂直廓线中最大反射率强度 |
| | 近地表差分反射率 | dfr_ns | dB | 近地表粒子尺寸信息 |
| | 峰值差分反射率 | dfr_max | dB | 最大粒子尺寸信号 |
| 垂直结构特征 | 风暴顶高 | heightStormTop | m | 回波顶高度，指示对流发展深度 |
| | 0℃层高度 | heightZeroDeg | m | 融化层高度，反映热力环境 |
| | 自由层底高 | free_bottom_height | m | 亮带底部高度，指示液态降水起始层 |
| | Ku反射率梯度 | slope_zku | dBZ/km | 垂直廓线反射率变化率 |
| | Ka反射率梯度 | slope_zka | dBZ/km | 垂直廓线反射率变化率 |
| | DFR梯度 | slope_dfr | dB/km | 差分频率比垂直梯度 |
| | 峰值Ku反射率高度 | zku_max_height | m | 最大反射率出现高度 |
| | 峰值Ka反射率高度 | zka_max_height | m | 最大反射率出现高度 |
| | 峰值DFR高度 | dfr_max_height | m | 最大粒子尺寸信号高度 |
| 相态结构特征 | 冰相层厚度 | ipl | m | 0℃层以上冰相过程活跃层厚度 |
| | 液相层厚度 | lpl | m | 0℃层以下液态降水层厚度 |
| | 亮带高度 | heightBB | m | 融化层增强回波高度 |
| | 亮带厚度 | bb_thickness | m | 融化层增强回波厚度 |
| 粒径分布特征 | 质量加权平均直径 | dm_column_max | mm | 雨滴质量加权平均直径 |
| | 广义截距参数 | nw_column_max | mm⁻¹m⁻³ | 雨滴浓度归一化参数 |
| 降水强度 | 近地表降水率 | precipRateNearSurface | mm/h | 地表附近降水强度 |

反射率梯度通过线性回归拟合垂直廓线斜率获得。设廓线采样点高度为$h_i$，对应反射率因子为$Z_i$，则斜率计算公式为：

$$slope = \frac{\sum_{i}(h_i - \bar{h})(Z_i - \bar{Z})}{\sum_{i}(h_i - \bar{h})^2} \tag{2}$$

式中，$\bar{h}$和$\bar{Z}$分别为高度和反射率因子的样本均值。该式为最小二乘回归系数，量纲为dBZ/km，反映反射率因子随高度的变化率。亮带参数由DPR算法自动识别。所有参数均从DPR L2产品原始数据中提取，未经过额外平滑或插值处理。

### 2.3 环境场参数

本研究将每个GPM DPR降水像元与同期ERA5再分析资料进行空间最近邻匹配。ERA5环境场参数共提取14个变量（表3）。

**表3 ERA5环境场参数**

| 类别 | 参数 | 符号 | 单位 | 物理意义 |
|------|------|------|------|---------|
| 热力参数 | 2米温度 | t2m | K | 地表附近热力状态 |
| | 对流有效位能 | cape | J/kg | 大气不稳定能量 |
| | 对流抑制 | cin | J/kg | 抑制对流发展的能量 |
| 水汽参数 | 总柱水汽 | tcwv | kg/m² | 整层大气水汽含量 |
| | 总柱水 | tcw | kg/m² | 整层大气液态水含量 |
| | 850 hPa相对湿度 | rh850 | % | 中低层湿度条件 |
| 动力参数 | 10米经向风速 | u10 | m/s | 近地面风场 |
| | 10米纬向风速 | v10 | m/s | 近地面风场 |
| | 边界层高度 | blh | m | 对流边界层发展高度 |
| 地形参数 | 地表气压 | sp | hPa | 反映海拔高度信息 |
| | 地表高度 | z_surf | m | 地形海拔 |
| 云参数 | 云底高 | cbh | m | 云系底部高度 |
| | 总云量 | tcc | 0–1 | 天空被云覆盖比例 |
| 温度层结 | 0℃层高度 | deg0l | m | 融化层高度（与DPR独立来源） |

### 2.4 GPM-ERA5数据匹配过程

本研究开发了多GPU并行数据匹配框架，处理流程如下。

（1）时间信息提取：从GPM DPR L2产品的HDF5文件中读取FS（Full Scan）组下的ScanTime字段，提取每个扫描像元的年、月、日、时信息，确保与ERA5时间坐标一致。

（2）GPM数据筛选与加载：根据研究区域范围（15°–54°N, 105°–135°E）和最低降水率阈值（0.5 mm/h）筛选有效降水像元。采用分块读取策略处理大数组（如4维雷达反射率因子zFactorMeasured），避免内存溢出。为每个有效像元提取47维GPM特征向量，包括近地表反射率、垂直廓线特征、风暴顶高、0℃层高度、融化层参数、雨滴谱参数（$D_m$、$N_w$）等。

（3）ERA5数据查找与加载：压力层数据（37层）按年月日查找对应ERA5文件，空间裁剪至研究区域，加载位势高度、温度、比湿、风速分量、垂直速度等14个变量；单层数据（22个变量）按月查找ERA5单层文件，加载2米温度、CAPE、CIN、总柱水汽、云底高度、边界层高度等。

（4）最近邻空间匹配：将每个GPM有效降水像元的地理位置（经纬度）与ERA5网格进行最近邻空间匹配。每个压力层变量匹配27层 × 14变量 = 378维数据，单层变量匹配22维数据。

（5）构建完整匹配向量：将GPM特征与ERA5压力层特征、ERA5单层特征拼接为匹配向量。匹配向量维度明确如下：

$$V_{matched} = \left[V_{GPM}^{47}, V_{ERA5-PL}^{378}, V_{ERA5-SL}^{22}\right] \in \mathbb{R}^{447} \tag{3}$$

式中，$V_{GPM}^{47}$为GPM DPR微物理特征向量（47维），$V_{ERA5-PL}^{378}$为ERA5压力层环境场向量（378维），$V_{ERA5-SL}^{22}$为ERA5单层环境场向量（22维）。$V_{matched}$为拼接后的完整匹配向量，维度为447维。该向量按月增量保存为npz格式，避免内存堆积。

（6）多GPU并行加速：实现多GPU并行处理框架（基于PyTorch和torch.multiprocessing），按月分配处理任务。单个GPU工作进程使用结果队列实时传递数据，主进程统一收集并保存。经测试，该框架将完整数据处理时间从单GPU的约120小时缩短至4 GPU的约35小时，加速比达3.4倍。

### 2.5 聚类方法

本研究采用两阶段层次聚类（Hierarchical Clustering）框架对东亚暖季降水分类（Ward, 1963; Murtagh and Contreras, 2012）。第一阶段对全部有效样本执行层次聚类以获取初始聚类结构；第二阶段在各类别内部进行稳定性验证和敏感性分析。

聚类采用欧氏距离作为样本间相似性度量。设两个样本分别为$x = (x_1, x_2, \ldots, x_n)$和$y = (y_1, y_2, \ldots, y_n)$，欧氏距离定义为：

$$d(x, y) = \sqrt{\sum_{i=1}^{n}(x_i - y_i)^2} \tag{4}$$

式中，$n$为特征维度（本研究中$n = 22$），$x_i$和$y_i$为样本$x$和$y$在第$i$维上的取值。欧氏距离对连续型变量敏感，适用于本研究中经标准化后的微物理特征参数。

层次聚类采用Ward最小方差法（Ward's minimum variance method）作为合并准则。设簇$A$和簇$B$合并为新簇$AB$，合并前后总类内离差平方和的增量为：

$$\Delta E = \frac{n_A n_B}{n_A + n_B} \|\bar{x}_A - \bar{x}_B\|^2 \tag{5}$$

式中，$n_A$和$n_B$分别为簇$A$和簇$B$的样本数，$\bar{x}_A$和$\bar{x}_B$分别为两簇的样本均值向量，$\|\cdot\|$表示欧氏范数。Ward法在每一步选择使$\Delta E$最小的两个簇进行合并，倾向于生成大小相近、轮廓紧凑的簇，适合处理具有连续分布特征的微物理参数。

特征标准化在聚类前进行。所有22维微物理特征参数均进行Z-score标准化，以消除量纲差异对距离计算的影响：

$$z = \frac{x - \mu}{\sigma} \tag{6}$$

式中，$x$为原始参数值，$\mu$和$\sigma$分别为该参数在全样本中的均值和标准差。标准化后各参数均值为0，标准差为1。

聚类数通过多指标综合评估确定。本研究采用轮廓系数（Silhouette Coefficient, Rousseeuw, 1987）、Davies-Bouldin指数（Davies-Bouldin Index, Davies and Bouldin, 1979）和Calinski-Harabasz指数（Calinski and Harabasz, 1974）评估聚类质量。

轮廓系数衡量样本与所属簇的契合程度。对于样本$i$，设$a(i)$为$i$与同簇其他样本的平均距离，$b(i)$为$i$与最近邻簇中所有样本的平均距离，则轮廓系数定义为：

$$s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}} \tag{7}$$

式中，$a(i)$反映簇内紧密度，$b(i)$反映簇间分离度。$s(i)$取值范围为$[-1, 1]$，值越大表明样本聚类越合理。全样本平均轮廓系数越接近1，聚类效果越好。

Davies-Bouldin指数评估簇间分离度与簇内紧密度的比值。设聚类数为$K$，$c_i$为第$i$个簇的质心，$\sigma_i$为第$i$个簇内样本到质心的平均距离，$d(c_i, c_j)$为质心$c_i$与$c_j$之间的距离，则：

$$DB = \frac{1}{K}\sum_{i=1}^{K} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)} \tag{8}$$

式中，分子$\sigma_i + \sigma_j$反映簇$i$和簇$j$的分散程度，分母$d(c_i, c_j)$反映两簇质心间距离。$DB$值越小，簇内越紧凑且簇间越分离，聚类质量越高。

Calinski-Harabasz指数评估类间离散与类内离散的比值：

$$CH = \frac{Tr(B_K)/(K-1)}{Tr(W_K)/(N-K)} \tag{9}$$

式中，$N$为总样本数，$K$为聚类数，$B_K$为类间离散矩阵，$W_K$为类内离散矩阵，$Tr(\cdot)$表示矩阵迹。$CH$值越大，类间分离越明显，聚类效果越好。

结合上述指标与物理可解释性分析，最终确定最优聚类数为$K = 4$。四类降水系统的物理特征与组会分析结果一致（详见后文）。

### 2.6 统计验证方法

本研究采用三种非参数统计方法评估各类别间差异的显著性和实际重要性（Mann and Whitney, 1947; Kolmogorov, 1933; Smirnov, 1948; Cliff, 1993）。

Mann-Whitney U检验用于比较两组独立样本的分布差异。设两组样本量分别为$n_1$和$n_2$，将两组样本混合排序后第一组样本的秩和为$R_1$，则U统计量计算公式为：

$$U = n_1 n_2 + \frac{n_1(n_1+1)}{2} - R_1 \tag{10}$$

式中，$n_1 n_2$为两组样本量的乘积，$\frac{n_1(n_1+1)}{2}$为$R_1$的理论最小值。U统计量服从近似正态分布，当$p < 0.001$时判定差异统计显著。该方法不要求数据服从正态分布，适用于本研究中各类别微物理参数的分布比较。

Kolmogorov-Smirnov检验用于判断两组样本是否来自同一分布。设两组样本的经验分布函数分别为$F_1(x)$和$F_2(x)$，则K-S统计量为：

$$D = \sup_x |F_1(x) - F_2(x)| \tag{11}$$

式中，$\sup_x$表示对所有$x$取上确界，$|F_1(x) - F_2(x)|$为两经验分布函数在点$x$处的绝对差值。$D$反映两分布的最大偏离程度。当$p < 0.001$时判定差异统计显著。K-S检验对分布形状敏感，可检测位置、尺度及形态差异。

Cliff's Delta衡量差异的实际重要性（效应量）。设随机抽取一个来自组$X$的样本大于来自组$Y$的样本的概率为$P(X > Y)$，则：

$$d = 2 \times P(X > Y) - 1 \tag{12}$$

式中，$d$取值范围为$[-1, 1]$，绝对值越大表明效应越强。效应量等级判定：$|d| > 0.33$为中等效应，$|d| > 0.474$为大效应，$|d| > 0.741$为极大效应。该指标对异常值稳健，适用于非正态数据。

所有配对比较的样本量均超过90万（如C0 vs C1：230万 vs 281万），统计功效较高，即使微小分布差异也能被检测到。

---

## 3. 结果

### 3.1 聚类数确定与样本分布

基于两阶段层次k-means聚类方法，对2014–2024年暖季东亚地区GPM DPR有效降水样本（n = 9 618 893）进行分类。阶段一针对k-means超参数k_over（层次聚类第一阶段覆盖采样比例）进行优化，固定K=5，在20万随机子样本上测试k_over = 50–300的取值范围。评估指标包括Davies-Bouldin指数（DB，越低越好）、轮廓系数（Silhouette，越高越好）和Calinski-Harabasz指数（CH，越高越好）。结果表明，k_over = 150时DB指数最低（1.257），轮廓系数较高（0.334），且最小类样本量为14 580，满足各类别统计代表性要求，据此确定最优k_over = 150。

阶段二在k_over = 150固定条件下，对全样本评估K = 3–10的聚类质量（表4）。K = 4时DB指数达到最小值（1.146），轮廓系数为0.334，CH指数为88 414.4；K = 3时轮廓系数相近但DB指数较高（1.179），聚类分离度不足；K = 5时DB指数明显上升（1.257），且最小类样本量降至14 580，类别不平衡加剧；K ≥ 6时DB指数持续上升、轮廓系数下降，且最小类样本量降至3 628–4 739，统计可靠性显著降低。因此，综合DB指数拐点、轮廓系数稳定区间及类别平衡性，确定最优聚类数K = 4。树状图（图S2）显示，在Ward最小方差连接准则下，全样本首先分为两大支系，随后各支系进一步二分，形成四个主要分支，与k-means阶段的四分类结构一致，表明层次结构的稳定性。

**表4 聚类数K = 3–8评估指标汇总**

| K | DB指数 | 轮廓系数 | CH指数 | 最小类样本量 | 最大类样本量 |
|---|--------|----------|--------|-------------|-------------|
| 3 | 1.179 | 0.334 | 103 394.8 | 48 041 | 93 732 |
| 4 | **1.146** | **0.334** | **88 414.4** | 14 580 | 79 152 |
| 5 | 1.257 | 0.334 | 81 645.0 | 14 580 | 79 152 |
| 6 | 1.188 | 0.336 | 71 087.8 | 4 739 | 79 152 |
| 7 | 1.224 | 0.302 | 71 211.8 | 4 739 | 58 227 |
| 8 | 1.239 | 0.296 | 63 972.1 | 3 628 | 58 227 |

注：DB指数越低越好，轮廓系数和CH指数越高越好。最优K = 4以粗体标注。

K = 4聚类将全样本分为四类：C0（深厚弱对流/层状云混合，2 303 779，24.0%）、C1（亮带主导的层状云，2 810 153，29.3%）、C2（强对流，1 132 609，11.8%）、C3（浅薄降水/弱对流，3 372 352，35.1%）。四类样本量均超过百万，为后续统计分析提供了充分的统计功效。

### 3.2 微物理特征

#### 3.2.1 总体微物理差异

图1展示了四类在关键微物理参数上的箱线分布，包括近地表降水率、风暴顶高、近地表Ku反射率、峰值Ku反射率、冰相层厚度、亮带高度、最大质量加权平均直径（Dm）和最大广义截距参数（Nw）。由图可见，四类在垂直结构深度（风暴顶高）、降水强度、反射率强度及微物理属性上呈现系统差异。C2在风暴顶高（~8 400 m）、降水率（~8.8 mm h⁻¹）、峰值反射率（~38.5 dBZ）和Dm（~2.2 mm）上均为最高，表现出典型的深对流特征；C3在各参数上均为最低，风暴顶高约5 000 m，降水率约1.6 mm h⁻¹。C0和C1介于两者之间，但C1具有显著的亮带结构（亮带高度~4.2 km），而C0的亮带出现率较低。补充图S3（微物理参数小提琴图）展示了分布形态的更多细节，证实了上述参数在四类间不仅中位数差异显著，且分布形态亦存在系统性偏移。

图S6（冰相参数箱线图）进一步展示了冰相层厚度、液相层厚度、亮带高度和亮带厚度四类的对比。C2冰相层厚度最大（中位数~3 500 m），C3最小（~1 300 m），C0和C1居中（~2 000–2 700 m）。C1亮带高度最高（中位数~4.2 km），C0约为2.0 km，C2和C3接近0 km。亮带厚度方面，C1为薄亮带（负值表示薄），C0为厚亮带，C2和C3无亮带。这些冰相结构特征与图1的微物理参数一致，从相态结构角度独立验证了分类的物理基础。

图S7（亮带出现率）显示，C1亮带出现率接近100%，C0约48%，C2和C3接近0%。结合图S6的亮带高度统计，C1是唯一具有完整亮带结构的类别，这与其层状云降水的物理属性一致。C0的部分样本具有亮带而部分无亮带，表明其对流与层状混合的过渡性质。

#### 3.2.2 各类别详细特征

C0（深厚弱对流/层状云混合）的主导特征为峰值Ka反射率高度显著偏高（z-score = +1.73）和峰值DFR高度显著偏高（z-score = +1.71），同时自由层底高度偏低（z-score = −1.23）。降水粒子在较高高度达到最大尺寸，自由层底较低，云层较为深厚。近地表降水率3.56 mm h⁻¹（中等），风暴顶高6 307 m。冰相层厚度中位数~2 200 m，亮带出现率约48%，约半数样本具有亮带结构（层状云成分），其余无亮带（对流成分）。Dm和Nw均接近总体均值（z-score ≈ −0.4），无极端微物理特征，整体呈现对流与层状混合的过渡性质。

C1（亮带主导的层状云）的主导特征为峰值Ku反射率高度显著偏高（z-score = +1.66）、亮带高度显著偏高（z-score = +1.54）和亮带厚度显著偏薄（z-score = −1.54）。峰值Ku反射率高度较高（~3 730 m）与亮带高度（~4.2 km）相呼应，表明该类降水以融化层为核心结构层。近地表降水率2.76 mm h⁻¹（四类中最弱），风暴顶高7 316 m。冰相层厚度~2 700 m，亮带出现率接近100%，是四类中唯一具有几乎完整亮带结构的类别。亮带高度最高、厚度较薄，典型的大尺度层状云降水结构，对应梅雨锋和准静止锋系统。

C2（强对流）的主导特征极为鲜明：广义截距参数Nw显著偏低（z-score = −1.72），近地表DFR显著偏高（z-score = +1.71），质量加权平均直径Dm显著偏大（z-score = +1.68），近地表降水率显著偏高（z-score = +1.67），近地表Ku反射率显著偏高（z-score = +1.66）。这些特征共同表明大粒子（Dm高）、低浓度（Nw低）的滴谱特征，与强烈冰相过程和大雨滴碰并增长一致。降水率最强（8.83 mm h⁻¹），风暴顶最高（8 436 m），冰相层最厚（~3 500 m），亮带出现率接近0%，表明强烈对流破坏了融化层结构。典型天气系统包括台风、飑线和强季风对流。

C3（浅薄降水/弱对流）的主导特征为Ku反射率梯度显著偏低（z-score = −1.71），液相层厚度显著偏薄（z-score = −1.61），峰值DFR显著偏低（z-score = −1.57）。降水率最弱（1.56 mm h⁻¹），风暴顶最低（5 054 m），液相层最薄（~2 600 m），亮带出现率接近0%。冰相层厚度仅~1 300 m，Dm和Nw均显著低于总体均值（z-score ≈ −0.9），表明粒子尺寸小，以暖雨过程为主。典型天气系统包括地形抬升降水和局地热对流，对流发展受限于边界层高度。

**表5 四类关键微物理参数统计特征（标准化z-score）**

| 参数 | C0 | C1 | C2 | C3 | 物理意义 |
|------|-----|-----|-----|-----|---------|
| precipRateNearSurface | −0.22 | −0.51 | +1.67 | −0.94 | 降水强度 |
| heightStormTop | −0.38 | +0.43 | +1.33 | −1.38 | 垂直发展深度 |
| zku_ns | −0.34 | −0.28 | +1.66 | −1.03 | 近地表散射强度 |
| zku_max | −0.25 | +0.09 | +1.48 | −1.32 | 峰值散射强度 |
| dm_column_max | −0.41 | −0.34 | +1.68 | −0.94 | 粒子平均尺寸 |
| nw_column_max | +0.77 | +0.51 | −1.72 | +0.45 | 粒子浓度 |
| heightBB | +0.23 | +1.54 | −0.89 | −0.89 | 亮带高度 |
| bb_thickness | −0.24 | −1.54 | +0.89 | +0.89 | 亮带厚度 |
| ipl | −0.27 | +0.26 | +1.39 | −1.39 | 冰相层厚度 |
| lpl | +0.04 | +0.48 | +1.09 | −1.61 | 液相层厚度 |

注：数值为各类别参数均值经全样本z-score标准化后的结果。正值表示高于总体均值，负值表示低于总体均值。

### 3.3 垂直结构特征

为刻画四类的三维垂直结构差异，本研究构建多频准CFAD图（图4），展示Ku波段最大反射率、Ka波段最大反射率及双频差（DFR）随高度和反射率的联合概率分布。与经典CFAD基于固定高度层的反射率直方图不同，本研究的准CFAD以最大反射率及其出现高度为坐标，反映每类降水的代表性垂直结构特征，中位数轨迹（实线）和四分位包络（虚线）标注于各面板。由于当前聚类特征集为22维参数汇总，不包含逐高度廓线数据，本图定位为结构代理而非严格剖面CFAD。

C0的Ku最大反射率峰值出现在2–4 km高度，约20–30 dBZ，向上逐渐减弱至12 km。Ka最大反射率峰值也在2–4 km，约20–25 dBZ，但向上减弱更快。低层（2–4 km）DFR约0–0.05 dB，高层接近0，表明粒子尺寸随高度减小，整体为深厚但强度中等的对流结构，兼具部分层状云特征。

C1的Ku和Ka最大反射率峰值均在4–6 km，约25–35 dBZ（Ku）和25–30 dBZ（Ka），恰好对应亮带高度（~4.2 km）。DFR在4–6 km有微弱增强（0.05–0.10 dB），反映融化层中粒子碰并增长导致的尺寸增大。向上至11 km反射率逐渐减弱，清晰展示了层状云降水的典型融化层增强特征。

C2的Ku最大反射率峰值在4–8 km，约35–45 dBZ，可延伸至14 km。Ka最大反射率峰值在4–6 km，约30–35 dBZ，但向上快速减弱。DFR在4–8 km显著增强，达到0.10–0.30 dB，且随高度增加（4 km约0.05 dB → 8 km约0.20 dB），表明强对流中高层存在大量大冰相粒子（冰雹、霰）。这是深对流的典型特征，与C2亮带出现率接近0%（融化层结构被强烈对流破坏）一致。

C3的Ku和Ka最大反射率峰值均在2–4 km，约20–25 dBZ（Ku）和20–22 dBZ（Ka），向上快速减弱至10 km。低层DFR接近0甚至略负（−0.02–0 dB），表明粒子尺寸小，以小雨滴为主。对流发展高度低，0℃层以上冰相过程弱，整体以暖雨过程为主。

图S8（补充图）展示了DSD参数（Dm和Nw）与亮带高度的联合分布，图S9展示了DSD参数与风暴顶高的联合分布。在图S8中，C1的亮带高度集中在4 km附近，与Dm约1.5 mm、Nw约35的DSD特征对应，反映层状云融化层中粒子增长的典型结构。C2的亮带高度接近0，而Dm分布在2.0 mm以上，Nw分布在32以下，对应深对流中活跃的冰相过程和碰并增长。图S9中，C2的风暴顶高分布最广（延伸至14 km），与Dm和Nw的极端值对应，进一步证实了C2的深对流结构。这些DSD结构代理图与图4的准CFAD结果互为补充，从不同维度验证了四类的垂直结构差异。

### 3.4 空间分布

图3展示了研究区域（15°–54°N, 105°–135°E）内有效降水样本的空间频率分布。降水样本主要集中在华南沿海地区、台湾海峡、长江中下游、江淮流域、四川盆地及黄淮地区，与东亚季风降水的核心区域高度吻合。高频区（>500个样本/0.5°网格）沿华南—江南—江淮呈东北—西南带状分布，与夏季风主雨带的推进路径一致。华北、东北及西北东部地区降水频率相对较低，青藏高原东部边缘有零星高频点，反映地形抬升降水的局地特征。

图S4展示了四类降水的各自空间分布。C0分布较为均匀，覆盖长江中下游、黄淮地区及四川盆地，空间范围广，无明显集中区。C1主要分布于华北、东北地区及江淮流域，中高纬度特征显著，与温带气旋和锋面系统的活动路径一致。C2高度集中于华南沿海、台湾海峡、东海及南海北部，沿海及海洋区域特征突出，与台风、季风对流的频繁活动区域一致。C3主要分布在西南地区、西北东部及青藏高原边缘，内陆及高原地形特征明显，与地形抬升和局地热对流相关。

图S5展示了每个0.5°网格内样本数最多的主导降水类型。华南及沿海地区以C2为主导，反映热带季风爆发的强烈对流活动。长江中下游地区以C0和C1交替主导，对应梅雨锋面和夏季风推进过程中的混合降水结构。华北及东北地区以C1占主导，反映中高纬度锋面系统的主导作用。西南及西北东部地区以C3占主导，表明地形降水在该区域的重要贡献。主导类型的地理分异与东亚季风区降水的气候学格局一致，从物理上支撑了聚类的区域代表性。

### 3.5 环境场特征

为验证聚类结果的物理可信度，本研究将每个GPM DPR降水像元与同期ERA5再分析资料进行空间最近邻匹配，提取14个环境参数，从热力、水汽、动力和地形四个维度评估各类别的环境特征。

图2展示了四类在关键环境参数上的箱线分布，包括2 m温度、CAPE、总柱水、总柱水汽、850 hPa相对湿度、0℃层高度和地表气压。四类在环境参数上呈现系统差异，与微物理特征和空间分布一致。C2在温度、CAPE、TCW和0℃层高度上均为最高；C1在湿度和云量上表现突出，但CAPE和边界层高度最低；C0各参数均居中；C3在大部分参数上为最低，但边界层高度相对较高。

图5展示了各类别环境参数经z-score标准化后的比较模式。C2在温度（z = +1.68）、CAPE（z = +1.72）、总柱水（z = +1.26）、总柱水汽（z = +1.34）、0℃层高度（z = +1.50）和地表气压（z = +1.39）上均为强正值，边界层高度亦较高（z = +1.10），仅云底高度（z = −1.55）和总云量（z = −1.05）为负值，表明高温、高湿、高不稳定、高0℃层的大气状态，最有利强对流发展。C1在850 hPa相对湿度（z = +1.51）和总云量（z = +1.57）上极高，云底高度较高（z = +1.21），但CAPE极低（z = −0.70）、边界层高度极低（z = −1.52），表明高湿度、弱不稳定、强云覆盖、低边界层的大气状态，典型的大尺度层状云环境。C3在温度（z = −0.80）、CAPE（z = −0.67）、总柱水（z = −1.22）、总柱水汽（z = −1.18）、850 hPa相对湿度（z = −1.30）、0℃层高度（z = −1.08）和地表气压（z = −1.37）上均为负值，仅边界层高度（z = +0.66）为正值，表明低能量、干燥、低0℃层的大气状态，不利于强对流发展。C0在各参数上均接近0（z-score在−0.7至+0.4之间），无极端特征，反映环境条件的中等性和过渡性。

图S10（补充图）以效应量（Effect Size = (类别中位数 − 总体中位数) / IQR）展示了各类别相对于总体中位数的绝对偏离。与图5（以类别间均值标准化）互为补充：图S10突出各类别的绝对特征，图5突出类别间的相对差异。两者一致表明：C2的CAPE和温度显著高于总体（效应量~0.37–0.68），C1的湿度和总柱水显著高于总体（效应量~0.10–0.17），C3的总柱水、总柱水汽和850 hPa相对湿度显著低于总体（效应量~−0.15至−0.20）。

环境场分析表明，四类降水对应截然不同的气象环境：C2处于高温、高湿、高CAPE的极端不稳定环境，对应台风和强对流；C1处于高湿度、低CAPE、强锋面强迫的层状云环境；C0处于中等条件、无极端特征的过渡环境；C3处于低能量、干燥、地形制约的浅薄降水环境。环境场与微物理特征、垂直结构和空间分布呈现多维度一致性，从独立数据源（ERA5再分析，非DPR反演量）的角度支撑了聚类结果的物理可信度。

### 3.6 降水强度谱

图6展示了四类降水强度的频率分布、类别平均降水率和分位数分布。在降水强度频率分布（图6a）中，C2在>5 mm h⁻¹和>10 mm h⁻¹区间的频率显著高于其他三类，>20 mm h⁻¹的强降水几乎全部由C2贡献。C0和C1的频率峰值均集中在1–5 mm h⁻¹区间，C3的峰值集中在<1 mm h⁻¹区间。类别平均降水率（图6b）显示，C2（8.8 mm h⁻¹）显著高于C0（3.6 mm h⁻¹）、C1（2.8 mm h⁻¹）和C3（1.6 mm h⁻¹）。分位数分布（图6c）进一步揭示了C2在高分位区间（>0.8分位）降水率急剧上升的特征，而C0、C1和C3在各分位区间均较为平缓，表明C2不仅平均降水率最高，且极端降水事件的发生概率亦显著高于其他类别。这一降水强度谱特征与微物理参数（C2的Dm最大、Nw最小）一致，大粒子低浓度的滴谱结构对应更高的降水效率。

### 3.7 统计显著性检验

为定量验证四类降水的客观性和物理可区分性，对微物理参数和环境参数进行非参数统计检验。所有配对比较（6组配对）均采用Mann-Whitney U检验和Kolmogorov-Smirnov检验，并以Cliff's Delta衡量效应量。所有检验样本量均超过90万，统计功效极高。

表6展示了关键微物理参数配对检验的显著结果。所有配对在所有参数上均通过双检验（p < 0.001），表明四类在统计上完全可区分。效应量分析揭示了核心区分指标：粒径分布（dm_column_max）是区分强对流的最关键指标，C2 vs其他类别差异达到极大效应（|d| = 0.73–0.96）；峰值反射率（zku_max）是区分深厚对流与浅薄降水的核心指标，C2 vs C3效应量达0.953；亮带高度（heightBB）是层状云与其他类别的判别指标，C1 vs C2和C1 vs C3的效应量分别达0.999和1.000。此外，风暴顶高在C1 vs C3（d = 0.601）和C0 vs C2（d = 0.471）上达到大效应，广义截距参数（nw_column_max）在C0 vs C2上达到大效应（d = 0.615）。

**表6 关键微物理参数统计显著性检验结果（节选）**

| 对比组 | 参数 | 中位数A | 中位数B | Mann-Whitney p | KS p | Cliff's Delta | 效应等级 |
|--------|------|---------|---------|----------------|------|---------------|---------|
| C0 vs C1 | heightBB (km) | 0.0 | 4.42 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | −0.606 | 大 |
| C0 vs C2 | zku_max (dBZ) | 30.92 | 38.53 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | −0.569 | 大 |
| C0 vs C2 | dm_column_max (mm) | 1.39 | 2.16 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | −0.734 | 极大 |
| C0 vs C3 | zku_max (dBZ) | 30.92 | 26.29 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.498 | 大 |
| C1 vs C2 | zku_max (dBZ) | 32.51 | 38.53 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | −0.719 | 极大 |
| C1 vs C2 | dm_column_max (mm) | 1.46 | 2.16 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | −0.910 | 极大 |
| C1 vs C2 | heightBB (km) | 4.42 | 0.0 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.999 | 极大 |
| C1 vs C3 | heightStormTop (m) | 7 195 | 4 833 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.601 | 大 |
| C1 vs C3 | zku_max (dBZ) | 32.51 | 26.29 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.668 | 大 |
| C1 vs C3 | heightBB (km) | 4.42 | 0.0 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 1.000 | 极大 |
| C2 vs C3 | zku_max (dBZ) | 38.53 | 26.29 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.953 | 极大 |
| C2 vs C3 | dm_column_max (mm) | 2.16 | 1.25 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.958 | 极大 |
| C2 vs C3 | zku_ns (dBZ) | 36.74 | 24.75 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.939 | 极大 |

表7展示了关键环境参数配对检验结果。环境参数的区分能力总体上弱于微物理参数，但仍有多个参数达到大效应。CAPE在C2 vs其他类别上差异显著（d = −0.23至−0.66），但C0 vs C3的差异仅达中等（d = 0.48），表明CAPE主要区分强对流与非强对流。地表气压在C0 vs其他类别上差异较大（d = 0.48–0.71），反映C0主要分布于低海拔平原区。温度在C2 vs其他类别上差异大（d = −0.53至0.56），表明高温环境是强对流的必要条件。水汽（TCWV）在C2 vs C3上差异大（d = 0.49），表明充足水汽是强对流发展的关键。

**表7 关键环境参数统计显著性检验结果（节选）**

| 对比组 | 参数 | 中位数A | 中位数B | Mann-Whitney p | KS p | Cliff's Delta | 效应等级 |
|--------|------|---------|---------|----------------|------|---------------|---------|
| C0 vs C1 | sp (hPa) | 98 680 | 99 102 | 2.1×10⁻²⁵⁹ | <1×10⁻³⁰⁰ | 0.643 | 大 |
| C0 vs C1 | cape (J kg⁻¹) | 151.6 | 145.5 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.586 | 大 |
| C0 vs C3 | sp (hPa) | 98 680 | 97 875 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.714 | 大 |
| C1 vs C2 | t2m (K) | 295.9 | 299.8 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | −0.525 | 大 |
| C1 vs C2 | cape (J kg⁻¹) | 145.5 | 599.2 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | −0.658 | 大 |
| C2 vs C3 | t2m (K) | 299.8 | 295.3 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.557 | 大 |
| C2 vs C3 | cape (J kg⁻¹) | 599.2 | 104.5 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.566 | 大 |
| C2 vs C3 | tcwv (kg m⁻²) | 57.5 | 46.2 | <1×10⁻³⁰⁰ | <1×10⁻³⁰⁰ | 0.489 | 大 |

综合微物理参数和环境参数的全部90组配对检验（6对×15参数），100%通过双检验（p < 0.001），约65%的参数对比达到大效应（Cliff's Delta |d| > 0.474）。微物理参数的区分能力普遍强于环境参数，其中粒径分布和反射率结构是最核心的分类依据。C2与其他类别的差异最为显著，在几乎所有参数上均达到大效应或极大效应。C0 vs C3的区分能力相对较弱，部分环境参数效应量较小，表明这两个中等/弱类别在环境条件上存在一定重叠，但在微物理结构上仍有显著差异。统计检验从定量角度充分证实了四类降水的客观性和物理可区分性，为基于22维微物理特征的两阶段层次聚类提供了严格的统计支撑。

---

## 4. 讨论

### 4.1 与已有降水分类研究的对比

本研究基于22维GPM DPR微物理特征，将东亚暖季降水客观分为四类：深厚弱对流/层状云混合（C0）、亮带主导的层状云（C1）、强对流（C2）和浅薄降水/弱对流（C3）。这一分类结果与已有研究既有共通之处，亦呈现出区域特异性。

在全球尺度分类研究中，Zhang et al. (2026) 基于GPM DPR数据（2018–2022）和K-means+PCA方法，将全球降水系统分为八类，包括高纬度浅薄、副热带浅薄、中等、深对流、极端深对流、强降水、极端强降水和海洋性极端降水。其"深对流"和"强降水"类别与本研究的C2（强对流）在微物理特征上具有相似性：均表现为高反射率、高风暴顶和大Dm值。然而，本研究将全球分类中的"中等"和"副热带浅薄"进一步整合为C0和C3，将"高纬度浅薄"和"极端"类别因东亚区域气候特征而不显著。东亚季风区的降水以暖季（4–9月）为主，冬季降水类型（如降雪、冻雨）不在本研究范围内，因此四类体系更贴合东亚暖季降水的实际多样性。此外，Zhang et al. (2026) 使用PCA降维后聚类，本研究直接在22维原始特征空间中执行层次聚类，保留了更多垂直结构细节，这是两者在方法论上的根本差异。

Luo et al. (2017) 基于TRMM、CloudSat和CALIPSO近同步观测，利用k-means将有组织深对流、对流抑制厚砧降水和对流抑制系统分为三类。其"有组织深对流"与本研究的C2具有可比性，但Luo et al. (2017) 的多传感器融合方法限制了样本量和时空覆盖，而本研究仅使用GPM DPR单源数据，样本量达960万，空间覆盖整个东亚地区，在统计代表性和区域精细度上具有优势。Aoki et al. (2026) 基于EarthCARE/CPR和GPM/DPR联合观测，将降水分为深层层状、中等对流、高大对流和浅层四种类型，其分类结果与本研究的C1、C0、C2和C3大致对应，但Aoki et al. (2026) 的联合观测限制了时空覆盖，而本研究基于十余年连续DPR观测，在气候学代表性上更为充分。

在基于雨滴大小分布（DSD）的分类研究中，Ryu et al. (2021) 利用GMM将全球大雨划分为大陆性、海洋性深对流和海洋性浅对流三类，Ryu et al. (2025) 进一步将温带大雨分为高风暴高度和低风暴高度两类。本研究的C2在Dm-Nw参数空间上位于大陆性深对流区域（Dm大、Nw低），C3位于海洋性浅对流区域（Dm小、Nw高），C0和C1介于两者之间。这一结果与Bringi et al. (2003) 建立的海洋性-大陆性二分框架一致，但本研究进一步将中间过渡状态细分为C0和C1，揭示了东亚区域降水DSD的连续变异特征。Wen et al. (2026) 基于中国1031个雨滴谱仪和10年GPM DPR数据，证实了降水微物理的连续谱特征，本研究的四分类结果从聚类分析的角度为这一连续谱提供了离散化的客观节点，四类在DSD参数空间上的分布与Wen et al. (2026) 的连续分布一致。

### 4.2 与东亚降水环境场研究的对比

本研究识别的四类降水对应不同的气象环境，与东亚暖季降水环境场研究的结果一致。C2（强对流）对应高温、高湿、高CAPE的极端不稳定环境，CAPE中位数达599 J kg⁻¹，TCWV达57.5 kg m⁻²，这与华南前汛期和盛夏台风降水的高CAPE、高水汽环境特征相符（Yamaji and Takahashi, 2023; Wen et al., 2023）。C1（层状云）对应高湿度、低CAPE、强锋面强迫的环境，RH850达96.4%，CAPE仅145.5 J kg⁻¹，这与江淮梅雨锋和准静止锋的大尺度层状云降水环境一致（Houze, 1997）。C0（深厚弱对流）的CAPE中等（423.8 J kg⁻¹），环境条件介于C1和C2之间，可能对应梅雨锋上的对流嵌入结构或弱对流系统。C3（浅薄降水）对应低CAPE（373.2 J kg⁻¹）、低TCWV（43.4 kg m⁻²）的干燥环境，与青藏高原边缘和西北东部的地形降水环境相符。

本研究的一个关键发现是环境场与微物理特征的多维度一致性：C2的高CAPE与高Dm、低Nw对应，反映了强烈不稳定环境下活跃的冰相过程和碰并增长；C1的高湿度与低CAPE对应，反映了大尺度动力强迫主导的暖雨过程；C3的低CAPE与低Dm、低反射率对应，反映了边界层制约下的浅薄暖雨过程。这种环境-微物理耦合关系与Dolan et al. (2018) 基于全球雨滴谱仪识别的六种DSD模态的环境控制机制一致，但本研究在区域尺度上提供了更精细的空间分布和季节演变信息。

### 4.3 与强降水微物理研究的对比

在强降水微物理方面，本研究的C2（强对流）表现出典型的深对流微物理特征：Dm最大（2.2 mm）、Nw最低（32.6）、反射率最强（zku_max ~38.5 dBZ）、风暴顶最高（8 436 m）。这与Schumacher and Funk (2023) 基于GPM DPR识别的热带和温带深对流特征一致，即深对流以少数量的大雨滴为主，Nw较低而Dm较高。多频准CFAD图显示，C2的DFR随高度增加而增大，表明中高层存在大量大冰相粒子（冰雹、霰），这与深对流中活跃的冰相过程和凇附增长机制一致（Bringi et al., 2003）。

C1（层状云）的亮带结构在Ka波段4–6 km高度表现为反射率增强，DFR在该层有微弱峰值，这与层状云融化层中粒子碰并增长导致的尺寸增大一致（Testud et al., 2001）。C1的Dm（1.5 mm）和Nw（35.3）均接近总体均值，反映了层状云降水以中等尺寸、中等浓度的雨滴为主，与典型的层状云DSD特征一致（Tokay and Short, 1996）。

C3（浅薄降水）的Dm最小（1.3 mm）、Nw中等（35.3）、反射率最低（zku_max ~26.3 dBZ），与暖雨过程中碰并增长有限、以中小雨滴为主的特征一致。C3的DFR接近0甚至略负，表明粒子尺寸小，这与浅对流中以云滴和小雨滴为主的微物理结构相符（Mapes and Houze, 1993）。

C0（深厚弱对流）的微物理特征介于C1和C2之间，Dm（1.5 mm）和Nw（35.7）接近总体均值，但风暴顶高（6 307 m）和降水率（3.6 mm h⁻¹）高于C1而低于C2。C0的亮带出现率约48%，表明其对流与层状混合的过渡性质。这一类别在已有文献中较少被单独识别，可能对应于梅雨锋上的对流嵌入结构或弱对流系统，其微物理特征反映了从层状云到对流的过渡状态。

### 4.4 方法局限性

本研究的方法存在若干局限性。首先，聚类方法为硬划分（k-means），每个样本被强制归入单一类别，这与真实大气中降水微物理的连续谱特征不符（Wen et al., 2026）。GMM等软分类方法可提供概率性归属，但计算复杂度限制了其在大样本卫星数据集上的可扩展性。未来研究可考虑在层次聚类框架中引入模糊聚类或混合模型，以更好地刻画过渡态降水。

其次，当前聚类特征集为22维参数汇总，不包含逐高度廓线数据，因此垂直结构分析基于准CFAD（结构代理）而非严格的高度剖面CFAD。严格CFAD需要原始逐层雷达反射率数据，本研究基于现有特征集构建的准CFAD虽能揭示代表性垂直结构特征，但在高度分辨率上存在局限。若上游匹配后的原始垂直剖面数据可用，未来可重建严格CFAD以进一步验证分类结果。

第三，本研究仅覆盖暖季（4–9月），未包含冬季降水类型。东亚冬季降水以锋面雪、冻雨和混合型降水为主，其微物理特征与暖季降水显著不同。全年分类可能识别出更多类别，但冬季DPR观测受限于降雪探测能力，需结合其他传感器（如CloudSat CPR）进行综合分类。

第四，ERA5环境场与GPM DPR的时空分辨率存在差异（ERA5为0.25°×1 h，DPR为~5 km×瞬时），空间最近邻匹配可能引入系统偏差。尽管本研究采用同期最近邻匹配，但ERA5的网格尺度平均效应可能平滑了DPR观测的小尺度环境特征。未来研究可考虑更高分辨率的环境场数据（如ERA5-Land或区域模式输出）进行验证。

第五，聚类数K = 4的确定基于多指标综合评估，但不同指标之间存在一定分歧（如DB指数推荐K = 4，轮廓系数在K = 4和K = 6之间波动）。尽管K = 4在综合指标上最优，但其他K值可能识别出具有物理意义的过渡类别。敏感性分析表明，K = 5时DB指数上升且最小类样本量降至14 580，统计可靠性下降，因此K = 4为当前数据条件下的最优选择。

### 4.5 应用价值与展望

本研究建立的东亚暖季降水四分类体系及其微物理气候学特征，在多个领域具有应用价值。在数值模式验证方面，分类结果可作为评估区域模式（如WRF、GRAPES）微物理参数化方案的观测约束。不同类别对应的Dm-Nw参数范围可直接与模式输出对比，检验模式对东亚降水微物理的模拟能力。在定量降水估计（QPE）改进方面，分类结果可指导双频雷达反演算法在不同降水类型中的参数优化，提高QPE精度。在灾害预警方面，C2（强对流）的空间分布和强度谱特征可为台风、暴雨和强对流灾害的风险评估提供依据。

未来研究可从以下方向深化本工作：（1）将分类扩展至全年，识别冬季降水类型，构建完整的东亚降水分类体系；（2）结合多源卫星数据（如CloudSat CPR、EarthCARE）和地面雷达网络，验证并扩展分类结果；（3）利用分类结果驱动数值模式微物理参数化方案的改进，评估分类对降水预报精度的提升；（4）分析不同年份间各类别频率的长期变化趋势，探讨气候变化背景下东亚降水类型的演变规律。

---

## 5. 结论

本研究基于2014–2024年暖季（4–9月）GPM DPR双频雷达观测数据，采用两阶段层次k-means聚类方法，对东亚地区960万有效降水样本进行客观分类。通过22维微物理特征参数和ERA5再分析环境场的综合分析，识别出四类降水系统：深厚弱对流/层状云混合（C0，24.0%）、亮带主导的层状云（C1，29.3%）、强对流（C2，11.8%）和浅薄降水/弱对流（C3，35.1%）。

主要结论如下：

（1）四类降水在微物理特征、垂直结构、空间分布和环境条件上呈现系统差异。C2具有最强的降水率（8.8 mm h⁻¹）、最高的风暴顶（8 436 m）、最大的粒子尺寸（Dm ~2.2 mm）和最低的粒子浓度（Nw ~32.6），对应高温、高湿、高CAPE的极端不稳定环境，典型天气系统为台风和强对流。C1具有最高的亮带出现率（~100%）、最高的亮带高度（~4.2 km）和最高的湿度（RH850 ~96.4%），但CAPE最低（367.6 J kg⁻¹），对应大尺度锋面系统主导的层状云降水。C3具有最弱的降水率（1.6 mm h⁻¹）、最低的风暴顶（5 054 m）和最小的粒子尺寸（Dm ~1.3 mm），对应低能量、干燥、地形制约的环境。C0介于C1和C2之间，无极端特征，呈现对流与层状混合的过渡性质。

（2）分类结果具有统计可靠性和物理一致性。全部90组配对检验（6对×15参数）均通过Mann-Whitney U检验和Kolmogorov-Smirnov检验（p < 0.001），约65%的参数对比达到大效应（Cliff's Delta |d| > 0.474）。微物理参数的区分能力普遍强于环境参数，其中粒径分布（Dm）和峰值反射率（zku_max）是最核心的分类依据。环境场分析表明，四类降水对应截然不同的气象环境，从独立数据源（ERA5再分析）的角度支撑了聚类结果的物理可信度。多频准CFAD图揭示了四类在垂直结构上的差异：C2的DFR随高度增加而增大，表明中高层大冰相粒子（冰雹、霰）的存在；C1在亮带高度呈现反射率增强和DFR微弱峰值，对应层状云融化层结构；C3的DFR接近0，表明以小雨滴为主的暖雨过程。空间分布显示，C2集中于华南沿海和海洋区域，C1集中于华北和江淮流域，C3集中于西南和高原边缘，主导类型的地理分异与东亚季风降水的气候学格局一致。降水强度谱分析表明，C2在>5 mm h⁻¹和>10 mm h⁻¹区间的频率显著高于其他三类，>20 mm h⁻¹的强降水几乎全部由C2贡献。

（3）本研究的主要创新在于：建立了基于22维微物理特征和两阶段层次聚类的东亚暖季降水客观分类框架，识别了四类在微物理特征和环境条件上具有系统差异的降水类型；通过ERA5环境场独立验证，证实了分类结果的多维度物理一致性，而非单纯的统计划分；揭示了不同类型对应的环境-结构耦合路径，为区域暖季降水识别和数值模式微物理参数化方案提供了观测约束。

本研究的分类体系可应用于数值模式验证、定量降水估计改进和灾害天气风险评估。未来研究将进一步扩展至全年降水类型分类，结合多源卫星和地面雷达数据验证分类结果，并分析气候变化背景下东亚降水类型频率的长期演变规律。

---

## 数据可用性声明

本研究使用的GPM DPR数据可通过NASA戈达德航天中心GES DISC数据存档获取（https://disc.gsfc.nasa.gov/）。ERA5再分析数据可通过ECMWF哥白尼气候变化服务（C3S）气候数据存储库获取（https://cds.climate.copernicus.eu/）。研究过程中生成的分类结果数据集和分析代码可根据合理请求向通讯作者获取。

---

## 作者贡献

[待补充：建议按CRediT分类标准列出各作者贡献，如：Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data Curation, Writing – Original Draft, Writing – Review & Editing, Visualization, Supervision, Project administration, Funding acquisition]

---

## 利益冲突声明

作者声明不存在可能影响本研究工作的利益冲突。

---

## 资助声明

[待补充：本研究受……资助]

---

## 参考文献

Aoki, S., Kubota, T., Turk, F.J., 2026. Exploring vertical motions in convective and stratiform precipitation using spaceborne radar observations: insights from EarthCARE and GPM coincidence dataset. Atmos. Meas. Tech. 19, 79–100. https://doi.org/10.5194/amt-19-79-2026

Awaka, J., Iguchi, T., Seto, S., Meneghini, R., Yoshida, N., Le, M., Durden, S., 2021. Development of precipitation type classification algorithms for a full scan mode of GPM Dual-frequency Precipitation Radar. J. Meteorol. Soc. Japan 99, 1253–1270. https://doi.org/10.2151/jmsj.2021-061

Bringi, V.N., Chandrasekar, V., Hubbert, J., Gorgucci, E., Randeu, W.L., Schoenhuber, M., 2003. Raindrop size distribution in different climatic regimes from disdrometer and dual-polarized radar analysis. J. Atmos. Sci. 60, 354–365. https://doi.org/10.1175/1520-0469(2003)060<0354:RSDIDC>2.0.CO;2

Calinski, T., Harabasz, J., 1974. A dendrite method for cluster analysis. Commun. Statist. 3(1), 1–27. https://doi.org/10.1080/03610927408827101

Churchill, D.D., Houze, R.A., 1984. Development and structure of winter monsoon cloud clusters on 10 December 1978. J. Atmos. Sci. 41, 933–960. https://doi.org/10.1175/1520-0469(1984)041<0933:DOACSP>2.0.CO;2

Cliff, N., 1993. Dominance statistics: Ordinal analyses to answer ordinal questions. Psychol. Bull. 114(3), 494–509. https://doi.org/10.1037/0033-2909.114.3.494

Davies, D.L., Bouldin, D.W., 1979. A cluster separation measure. IEEE Trans. Pattern Anal. Mach. Intell. PAMI-1, 224–227. https://doi.org/10.1109/TPAMI.1979.4766909

Dixon, M., Romatschke, U., 2022. Three-dimensional convective-stratiform echo-type classification and convectivity retrieval from radar reflectivity. J. Atmos. Oceanic Technol. 39, 1685–1704. https://doi.org/10.1175/JTECH-D-22-0018.1

Dolan, B., Fuchs, B., Rutledge, S.A., Barnes, E.A., Thompson, E.J., 2018. Primary modes of global drop size distributions. J. Atmos. Sci. 75, 1453–1476. https://doi.org/10.1175/JAS-D-17-0248.1

Schumacher, C., Funk, A., 2023. Assessing convective-stratiform precipitation regimes in the tropics and extratropics with the GPM satellite radar. Geophys. Res. Lett. 50, e2023GL102786. https://doi.org/10.1029/2023GL102786

Gatlin, P., Stough, S., 2023. The microphysics and kinematics of GPM's satellite radar profiles. AMS 40th Conference on Radar Meteorology.

Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., De Chiara, G., Dahlgren, P., Dee, D., Diamantakis, M., Dragani, R., Flemming, J., Forbes, R., Fuentes, M., Geer, A., Haimberger, L., Healy, S., Hogan, R.J., Hólm, E., Janisková, M., Keeley, S., Laloyaux, P., Lopez, P., Lupu, C., Radnoti, G., de Rosnay, P., Rozum, I., Vamborg, F., Villaume, S., Thépaut, J.N., 2020. The ERA5 global reanalysis. Q. J. Roy. Meteorol. Soc. 146, 1999–2049. https://doi.org/10.1002/qj.3803

Hou, A.Y., Kakar, R.K., Neeck, S., Azarbarzin, A.A., Kummerow, C.D., Kojima, M., Oki, R., Nakamura, K., Iguchi, T., 2014. The Global Precipitation Measurement Mission. Bull. Am. Meteorol. Soc. 95, 701–722. https://doi.org/10.1175/BAMS-D-13-00164.1

Houze, R.A., 1989. Meteorological paradoxes. Bull. Am. Meteorol. Soc. 70, 1171–1175. https://doi.org/10.1175/1520-0477(1989)070<1171:MP>2.0.CO;2

Houze, R.A., 1997. Stratiform precipitation in regions of convection: A meteorological paradox? Bull. Am. Meteorol. Soc. 78, 2179–2196. https://doi.org/10.1175/1520-0477(1997)078<2179:SPIROC>2.0.CO;2

Iguchi, T., Seto, S., Meneghini, R., Yoshida, N., Awaka, J., Kubota, T., Masaki, T., Takahashi, N., 2012. An overview of the precipitation retrieval algorithm for the dual-frequency precipitation radar (DPR) on the Global Precipitation Measurement (GPM) mission's core satellite. Proc. SPIE 8528, 85281C. https://doi.org/10.1117/12.977352

Kolmogorov, A.N., 1933. Sulla determinazione empirica di una legge di distribuzione. Giorn. Ist. Ital. Attuari 4, 83–91.

Liu, C., Zipser, E.J., 2015. The global distribution of largest, deepest, and most intense precipitation systems. Geophys. Res. Lett. 42, 3591–3595. https://doi.org/10.1002/2015GL063776

Lloyd, S., 1982. Least squares quantization in PCM. IEEE Trans. Inf. Theory 28, 129–137. https://doi.org/10.1109/TIT.1982.1056489

Luo, Y., Li, Z., Zhang, J., Liu, S., Wang, T., Matsui, T., 2017. Cloud and precipitation properties and their structures observed by the A-Train satellite. Atmos. Res. 196, 1–16. https://doi.org/10.1016/j.atmosres.2017.06.007

Mann, H.B., Whitney, D.R., 1947. On a test of whether one of two random variables is stochastically larger than the other. Ann. Math. Statist. 18, 50–60.

Mapes, B.E., Houze, R.A., 1993. Cloud clusters and superclusters over the oceanic warm pool. Mon. Weather Rev. 121, 1398–1415. https://doi.org/10.1175/1520-0493(1993)121<1398:CCASOT>2.0.CO;2

Meneghini, R., Kim, H., Liao, L., Jones, J.A., Kwiatkowski, J.M., 2015. An initial assessment of the surface reference technique applied to data from the dual-frequency precipitation radar (DPR) on the GPM satellite. J. Atmos. Oceanic Technol. 32, 2281–2296. https://doi.org/10.1175/JTECH-D-15-0044.1

Murtagh, F., Contreras, P., 2012. Algorithms for hierarchical clustering: an overview. Wiley Interdiscip. Rev. Data Min. Knowl. Discov. 2, 86–97. https://doi.org/10.1002/widm.53

Pendergrass, A.G., Hartmann, D.L., 2014. Changes in the distribution of rain frequency and intensity in response to global warming. J. Climate 27, 8372–8383. https://doi.org/10.1175/JCLI-D-14-00183.1

Rasmussen, K.L., Houze, R.A., 2016. Convective initiation near the Andes in subtropical South America. Mon. Weather Rev. 144, 2351–2374. https://doi.org/10.1175/MWR-D-15-0058.1

Rousseeuw, P.J., 1987. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. J. Comput. Appl. Math. 20, 53–65. https://doi.org/10.1016/0377-0427(87)90125-7

Ryu, J., Song, H.-J., Sohn, B.-J., Liu, C., 2021. Global distribution of three types of drop size distribution representing heavy rainfall from GPM/DPR measurements. Geophys. Res. Lett. 48, e2020GL090871. https://doi.org/10.1029/2020GL090871

Ryu, J., Lee, J., You, Y., 2025. A distinct type of heavy rainfall with large raindrops over extratropical regions revealed by 10 years of GPM spaceborne radar measurements. Int. J. Appl. Earth Obs. Geoinf. 144, 104879. https://doi.org/10.1016/j.jag.2025.104879

Seto, S., Iguchi, T., Meneghini, R., Awaka, J., Kubota, T., Masaki, T., Takahashi, N., 2021. The precipitation rate retrieval algorithms for the GPM Dual-frequency Precipitation Radar. J. Meteorol. Soc. Japan 99, 205–237. https://doi.org/10.2151/jmsj.2021-011

Shi, R., Lu, C., Xu, W., Luo, Y., 2025. A global view on microphysical discriminations between heavier and lighter convective rainfall. Commun. Earth Environ. 6, 511. https://doi.org/10.1038/s43247-025-02473-0

Skofronick-Jackson, G., Petersen, W.A., Berg, W., Kidd, C., Stocker, E.F., Kirschbaum, D.B., Kakar, R., Braun, S.A., Huffman, G.J., Iguchi, T., Kirstetter, P.E., Kummerow, C., Meneghini, R., Oki, R., Olson, W.S., Takayabu, Y.N., Furukawa, K., Wilheit, T., 2017. The Global Precipitation Measurement (GPM) Mission for Science and Society. Bull. Am. Meteorol. Soc. 98, 1679–1695. https://doi.org/10.1175/BAMS-D-15-00306.1

Smirnov, N., 1948. Table for estimating the goodness of fit of empirical distributions. Ann. Math. Statist. 19, 279–281. https://doi.org/10.1214/aoms/1177730256

Steiner, M., Houze, R.A., Yuter, S.E., 1995. Climatological characterization of three-dimensional storm structure from operational radar and rain gauge data. J. Appl. Meteorol. 34, 1978–2007. https://doi.org/10.1175/1520-0450(1995)034<1978:CCOTDS>2.0.CO;2

Testud, J., Oury, S., Black, R.A., Amayenc, P., Dou, X.K., 2001. The concept of 'normalized' distribution to describe raindrop spectra: A tool for cloud physics and cloud remote sensing. J. Appl. Meteorol. 40, 1118–1140. https://doi.org/10.1175/1520-0450(2001)040<1118:TCONDT>2.0.CO;2

Thurai, M., Dolan, B., Schultz, C., Petersen, W.A., 2017a. Toward completing the raindrop size spectrum: Case studies involving 2D-video disdrometer, droplet spectrometer, and polarimetric radar measurements. J. Appl. Meteor. Climatol. 56, 877–896. https://doi.org/10.1175/JAMC-D-16-0304.1

Thurai, M., Dolan, B., Schultz, C., Petersen, W.A., 2017b. Initial results of a new composite-weighted algorithm for dual-polarized X-band rainfall estimation. J. Hydrometeorol. 18, 1081–1100. https://doi.org/10.1175/JHM-D-16-0196.1

Tokay, A., Short, D.A., 1996. Evidence from tropical raindrop spectra of the origin of rain from stratiform versus convective clouds. J. Appl. Meteorol. 35, 355–371. https://doi.org/10.1175/1520-0450(1996)035<0355:EFTRSO>2.0.CO;2

Tokay, A., Bashor, P.G., Habib, E., Kasparis, T., 2008. Raindrop size distribution measurements in tropical cyclones. Mon. Weather Rev. 136, 1669–1685. https://doi.org/10.1175/2007MWR2122.1

Ulbrich, C.W., Atlas, D., 2007. Microphysics of raindrop size spectra: Tropical continental and maritime storms. J. Appl. Meteorol. Climatol. 46, 1777–1791. https://doi.org/10.1175/2007JAMC1649.1

Varga-Balogh, A., Leelőssy, Á., Mészáros, R., 2025. Characterization of mesoscale precipitation systems in Central Europe. Atmos. Res. 309, 107564. https://doi.org/10.1016/j.atmosres.2024.107564

Ward, J.H., Jr., 1963. Hierarchical grouping to optimize an objective function. J. Am. Statist. Assoc. 58(301), 236–244. https://doi.org/10.1080/01621459.1963.10500845

Wen, L., Zhao, K., Zhang, G., Xue, M., Zhou, B., Liu, S., Chen, X., 2016. Statistical characteristics of raindrop size distributions observed in East China during the Asian summer monsoon season using 2D-video disdrometer and micro-rain radar data. J. Geophys. Res. Atmos. 121, 2265–2282. https://doi.org/10.1002/2015JD024160

Wen, L., Zhao, K., Chen, G., Wang, M., Zhang, G., 2018. Drop size distribution characteristics of seven typhoons in China. J. Geophys. Res. Atmos. 123, 6529–6548. https://doi.org/10.1029/2017JD027950

Wen, L., Chen, G., Yang, C., Zhang, H., Fu, Z., 2023. Seasonal variations in precipitation microphysics over East China based on GPM DPR observations. Atmos. Res. 293, 106933. https://doi.org/10.1016/j.atmosres.2023.106933

Wen, L., Chen, G., Wang, S., Nie, J., Zhao, K., 2026. Observed universal continuum morphology of raindrops reveals a concise diagram of heavy precipitation microphysics. Proc. Natl. Acad. Sci. U.S.A. 123, e2525260123. https://doi.org/10.1073/pnas.2525260123

Yamaji, M., Takahashi, H.G., 2023. Seasonal differences of precipitation and microphysical characteristics over the Asian monsoon region using spaceborne dual-frequency precipitation radar. J. Atmos. Sci. 80, 2115–2128. https://doi.org/10.1175/JAS-D-22-0198.1

Zhang, Y., Zhang, X., Ni, X., 2026. Microphysical properties of various precipitation systems worldwide classified via objective methods based on dual-frequency precipitation radar observations. Atmos. Chem. Phys. 26, 4727–4747. https://doi.org/10.5194/acp-26-4727-2026
