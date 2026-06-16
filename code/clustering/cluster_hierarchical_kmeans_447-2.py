"""
两阶段层次-K-means聚类分析 (447维版本)
对GPM-ERA5最近邻匹配结果进行过度聚类+层次聚类

方法概述：
1. 第一阶段：使用K-means进行过度聚类（设置较高的k值），得到k个质心
2. 第二阶段：对质心集合使用Ward层次聚类方法进行二次聚类
3. 使用树状图（dendrogram）和Davies-Bouldin分数来确定最佳类别数
4. 将最终聚类标签映射回原始数据

输入：
- matched_nearest_*.npz（如 matched_nearest_20xx_0x.npz，447维匹配结果）
- 默认目录见 DEFAULT_MATCHED_NEAREST_INPUT_DIR，可用 --input-dir 覆盖

输出：
- 默认写入 DEFAULT_HIERARCHICAL_CLUSTERING_OUTPUT_DIR 下子目录 cluster_results_*（可用 --output-dir 覆盖）
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

# 限制 OpenBLAS 线程数，避免内存和线程冲突
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 项目根目录 (run/hierarchical_clustering/的上两级)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 工程数据根目录：匹配输入与聚类/敏感性输出默认均在此之下（可用 CLI 覆盖）
GPM_CLUSTER_PROJECT_ROOT = Path(r"E:\julei\GPM_ERA5_Cluster_Project")
_RESULTS_ROOT = GPM_CLUSTER_PROJECT_ROOT / "results"
DEFAULT_MATCHED_NEAREST_INPUT_DIR = str(_RESULTS_ROOT / "matched_nearest")
DEFAULT_HIERARCHICAL_CLUSTERING_OUTPUT_DIR = str(_RESULTS_ROOT / "hierarchical_clustering")

# 配置文件路径 (GPM-only配置，不含ERA5)
CLUSTER_FEATURE_CONFIG_PATH = PROJECT_ROOT / "config" / "cluster_gpm_features_only.yaml"
OUTPUT_ROOT = Path(DEFAULT_HIERARCHICAL_CLUSTERING_OUTPUT_DIR)

# 优化后的特征集(27维，剔除高NaN列)
# 基于2014-2017年数据质量检查结果：
# - 删除NaN>50%的列: 5,6,8,9,19,20,30,31,34,35 (Ka波段+预留位+雨滴谱2.5km)
# - 保留列NaN比例均<33%，其中核心特征NaN<8%
DEFAULT_FEATURE_INDICES = [0, 1, 2, 3, 4, 7, 10, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 36, 37]

# 双频核心区特征集(31维) - 敏感性分析专用
# 策略: 保留Ka相关特征(4,5,6,7,8,9)，但删除这些列含NaN的样本
# 样本损失: ~47%，剩余~53%样本(约220万)，足够聚类分析
# 适用场景: 专门分析双频数据有效的降水像元
DUAL_FREQ_CORE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 36, 37]
# 注意: 使用此配置时需配合 nan_filter='dual_freq' 删除Ka列含NaN的样本

# 高NaN列索引（用于样本筛选）
HIGH_NAN_FEATURE_INDICES = [5, 6, 8, 9, 19, 20, 30, 31, 34, 35]  # NaN>45%的列
DUAL_FREQ_NAN_INDICES = [5, 6, 8, 9]  # Ka波段相关列，用于双频核心区筛选


def load_cluster_feature_indices(preset_name: str) -> List[int]:
    """从 config/cluster_feature_config_447.yaml 读取指定预设的特征索引列表。"""
    if not CLUSTER_FEATURE_CONFIG_PATH.is_file():
        print(f"警告: 配置文件不存在 {CLUSTER_FEATURE_CONFIG_PATH}，使用默认447维")
        return DEFAULT_FEATURE_INDICES
    
    try:
        from yaml import safe_load
        with open(CLUSTER_FEATURE_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = safe_load(f)
    except Exception as e:
        print(f"读取配置失败: {e}，使用默认447维")
        return DEFAULT_FEATURE_INDICES
    
    presets = cfg.get("feature_presets") or {}
    if preset_name not in presets:
        available = list(presets.keys())
        print(f"未知预设: {preset_name}，可用: {available}")
        print("使用默认447维")
        return DEFAULT_FEATURE_INDICES
    
    indices = presets[preset_name].get("indices")
    if not indices:
        print(f"预设 {preset_name} 未定义indices，使用默认447维")
        return DEFAULT_FEATURE_INDICES
    
    print(f"使用特征预设: {preset_name}，维度: {len(indices)}")
    return [int(i) for i in indices]


def find_matched_nearest_files(input_dir: str) -> List[str]:
    """查找 matched_nearest_YYYY_MM.npz 文件"""
    if not os.path.isabs(input_dir):
        input_dir = os.path.join(PROJECT_ROOT, input_dir)
    
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"输入目录不存在: {input_dir}")
    
    files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.startswith("matched_nearest_") and f.endswith(".npz")
    ]
    files.sort()
    
    if not files:
        raise FileNotFoundError(f"在 {input_dir} 未找到 matched_nearest_*.npz 文件")
    
    print(f"找到 {len(files)} 个匹配结果文件")
    return files


def load_matched_nearest_file(npz_path: str, feature_indices: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    读取 matched_nearest 文件，提取指定特征
    
    返回: (features, lat, lon, metadata)
    """
    data = np.load(npz_path, allow_pickle=True)
    data_dict = dict(data)
    
    # 获取所有block
    block_keys = sorted([k for k in data_dict.keys() 
                        if k.startswith("block_") 
                        and not k.startswith("lat_") 
                        and not k.startswith("lon_")
                        and not k.startswith("timestamp_")
                        and not k.startswith("meta_")])
    
    if not block_keys:
        raise ValueError(f"{npz_path} 中未找到 block_* 数据")
    
    all_features = []
    all_lat = []
    all_lon = []
    
    for block_key in block_keys:
        block_data = data_dict[block_key]  # (N, 447)

        # 检查维度
        if block_data.ndim != 2:
            raise ValueError(f"{block_key} 维度异常: {block_data.shape}")

        if block_data.shape[1] < 447:
            print(f"警告: {block_key} 特征维度 {block_data.shape[1]} < 447，可能数据不完整")

        # 提取指定特征
        max_idx = max(feature_indices)
        if block_data.shape[1] <= max_idx:
            raise ValueError(f"{block_key} 维度 {block_data.shape[1]} 小于所需最大索引 {max_idx}")

        selected_features = block_data[:, feature_indices]
        all_features.append(selected_features)

        # 读取经纬度
        block_id = block_key.split("_")[1]
        lat_key = f"lat_block_{block_id}"
        lon_key = f"lon_block_{block_id}"

        if lat_key in data_dict:
            lat_raw = data_dict[lat_key]
            if len(lat_raw) == block_data.shape[0]:
                all_lat.append(lat_raw)
            else:
                all_lat.append(np.full(block_data.shape[0], np.nan))
        else:
            all_lat.append(np.full(block_data.shape[0], np.nan))

        if lon_key in data_dict:
            lon_raw = data_dict[lon_key]
            if len(lon_raw) == block_data.shape[0]:
                all_lon.append(lon_raw)
            else:
                all_lon.append(np.full(block_data.shape[0], np.nan))
        else:
            all_lon.append(np.full(block_data.shape[0], np.nan))

    
    # 合并所有block
    features = np.concatenate(all_features, axis=0)
    lat = np.concatenate(all_lat, axis=0)
    lon = np.concatenate(all_lon, axis=0)
    
    # 元数据
    metadata = {
        'year': int(data_dict.get('meta_year', 0)),
        'month': int(data_dict.get('meta_month', 0)),
        'n_blocks': len(block_keys),
        'original_file': npz_path
    }
    
    print(f"  加载 {metadata['year']}-{metadata['month']:02d}: {features.shape[0]} 样本, {features.shape[1]} 维")
    
    return features, lat, lon, metadata


def build_global_matrix(file_paths: List[str], feature_indices: List[int], 
                        max_samples: Optional[int] = None,
                        nan_filter: Optional[str] = None,
                        original_dim: int = 447,
                        random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """
    构建全局特征矩阵（智能加载优化版）
    
    优化点:
    1. 无需NaN过滤时，只加载需要的特征列（节省90%+内存）
    2. 需要NaN过滤时，先加载判断列过滤，再加载目标列
    3. 可复现的随机采样（带随机种子）
    
    Args:
        file_paths: 数据文件路径列表
        feature_indices: 使用的特征索引
        max_samples: 最大样本数（可选，用于子采样）
        nan_filter: NaN过滤策略（可选）
            - None: 不过滤，保留所有样本（后续填充）
            - 'dual_freq': 删除Ka波段列(5,6,8,9)含NaN的样本（双频核心区）
            - 'strict': 删除任何特征列含NaN的样本（最严格）
        original_dim: 原始特征维度（用于索引映射）
        random_state: 随机种子（用于可复现的子采样）
    """
    all_features = []
    all_lat = []
    all_lon = []
    all_metadata = []
    
    total_loaded = 0
    total_filtered = 0
    
    for file_path in file_paths:
        try:
            if nan_filter is None:
                # =====================================================
                # 优化路径1: 无需NaN过滤，直接只加载需要的列
                # =====================================================
                # 节省内存: 从 447维 → len(feature_indices)维
                features, lat, lon, metadata = load_matched_nearest_file(
                    file_path, feature_indices
                )
                
                # 统计NaN情况（仅用于信息展示）
                nan_ratio = np.isnan(features).sum() / features.size
                if nan_ratio > 0.01:  # 超过1% NaN时提示
                    print(f"    警告: NaN比例 {nan_ratio*100:.1f}%，将在后续填充")
                
                total_loaded += features.shape[0]
                
            elif nan_filter == 'dual_freq':
                # =====================================================
                # 优化路径2: 双频核心区过滤
                # 先加载Ka判断列，过滤后再加载目标列
                # =====================================================
                # 步骤1: 只加载Ka波段列(5,6,8,9)进行判断
                check_indices = list(set(DUAL_FREQ_NAN_INDICES) & set(range(original_dim)))
                check_features, lat, lon, metadata = load_matched_nearest_file(
                    file_path, check_indices
                )
                
                # 步骤2: 判断哪些样本需要保留
                ka_nan_mask = np.isnan(check_features).any(axis=1)
                valid_mask = ~ka_nan_mask
                n_deleted = np.sum(ka_nan_mask)
                
                if n_deleted > 0:
                    print(f"    删除Ka-NaN样本: {n_deleted} ({n_deleted/len(check_features)*100:.1f}%)")
                
                # 步骤3: 重新加载完整数据（仅保留的样本）
                # 为减少内存峰值，分批次处理
                if valid_mask.all():
                    # 全部有效，直接加载目标列
                    features, lat, lon, metadata = load_matched_nearest_file(
                        file_path, feature_indices
                    )
                else:
                    # 部分有效，需要过滤
                    # 先加载全部目标列，再过滤
                    full_features, lat, lon, metadata = load_matched_nearest_file(
                        file_path, feature_indices
                    )
                    features = full_features[valid_mask]
                    lat = lat[valid_mask]
                    lon = lon[valid_mask]
                    del full_features  # 释放内存
                
                total_loaded += features.shape[0]
                total_filtered += n_deleted
                
            elif nan_filter == 'strict':
                # =====================================================
                # 优化路径3: 严格模式
                # 如果需要判断的列在feature_indices中，复用；否则额外加载
                # =====================================================
                # 先加载目标列
                features, lat, lon, metadata = load_matched_nearest_file(
                    file_path, feature_indices
                )
                
                # 判断是否有任何NaN
                any_nan_mask = np.isnan(features).any(axis=1)
                n_deleted = np.sum(any_nan_mask)
                
                if n_deleted > 0:
                    print(f"    删除含NaN样本: {n_deleted} ({n_deleted/len(features)*100:.1f}%)")
                    features = features[~any_nan_mask]
                    lat = lat[~any_nan_mask]
                    lon = lon[~any_nan_mask]
                
                total_loaded += features.shape[0]
                total_filtered += n_deleted
                
            else:
                raise ValueError(f"未知的nan_filter策略: {nan_filter}")
            
            all_features.append(features)
            all_lat.append(lat)
            all_lon.append(lon)
            all_metadata.append(metadata)
            
        except Exception as e:
            print(f"警告: 加载 {file_path} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_features:
        raise ValueError("没有成功加载任何文件")
    
    # 合并所有文件的数据
    X = np.concatenate(all_features, axis=0)
    lat = np.concatenate(all_lat, axis=0)
    lon = np.concatenate(all_lon, axis=0)
    
    print(f"\n全局矩阵: {X.shape}")
    print(f"  原始加载样本: {total_loaded + total_filtered:,}")
    print(f"  有效样本: {X.shape[0]:,}")
    if nan_filter:
        print(f"  过滤策略: {nan_filter} (删除 {total_filtered:,} 样本)")
    
    # 子采样（如果指定）- 使用可复现的随机采样
    if max_samples and X.shape[0] > max_samples:
        print(f"\n可复现子采样: {X.shape[0]:,} → {max_samples:,} (随机种子={random_state})")
        rng = np.random.RandomState(random_state)
        indices = rng.choice(X.shape[0], max_samples, replace=False)
        X = X[indices]
        lat = lat[indices]
        lon = lon[indices]
    
    return X, lat, lon, all_metadata


def clean_and_normalize(X: np.ndarray, mode: str = "minmax") -> Tuple[np.ndarray, object]:
    """数据清洗和归一化"""
    # 处理NaN和Inf
    X = np.where(np.isinf(X), np.nan, X)
    
    # 对每列填充NaN（使用中位数）
    for i in range(X.shape[1]):
        col = X[:, i]
        nan_mask = np.isnan(col)
        if np.any(nan_mask):
            median_val = np.nanmedian(col)
            if np.isnan(median_val):
                median_val = 0
            X[nan_mask, i] = median_val
    
    # 归一化
    if mode == "minmax":
        scaler = MinMaxScaler()
    elif mode == "robust":
        scaler = RobustScaler()
    else:
        return X, None
    
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def stage1_over_clustering(X: np.ndarray, k_over: int = 100, random_state: int = 42) -> Tuple[np.ndarray, KMeans]:
    """第一阶段：过度聚类"""
    print(f"\n第一阶段: K-means 过度聚类 (k={k_over})")
    
    kmeans = KMeans(
        n_clusters=k_over,
        random_state=random_state,
        n_init=10,
        max_iter=300,
        verbose=0
    )
    
    labels = kmeans.fit_predict(X)
    centroids = kmeans.cluster_centers_
    
    print(f"  质心矩阵: {centroids.shape}")
    print(f"  样本分布: min={np.min(np.bincount(labels))}, max={np.max(np.bincount(labels))}")
    
    return centroids, kmeans


def stage2_hierarchical_clustering(centroids: np.ndarray, n_clusters_range: List[int]) -> Dict:
    """第二阶段：层次聚类并评估"""
    print(f"\n第二阶段: 层次聚类")
    print(f"  评估类别数: {n_clusters_range}")
    
    # 计算距离矩阵
    Z = linkage(centroids, method='ward')
    
    results = {}
    for n_clusters in n_clusters_range:
        # 使用fcluster获取标签
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 转为0-based
        
        # 计算评估指标
        db_score = davies_bouldin_score(centroids, labels)
        sil_score = silhouette_score(centroids, labels)
        
        results[n_clusters] = {
            'labels': labels,
            'db_score': db_score,
            'sil_score': sil_score
        }
        
        print(f"  K={n_clusters}: DB={db_score:.4f}, Sil={sil_score:.4f}")
    
    return results, Z


def plot_dendrogram(Z: np.ndarray, output_path: str, n_clusters_range: List[int]):
    """绘制树状图"""
    plt.figure(figsize=(16, 8))
    dendrogram(Z, truncate_mode='level', p=30, show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram (Stage 2)')
    plt.xlabel('Cluster Size')
    plt.ylabel('Ward Distance')
    
    # 标记评估的类别数
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_clusters_range)))
    for i, k in enumerate(n_clusters_range):
        plt.axhline(y=Z[-k+1, 2] if k > 1 else Z[-1, 2], 
                   color=colors[i], linestyle='--', alpha=0.5,
                   label=f'K={k}')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"树状图保存: {output_path}")


def plot_evaluation_curves(results: Dict, output_path: str):
    """绘制评估曲线"""
    n_clusters = sorted(results.keys())
    db_scores = [results[k]['db_score'] for k in n_clusters]
    sil_scores = [results[k]['sil_score'] for k in n_clusters]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # DB指数（越小越好）
    ax1.plot(n_clusters, db_scores, 'bo-')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Davies-Bouldin Index')
    ax1.set_title('DB Index (lower is better)')
    ax1.grid(True)
    best_k_db = n_clusters[np.argmin(db_scores)]
    ax1.axvline(x=best_k_db, color='r', linestyle='--', label=f'Best K={best_k_db}')
    ax1.legend()
    
    # 轮廓系数（越大越好）
    ax2.plot(n_clusters, sil_scores, 'go-')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score (higher is better)')
    ax2.grid(True)
    best_k_sil = n_clusters[np.argmax(sil_scores)]
    ax2.axvline(x=best_k_sil, color='r', linestyle='--', label=f'Best K={best_k_sil}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"评估曲线保存: {output_path}")


def save_results(X: np.ndarray, labels: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                metadata_list: List[dict], output_dir: str, n_clusters: int):
    """保存聚类结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 全局结果
    result_file = os.path.join(output_dir, f"cluster_results_K{n_clusters}.npz")
    np.savez_compressed(
        result_file,
        X=X,
        labels=labels,
        lat=lat,
        lon=lon,
        n_clusters=n_clusters
    )
    print(f"结果保存: {result_file}")
    
    # 按年月拆分保存
    start_idx = 0
    for meta in metadata_list:
        year = meta['year']
        month = meta['month']
        n_samples = sum(1 for m in metadata_list if m['year'] == year and m['month'] == month)
        
        # 简化：只保存全局结果，详细拆分可后续处理
        break


def main():
    parser = argparse.ArgumentParser(
        description="GPM-ERA5 447维数据层次聚类分析"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_MATCHED_NEAREST_INPUT_DIR,
        help=f"匹配结果目录 (默认: {DEFAULT_MATCHED_NEAREST_INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_HIERARCHICAL_CLUSTERING_OUTPUT_DIR,
        help=f"聚类结果根目录 (默认: {DEFAULT_HIERARCHICAL_CLUSTERING_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--feature-preset",
        type=str,
        default="default",
        help="GPM特征预设 (默认: default/S2, 可选: S1_radar_precipitation, S2_radar_microphysics_full, S3_radar_precip_dsd, S4_full_gpm)"
    )
    parser.add_argument(
        "--k-over",
        type=int,
        default=100,
        help="过度聚类K值 (默认: 100)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8],
        help="评估的类别数范围 (默认: 3 4 5 6 7 8)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200000,
        help="最大样本数 (默认: 200000)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子 (默认: 42)"
    )
    parser.add_argument(
        "--nan-filter",
        type=str,
        default=None,
        choices=[None, 'dual_freq', 'strict'],
        help="NaN样本过滤策略 (默认: None, 可选: dual_freq-双频核心区, strict-严格删除所有NaN)"
    )
    parser.add_argument(
        "--dual-freq-mode",
        action="store_true",
        help="启用双频核心区模式: 使用31维双频特征集并删除Ka波段含NaN的样本(~损失47%，剩余~220万样本)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GPM-ERA5 447维层次聚类分析")
    print("=" * 70)
    
    # 双频核心区模式
    if args.dual_freq_mode:
        print("【双频核心区模式】启用")
        print("  - 使用特征集: DUAL_FREQ_CORE_INDICES (31维)")
        print("  - NaN过滤: 删除Ka波段列(5,6,8,9)含NaN的样本")
        print("  - 预期样本损失: ~47%，剩余~220万样本用于聚类")
        feature_indices = DUAL_FREQ_CORE_INDICES
        nan_filter = 'dual_freq'
    else:
        # 加载特征索引
        feature_indices = load_cluster_feature_indices(args.feature_preset)
        nan_filter = args.nan_filter
    
    print(f"特征维度: {len(feature_indices)}")
    if nan_filter:
        print(f"NaN过滤: {nan_filter}")
    
    # 查找输入文件
    input_files = find_matched_nearest_files(args.input_dir)
    
    # 构建全局矩阵（使用智能加载优化）
    print("\n智能加载数据...")
    X, lat, lon, metadata_list = build_global_matrix(
        input_files, feature_indices, args.max_samples, 
        nan_filter=nan_filter,
        random_state=args.random_state
    )
    
    # 数据清洗和归一化
    print("\n数据预处理...")
    X_clean, scaler = clean_and_normalize(X, mode="minmax")
    
    # 第一阶段：过度聚类
    centroids, kmeans_model = stage1_over_clustering(
        X_clean, k_over=args.k_over, random_state=args.random_state
    )
    
    # 第二阶段：层次聚类
    results, Z = stage2_hierarchical_clustering(centroids, args.n_clusters)
    
    # 选择最佳类别数（根据DB指数）
    best_k = min(results.keys(), key=lambda k: results[k]['db_score'])
    print(f"\n推荐类别数: K={best_k} (DB指数最优)")
    
    # 使用最佳K重新聚类
    final_labels = fcluster(Z, best_k, criterion='maxclust') - 1
    
    # 将标签映射回原始样本
    stage1_labels = kmeans_model.labels_
    final_sample_labels = final_labels[stage1_labels]
    
    # 输出目录
    output_dir = os.path.join(args.output_dir, f"cluster_results_{args.feature_preset}_K{best_k}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化
    print("\n生成可视化...")
    plot_dendrogram(
        Z, 
        os.path.join(output_dir, "dendrogram.png"),
        args.n_clusters
    )
    plot_evaluation_curves(
        results,
        os.path.join(output_dir, "evaluation_curves.png")
    )
    
    # 保存结果
    save_results(X, final_sample_labels, lat, lon, metadata_list, output_dir, best_k)
    
    # 保存配置
    import json
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump({
            'feature_preset': args.feature_preset,
            'feature_indices': feature_indices,
            'feature_dim': len(feature_indices),
            'k_over': args.k_over,
            'best_k': int(best_k),
            'n_clusters_range': args.n_clusters,
            'max_samples': args.max_samples,
            'n_samples': X.shape[0]
        }, f, indent=2)
    print(f"配置保存: {config_file}")
    
    print("\n" + "=" * 70)
    print("聚类分析完成")
    print(f"输出目录: {output_dir}")
    print(f"样本数: {X.shape[0]}")
    print(f"特征维度: {len(feature_indices)}")
    print(f"类别数: {best_k}")
    print("=" * 70)


if __name__ == "__main__":
    main()
