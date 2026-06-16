"""
GPM特征子集敏感性分析系统

执行多个特征子集的对比实验，评估不同物理机制对聚类效果的影响
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# 限制线程数
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster
import warnings
warnings.filterwarnings('ignore')

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载聚类脚本中的函数
from cluster_hierarchical_kmeans_447 import (
    load_cluster_feature_indices,
    build_global_matrix,
    clean_and_normalize,
    DEFAULT_MATCHED_NEAREST_INPUT_DIR,
    DEFAULT_HIERARCHICAL_CLUSTERING_OUTPUT_DIR,
)

# 配置文件路径
SENSITIVITY_CONFIG_PATH = PROJECT_ROOT / "config" / "cluster_sensitivity_config.yaml"


def load_sensitivity_config() -> Dict:
    """加载敏感性分析配置"""
    from yaml import safe_load
    with open(SENSITIVITY_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return safe_load(f)


def calculate_physical_plausibility(X: np.ndarray, labels: np.ndarray,
                                   original_indices: List[int]) -> float:
    """
    动态物理合理性评估
    
    根据子集中实际包含的特征（通过 original_indices 识别），
    动态选择可用的物理指标进行一致性评估。
    
    核心假设：对流型降水应有更高的强度（降水率、反射率）和更高的回波顶。
    
    Args:
        X: 归一化后的特征矩阵（列顺序与 original_indices 一致）
        labels: 聚类标签
        original_indices: 原始特征索引列表（对应 X 的各列）
    
    返回: 0-1之间的物理合理性分数（越高越好），特征不足时返回 NaN
    """
    try:
        # 建立原始特征索引 -> X列位置的映射
        idx_to_col = {orig_idx: col_idx for col_idx, orig_idx in enumerate(original_indices)}
        
        # 定义物理特征类别（原始特征索引）
        INTENSITY_INDICES = [0, 4, 7]        # 降水率、近地面Ku、柱内最大Ku
        HEIGHT_INDICES = [1, 13, 14, 15]     # 风暴顶、最大反射率高度
        MICROPHYSICS_INDICES = [16, 17]      # 冰相层、液相层厚度
        
        # 检查当前子集中可用的特征
        available_intensity = [idx for idx in INTENSITY_INDICES if idx in idx_to_col]
        available_height = [idx for idx in HEIGHT_INDICES if idx in idx_to_col]
        available_microphysics = [idx for idx in MICROPHYSICS_INDICES if idx in idx_to_col]
        
        all_available = available_intensity + available_height + available_microphysics
        
        if len(all_available) < 2:
            return np.nan  # 可用特征不足，无法评估
        
        # 计算每个簇在各物理特征上的均值
        cluster_means = {}
        for k in np.unique(labels):
            mask = labels == k
            if np.sum(mask) > 0:
                cluster_means[k] = {}
                for idx in all_available:
                    col = idx_to_col[idx]
                    cluster_means[k][idx] = np.nanmean(X[mask, col])
        
        if len(cluster_means) < 2:
            return np.nan
        
        sorted_keys = sorted(cluster_means.keys())
        correlations = []
        
        # 1. 强度指标之间的一致性（降水率 vs 反射率 vs 回波顶）
        if len(available_intensity) >= 2:
            for i in range(len(available_intensity)):
                for j in range(i + 1, len(available_intensity)):
                    means_i = [cluster_means[k][available_intensity[i]] for k in sorted_keys]
                    means_j = [cluster_means[k][available_intensity[j]] for k in sorted_keys]
                    rank_i = np.argsort(np.argsort(means_i)[::-1])
                    rank_j = np.argsort(np.argsort(means_j)[::-1])
                    n = len(rank_i)
                    if n > 1:
                        corr = 1 - 6 * np.sum((rank_i - rank_j) ** 2) / (n * (n ** 2 - 1))
                        correlations.append(max(0, (corr + 1) / 2))
        
        # 2. 强度-高度相关性（对流应有高降水+高回波顶）
        if len(available_intensity) >= 1 and len(available_height) >= 1:
            for int_idx in available_intensity:
                for h_idx in available_height:
                    means_i = [cluster_means[k][int_idx] for k in sorted_keys]
                    means_h = [cluster_means[k][h_idx] for k in sorted_keys]
                    rank_i = np.argsort(np.argsort(means_i)[::-1])
                    rank_h = np.argsort(np.argsort(means_h)[::-1])
                    n = len(rank_i)
                    if n > 1:
                        corr = 1 - 6 * np.sum((rank_i - rank_h) ** 2) / (n * (n ** 2 - 1))
                        correlations.append(max(0, (corr + 1) / 2))
        
        # 3. 微物理-强度相关性（冰相层厚度应对流>层状）
        if len(available_microphysics) >= 1 and len(available_intensity) >= 1:
            for mp_idx in available_microphysics:
                for int_idx in available_intensity:
                    means_mp = [cluster_means[k][mp_idx] for k in sorted_keys]
                    means_i = [cluster_means[k][int_idx] for k in sorted_keys]
                    rank_mp = np.argsort(np.argsort(means_mp)[::-1])
                    rank_i = np.argsort(np.argsort(means_i)[::-1])
                    n = len(rank_mp)
                    if n > 1:
                        corr = 1 - 6 * np.sum((rank_mp - rank_i) ** 2) / (n * (n ** 2 - 1))
                        correlations.append(max(0, (corr + 1) / 2))
        
        if not correlations:
            return np.nan
        
        physical_score = float(np.mean(correlations))
        return physical_score
    except Exception as e:
        return np.nan


def calculate_stability_score(X: np.ndarray, sample_labels: np.ndarray,
                              k_over: int = 150, n_clusters: int = 4,
                              n_bootstrap: int = 5) -> Tuple[float, float]:
    """
    Bootstrap稳定性评估 — 使用与主实验相同的两阶段聚类流程
    
    流程：
    1. 子采样80%数据
    2. 对子样本重新执行两阶段聚类（K-means过度聚类 + Ward层次）
    3. 计算子样本标签与全量标签的ARI（Adjusted Rand Index）
    
    Args:
        X: 归一化后的特征矩阵
        sample_labels: 全量样本在当前聚类参数下的标签（参考标签）
        k_over: 过度聚类参数，默认150（与主实验一致）
        n_clusters: 最终类别数
        n_bootstrap: Bootstrap迭代次数
    
    返回: (平均ARI, ARI标准差)
    """
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    from scipy.cluster.hierarchy import linkage, fcluster
    
    try:
        n_samples = X.shape[0]
        sample_size = int(0.8 * n_samples)
        aris = []
        
        for i in range(n_bootstrap):
            rng = np.random.RandomState(42 + i * 100)
            indices = rng.choice(n_samples, sample_size, replace=False)
            X_sub = X[indices].copy()
            
            # 归一化子样本（与主实验一致的数据处理）
            X_sub = np.where(np.isinf(X_sub), np.nan, X_sub)
            for col in range(X_sub.shape[1]):
                c = X_sub[:, col]
                m = np.isnan(c)
                if np.any(m):
                    X_sub[m, col] = np.nanmedian(c) if not np.all(m) else 0.0
            X_sub_norm = MinMaxScaler().fit_transform(X_sub)
            
            # 第一阶段：过度聚类（与主实验一致）
            kmeans_sub = KMeans(n_clusters=k_over, random_state=42 + i, n_init=10, max_iter=300)
            sub_stage1 = kmeans_sub.fit_predict(X_sub_norm)
            sub_centroids = kmeans_sub.cluster_centers_
            
            # 第二阶段：Ward层次聚类（与主实验一致）
            Z_sub = linkage(sub_centroids, method='ward')
            sub_labels_centroid = fcluster(Z_sub, n_clusters, criterion='maxclust') - 1
            sub_labels = sub_labels_centroid[sub_stage1]
            
            # 计算ARI
            ari = adjusted_rand_score(sample_labels[indices], sub_labels)
            aris.append(ari)
        
        return np.mean(aris), np.std(aris)
    except Exception as e:
        return np.nan, np.nan


def calculate_balance_score(labels: np.ndarray) -> float:
    """
    类大小平衡性评估（归一化熵）
    
    避免极端不平衡（如99%样本在一个类）
    
    返回: 0-1之间的平衡分数（1表示完全平衡）
    """
    from scipy.stats import entropy
    
    try:
        counts = np.bincount(labels)
        total = np.sum(counts)
        
        if total == 0:
            return 0.0
        
        probs = counts / total
        # 计算归一化熵
        max_entropy = np.log(len(counts))
        actual_entropy = entropy(probs)
        
        return actual_entropy / max_entropy if max_entropy > 0 else 0.0
    except Exception as e:
        return 0.0


def run_single_clustering(X: np.ndarray, config_name: str, k_range: List[int],
                         feature_indices: List[int], k_over: int = 150) -> Dict:
    """
    对单个特征子集进行聚类分析（统一参数版）
    
    Args:
        X: 特征矩阵
        config_name: 配置名称
        k_range: 评估的类别数范围
        feature_indices: 原始特征索引列表（用于物理合理性评估）
        k_over: 过度聚类参数，默认150（与主实验一致）
    
    返回包含各种评估指标的字典
    """
    print(f"\n  处理配置: {config_name}")
    
    # 数据预处理
    X_clean, scaler = clean_and_normalize(X, mode="minmax")
    n_samples = X_clean.shape[0]
    
    # 子采样（如果样本过多）
    max_samples = 200000
    if n_samples > max_samples:
        indices = np.random.choice(n_samples, max_samples, replace=False)
        X_clean = X_clean[indices]
        n_samples = max_samples
    
    # 第一阶段：过度聚类（统一 k_over=150，与主实验一致）
    print(f"  两阶段聚类: k_over={k_over}, K范围={k_range}")
    kmeans = KMeans(n_clusters=k_over, random_state=42, n_init=10, max_iter=300)
    stage1_labels = kmeans.fit_predict(X_clean)
    centroids = kmeans.cluster_centers_
    
    # 第二阶段：层次聚类
    Z = linkage(centroids, method='ward')
    
    results = {}
    for n_clusters in k_range:
        # 获取聚类标签
        labels = fcluster(Z, n_clusters, criterion='maxclust') - 1
        sample_labels = labels[stage1_labels]
        
        # 计算传统统计指标
        try:
            db_score = davies_bouldin_score(X_clean, sample_labels)
        except:
            db_score = np.nan
            
        try:
            sil_score = silhouette_score(X_clean, sample_labels)
        except:
            sil_score = np.nan
            
        try:
            ch_score = calinski_harabasz_score(X_clean, sample_labels)
        except:
            ch_score = np.nan
        
        # 计算类分布
        unique, counts = np.unique(sample_labels, return_counts=True)
        min_class_size = np.min(counts)
        max_class_size = np.max(counts)
        
        # 物理合理性指标（动态特征检测，根据子集实际包含的特征评估）
        physical_score = calculate_physical_plausibility(X_clean, sample_labels, feature_indices)
        
        # 稳定性指标（使用与主实验相同的两阶段聚类流程）
        if n_clusters <= 5:
            stability_mean, stability_std = calculate_stability_score(
                X_clean, sample_labels, k_over=k_over, n_clusters=n_clusters, n_bootstrap=5
            )
        else:
            stability_mean, stability_std = np.nan, np.nan
        
        # 平衡性指标
        balance_score = calculate_balance_score(sample_labels)
        
        results[n_clusters] = {
            'davies_bouldin': db_score,
            'silhouette': sil_score,
            'calinski_harabasz': ch_score,
            'physical_plausibility': physical_score,
            'stability_mean': stability_mean,
            'stability_std': stability_std,
            'balance_score': balance_score,
            'min_class_size': int(min_class_size),
            'max_class_size': int(max_class_size),
            'n_samples': n_samples
        }
    
    return results


def run_experiment_group(
    input_files: List[str],
    group_configs: Dict,
    k_range: List[int],
    output_dir: str,
    max_samples: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    运行一个实验组的所有配置（优化版，支持智能加载和可复现采样）
    
    Args:
        input_files: 输入文件列表
        group_configs: 实验组配置字典
        k_range: 类别数范围
        output_dir: 输出目录
        max_samples: 最大样本数（可选，用于子采样）
        random_state: 随机种子（用于可复现的子采样）
    
    返回结果DataFrame
    """
    all_results = []
    
    for config_name, config_info in group_configs.items():
        print(f"\n{'='*70}")
        print(f"配置: {config_name}")
        print(f"描述: {config_info.get('description', 'N/A')}")
        print(f"维度: {config_info.get('dim_count', 'N/A')}")
        print(f"{'='*70}")
        
        feature_indices = config_info['indices']
        
        # 检查是否有NaN过滤配置（默认使用strict，与主实验22维一致）
        nan_filter = config_info.get('nan_filter', 'strict')
        if nan_filter:
            print(f"  NaN过滤策略: {nan_filter}")
            if config_info.get('sample_loss'):
                print(f"  预期样本损失: {config_info.get('sample_loss')}")
        
        # 加载数据（使用智能加载优化）
        try:
            print(f"  智能加载数据...")
            X, lat, lon, metadata = build_global_matrix(
                input_files, feature_indices, 
                max_samples=max_samples,
                nan_filter=nan_filter,
                random_state=random_state
            )
            print(f"  数据矩阵: {X.shape}")
        except Exception as e:
            print(f"  错误: 加载数据失败 - {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 运行聚类
        clustering_results = run_single_clustering(X, config_name, k_range, 
                                                   feature_indices=feature_indices, 
                                                   k_over=150)
        
        # 整理结果（包含新指标）
        for k, metrics in clustering_results.items():
            all_results.append({
                'config_name': config_name,
                'description': config_info.get('description', ''),
                'dim_count': config_info.get('dim_count', len(feature_indices)),
                'k_clusters': k,
                'davies_bouldin': metrics['davies_bouldin'],
                'silhouette': metrics['silhouette'],
                'calinski_harabasz': metrics['calinski_harabasz'],
                'physical_plausibility': metrics['physical_plausibility'],
                'stability_mean': metrics['stability_mean'],
                'stability_std': metrics['stability_std'],
                'balance_score': metrics['balance_score'],
                'min_class_size': metrics['min_class_size'],
                'max_class_size': metrics['max_class_size'],
                'n_samples': metrics['n_samples']
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(all_results)
    
    # 保存
    output_file = os.path.join(output_dir, 'sensitivity_results.csv')
    df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")
    
    return df


def generate_comparison_report(df: pd.DataFrame, output_dir: str):
    """生成对比报告和可视化"""
    
    # 1. 热力图 - 各配置在不同K值下的DB指数
    pivot_db = df.pivot_table(
        index='config_name',
        columns='k_clusters',
        values='davies_bouldin',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    import seaborn as sns
    sns.heatmap(pivot_db, annot=True, fmt='.3f', cmap='YlGnBu_r', cbar_kws={'label': 'Davies-Bouldin (lower is better)'})
    plt.title('Davies-Bouldin Index Heatmap by Configuration and K')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_db.png'), dpi=150)
    plt.close()
    
    # 2. 轮廓系数热力图
    pivot_sil = df.pivot_table(
        index='config_name',
        columns='k_clusters',
        values='silhouette',
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_sil, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'Silhouette Score (higher is better)'})
    plt.title('Silhouette Score Heatmap by Configuration and K')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_silhouette.png'), dpi=150)
    plt.close()
    
    # 3. 维度-效果散点图
    best_results = df.loc[df.groupby('config_name')['davies_bouldin'].idxmin()]
    
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        best_results['dim_count'],
        best_results['davies_bouldin'],
        c=best_results['silhouette'],
        s=200,
        cmap='RdYlGn',
        edgecolors='black',
        linewidth=1
    )
    
    # 添加标签
    for idx, row in best_results.iterrows():
        plt.annotate(
            row['config_name'],
            (row['dim_count'], row['davies_bouldin']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    
    plt.xlabel('Number of Features')
    plt.ylabel('Best Davies-Bouldin Index')
    plt.title('Dimension vs Clustering Quality\n(color = Silhouette Score)')
    plt.colorbar(scatter, label='Silhouette Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dim_vs_quality.png'), dpi=150)
    plt.close()
    
    # 4. 物理合理性指标热力图
    if 'physical_plausibility' in df.columns:
        pivot_physical = df.pivot_table(
            index='config_name',
            columns='k_clusters',
            values='physical_plausibility',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_physical, annot=True, fmt='.3f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Physical Plausibility (higher is better)'})
        plt.title('Physical Plausibility Score (NEW)\nPhysical consistency of clustering results')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_physical.png'), dpi=150)
        plt.close()
    
    # 5. 稳定性指标热力图
    if 'stability_mean' in df.columns:
        pivot_stability = df.pivot_table(
            index='config_name',
            columns='k_clusters',
            values='stability_mean',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_stability, annot=True, fmt='.3f', cmap='RdYlGn',
                   cbar_kws={'label': 'Stability ARI (higher is better)'})
        plt.title('Bootstrap Stability Score (NEW)\nConsistency across sub-samples')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'heatmap_stability.png'), dpi=150)
        plt.close()
    
    # 6. 生成Markdown报告（包含新指标）
    report = []
    report.append("# GPM特征子集敏感性分析报告（优化版）\n")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("## 评价指标说明\n")
    report.append("- **DB指数**: 统计质量（越低越好）\n")
    report.append("- **轮廓系数**: 样本分离度（越高越好）\n")
    report.append("- **物理合理性**: 对流/层状特征一致性（NEW，越高越好）\n")
    report.append("- **稳定性**: Bootstrap一致性（NEW，越高越好）\n")
    report.append("- **平衡性**: 类大小均匀度（NEW，1=完全平衡）\n\n")
    
    # 总结表格（包含新指标）
    report.append("## 1. 各配置最佳表现对比\n")
    report.append("| 配置 | 维度 | 最佳K | DB指数 | 轮廓系数 | 物理合理性 | 稳定性 | 平衡性 |\n")
    report.append("|------|------|-------|--------|----------|------------|--------|--------|\n")
    
    for idx, row in best_results.iterrows():
        physical_str = f"{row['physical_plausibility']:.3f}" if not pd.isna(row['physical_plausibility']) else "N/A"
        stability_str = f"{row['stability_mean']:.3f}" if not pd.isna(row['stability_mean']) else "N/A"
        balance_str = f"{row['balance_score']:.3f}" if not pd.isna(row['balance_score']) else "N/A"
        
        report.append(
            f"| {row['config_name']} | {row['dim_count']} | {row['k_clusters']} | "
            f"{row['davies_bouldin']:.3f} | {row['silhouette']:.3f} | "
            f"{physical_str} | {stability_str} | {balance_str} |\n"
        )
    
    report.append("\n## 2. 关键发现\n")
    
    # 找出最佳配置（基于DB指数）
    best_config = best_results.loc[best_results['davies_bouldin'].idxmin()]
    report.append(f"### 2.1 统计最优配置（DB指数最低）\n")
    report.append(f"- **配置名称**: {best_config['config_name']}\n")
    report.append(f"- **维度**: {best_config['dim_count']}\n")
    report.append(f"- **最佳类别数**: K={best_config['k_clusters']}\n")
    report.append(f"- **DB指数**: {best_config['davies_bouldin']:.3f}\n")
    report.append(f"- **轮廓系数**: {best_config['silhouette']:.3f}\n\n")
    
    # 找出物理最合理的配置
    if 'physical_plausibility' in best_results.columns:
        valid_physical = best_results.dropna(subset=['physical_plausibility'])
        if len(valid_physical) > 0:
            best_physical = valid_physical.loc[valid_physical['physical_plausibility'].idxmax()]
            report.append(f"### 2.2 物理最合理配置\n")
            report.append(f"- **配置名称**: {best_physical['config_name']}\n")
            report.append(f"- **物理合理性**: {best_physical['physical_plausibility']:.3f}\n")
            report.append(f"- **说明**: 对流/层状分离最符合物理预期\n\n")
    
    # 找出最稳定配置
    if 'stability_mean' in best_results.columns:
        valid_stability = best_results.dropna(subset=['stability_mean'])
        if len(valid_stability) > 0:
            best_stable = valid_stability.loc[valid_stability['stability_mean'].idxmax()]
            report.append(f"### 2.3 最稳定配置\n")
            report.append(f"- **配置名称**: {best_stable['config_name']}\n")
            report.append(f"- **稳定性ARI**: {best_stable['stability_mean']:.3f} ± {best_stable['stability_std']:.3f}\n")
            report.append(f"- **说明**: Bootstrap子采样一致性最高\n\n")
    
    # 计算性价比（效果/维度比）
    best_results['efficiency'] = best_results['silhouette'] / best_results['dim_count']
    best_efficiency = best_results.loc[best_results['efficiency'].idxmax()]
    report.append(f"### 2.4 最佳性价比（效果/维度）\n")
    report.append(f"- **配置名称**: {best_efficiency['config_name']}\n")
    report.append(f"- **效率值**: {best_efficiency['efficiency']:.4f}\n")
    report.append(f"- **说明**: 用最少维度获得相对好的效果\n\n")
    
    # 保存报告
    report_path = os.path.join(output_dir, 'sensitivity_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="GPM特征子集敏感性分析")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_MATCHED_NEAREST_INPUT_DIR,
        help=f"匹配结果目录 (默认: {DEFAULT_MATCHED_NEAREST_INPUT_DIR})"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(
            DEFAULT_HIERARCHICAL_CLUSTERING_OUTPUT_DIR, "sensitivity_analysis"
        ),
        help="敏感性分析输出目录",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=['all', 'n1', 'baseline', 'mechanism', 'progressive', 'reduction', 'dual_freq'],
        default='all',
        help="要运行的实验组: n1(N1对照), baseline(基线), mechanism/progressive(机制), dual_freq(双频核心区), reduction(降维), all(全部)"
    )
    parser.add_argument(
        "--k-range",
        type=int,
        nargs="+",
        default=[3, 4, 5, 6, 7, 8],
        help="评估的类别数范围"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大样本数（可选，用于子采样。None表示使用全部样本）"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子（用于可复现的子采样，默认42）"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GPM特征子集敏感性分析系统")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    print("\n加载敏感性分析配置...")
    config = load_sensitivity_config()
    
    # 查找输入文件
    from cluster_hierarchical_kmeans_447 import find_matched_nearest_files
    input_files = find_matched_nearest_files(args.input_dir)
    
    # 确定要运行的实验组（包含N1对照组 + 4个核心组）
    if args.experiment == 'all':
        # 核心实验组（增加N1对照组）
        groups_to_run = {
            '0_N1_Control': config['n1_control_group'],            # N1 22维精确对照
            '1_Baseline_Necessity': config['baseline_group'],         # 基线与必要性
            '2_Mechanism_Progressive': config['mechanism_group'],   # 机制分离+渐进（合并）
            '3_Dual_Frequency_Core': config['dual_freq_group'],     # 双频核心区专项（重点）
            '4_Dimension_Reduction': config['reduction_group']      # 降维验证
        }
    else:
        # 支持的单个实验组
        group_map = {
            'n1': ('0_N1_Control', config['n1_control_group']),
            'baseline': ('1_Baseline_Necessity', config['baseline_group']),
            'mechanism': ('2_Mechanism_Progressive', config['mechanism_group']),
            'progressive': ('2_Mechanism_Progressive', config['mechanism_group']),
            'dual_freq': ('3_Dual_Frequency_Core', config['dual_freq_group']),
            'reduction': ('4_Dimension_Reduction', config['reduction_group'])
        }
        if args.experiment not in group_map:
            print(f"错误: 未知实验组 '{args.experiment}'")
            print(f"可用选项: {list(group_map.keys())}")
            return
        name, group = group_map[args.experiment]
        groups_to_run = {name: group}
    
    # 运行所有实验组
    all_results = []
    for group_name, group_configs in groups_to_run.items():
        print(f"\n{'='*80}")
        print(f"实验组: {group_name}")
        print(f"{'='*80}")
        
        group_output = os.path.join(args.output_dir, group_name.lower())
        os.makedirs(group_output, exist_ok=True)
        
        df = run_experiment_group(
            input_files,
            group_configs,
            args.k_range,
            group_output
        )
        
        df['experiment_group'] = group_name
        all_results.append(df)
        
        # 生成该组的报告
        generate_comparison_report(df, group_output)
    
    # 合并所有结果
    if len(all_results) > 1:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_output = args.output_dir
        generate_comparison_report(combined_df, combined_output)
        
        # 保存合并结果
        combined_df.to_csv(
            os.path.join(combined_output, 'all_sensitivity_results.csv'),
            index=False
        )
    
    print("\n" + "=" * 80)
    print("敏感性分析完成")
    print(f"输出目录: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
