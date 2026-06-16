import torch
from .distance_calc import haversine_distance


def nearest_neighbor_match(gpm_tensor, era5_tensor, gpm_lat, gpm_lon, era5_lat, era5_lon, threshold=20.0, era5_feat_dim=None):
    """
    最邻近匹配：为每个GPM像元匹配最近的ERA5像元（≤阈值）
    :param gpm_tensor: GPM数据Tensor (h, w, gpm_feat) - gpm_feat=38（根据tezhengjiyouhua.txt）
    :param era5_tensor: ERA5数据Tensor (h, w, era5_feat) - era5_feat=16（pressure 5 + single 11）
    :param gpm_lat: GPM纬度数组 (h, w)
    :param gpm_lon: GPM经度数组 (h, w)
    :param era5_lat: ERA5纬度数组 (h, w)
    :param era5_lon: ERA5经度数组 (h, w)
    :param threshold: 距离阈值(km)，建议15-25（ERA5分辨率约0.25°≈25km）
    :param era5_feat_dim: ERA5特征维度（可选，如果不提供则从era5_tensor.shape获取）
    :return: 匹配后的ERA5数据Tensor (h, w, era5_feat)
    """
    h, w, gpm_feat = gpm_tensor.shape
    if era5_feat_dim is None:
        era5_feat = era5_tensor.shape[-1]
    else:
        era5_feat = era5_feat_dim
    
    # 创建匹配后的ERA5数据Tensor，特征维度应与ERA5一致
    # 使用NaN初始化，表示未匹配的位置（而不是0，避免在聚类时被误认为是有效值）
    matched_era5 = torch.full((h, w, era5_feat), float('nan'), dtype=era5_tensor.dtype, device=era5_tensor.device)

    # 展平维度
    gpm_lat_flat = gpm_lat.flatten()
    gpm_lon_flat = gpm_lon.flatten()
    era5_lat_flat = era5_lat.flatten()
    era5_lon_flat = era5_lon.flatten()
    era5_flat = era5_tensor.flatten(0, 1)  # (h*w, era5_feat)

    for i in range(len(gpm_lat_flat)):
        # 计算当前GPM像元到所有ERA5像元的距离
        dist = haversine_distance(gpm_lat_flat[i], gpm_lon_flat[i], era5_lat_flat, era5_lon_flat)
        # 找到最近且≤阈值的ERA5像元
        min_dist_idx = torch.argmin(dist)
        if dist[min_dist_idx] <= threshold:
            matched_era5.flatten(0, 1)[i] = era5_flat[min_dist_idx]
        # 如果距离超过阈值，保持NaN（表示未匹配）

    return matched_era5