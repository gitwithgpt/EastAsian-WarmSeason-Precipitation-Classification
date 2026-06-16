import os
import torch
import numpy as np
from yaml import safe_load
from .nearest_match import nearest_neighbor_match


def load_match_config():
    """加载匹配配置文件"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config",
                               "match_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return safe_load(f)


def spatial_matching(gpm_tensor, era5_tensor, gpm_lat, gpm_lon, era5_lat, era5_lon, method="hybrid", **kwargs):
    """
    推荐流程：
    1. hybrid: 优先使用 IDW 插值（周围 k 个格点，更平滑），再用距离阈值(20–25km)做安全检查，过远像元 ERA5 置 NaN 并视为 far_match。
    2. nearest_only: 仅最近邻，阈值放宽到 20km。
    3. idw: 纯 IDW，无距离截断。
    4. nearest: 最近邻，阈值从配置读取。

    参数:
        gpm_tensor: (h, w, gpm_feat)
        era5_tensor: (h_era5, w_era5, era5_feat)
        gpm_lat, gpm_lon: (h, w)
        era5_lat, era5_lon: (h_era5, w_era5)
        method: "hybrid" | "nearest_only" | "idw" | "nearest"
        **kwargs: threshold_km, idw_k, idw_power, return_quality_flags

    返回:
        matched_raw: (h, w, gpm_feat + era5_feat)
        若 return_quality_flags=True 且 method="hybrid"，额外返回 quality_flags (h, w) 字符串数组: "ok" | "far_match" | "no_match"
    """
    era5_feat_dim = era5_tensor.shape[-1]
    h, w = gpm_lat.shape[0], gpm_lat.shape[1]
    device = era5_tensor.device
    dtype = era5_tensor.dtype
    threshold_km = kwargs.get("threshold_km", 20.0)
    idw_k = kwargs.get("idw_k", 4)
    idw_power = kwargs.get("idw_power", 2)
    return_quality_flags = kwargs.get("return_quality_flags", False)

    if method == "hybrid":
        from src.utils.spatial_interpolation import idw_interpolate_era5
        gpm_lat_np = gpm_lat.cpu().numpy().ravel()
        gpm_lon_np = gpm_lon.cpu().numpy().ravel()
        era5_lat_np = era5_lat.cpu().numpy().ravel()
        era5_lon_np = era5_lon.cpu().numpy().ravel()
        era5_flat = era5_tensor.cpu().numpy().reshape(-1, era5_feat_dim)
        # 返回 IDW 插值结果 + 每个像元到最近 ERA5 格点的 Haversine 距离(km)
        interp, nearest_dist_km = idw_interpolate_era5(
            gpm_lat_np, gpm_lon_np, era5_lat_np, era5_lon_np, era5_flat, k=idw_k, power=idw_power, return_max_dist_km=True
        )
        matched_era5 = torch.from_numpy(interp).reshape(h, w, era5_feat_dim).to(device=device, dtype=dtype)
        # 说明：
        # - IDW 本身对所有 GPM 像元进行插值，不再因为「距离过远」而直接置 NaN；
        # - threshold_km 仅用于质量标记（ok / far_match），真正无邻点时（nearest_dist_km 为 NaN）才视为 no_match。
        too_far = nearest_dist_km > threshold_km
        no_match = np.isnan(nearest_dist_km)
        # 关键修复：hybrid 方法必须将 far_match / no_match 像元的 ERA5 部分置 NaN
        # 否则即使不返回 quality_flags，这些不可靠插值也会进入下游特征提取和聚类
        invalid_mask = too_far | no_match  # 展平后的 bool 数组
        if invalid_mask.any():
            matched_era5_np = matched_era5.cpu().numpy()
            matched_era5_np.reshape(-1, era5_feat_dim)[invalid_mask, :] = np.nan
            matched_era5 = torch.from_numpy(matched_era5_np).to(device=device, dtype=dtype)

        if return_quality_flags:
            quality_flags = np.empty(nearest_dist_km.shape, dtype=object)
            quality_flags[:] = "ok"
            quality_flags[too_far] = "far_match"
            quality_flags[no_match] = "no_match"
            quality_flags = quality_flags.reshape(h, w)
            matched_raw = torch.cat([gpm_tensor, matched_era5], dim=-1)
            return matched_raw, quality_flags
        matched_raw = torch.cat([gpm_tensor, matched_era5], dim=-1)
        return matched_raw

    if method == "nearest_only":
        matched_era5 = nearest_neighbor_match(
            gpm_tensor, era5_tensor, gpm_lat, gpm_lon, era5_lat, era5_lon, threshold=threshold_km, era5_feat_dim=era5_feat_dim
        )
        return torch.cat([gpm_tensor, matched_era5], dim=-1)

    if method == "idw":
        from src.utils.spatial_interpolation import idw_interpolate_era5
        gpm_lat_np = gpm_lat.cpu().numpy().ravel()
        gpm_lon_np = gpm_lon.cpu().numpy().ravel()
        era5_lat_np = era5_lat.cpu().numpy().ravel()
        era5_lon_np = era5_lon.cpu().numpy().ravel()
        era5_flat = era5_tensor.cpu().numpy().reshape(-1, era5_feat_dim)
        interp = idw_interpolate_era5(
            gpm_lat_np, gpm_lon_np, era5_lat_np, era5_lon_np, era5_flat, k=idw_k, power=idw_power
        )
        matched_era5 = torch.from_numpy(interp).reshape(h, w, era5_feat_dim).to(device=device, dtype=dtype)
        return torch.cat([gpm_tensor, matched_era5], dim=-1)

    # method == "nearest"
    matched_era5 = nearest_neighbor_match(
        gpm_tensor, era5_tensor, gpm_lat, gpm_lon, era5_lat, era5_lon, threshold=threshold_km, era5_feat_dim=era5_feat_dim
    )
    return torch.cat([gpm_tensor, matched_era5], dim=-1)


def match_gpm_era5(gpm_tensor, era5_tensor, gpm_lat, gpm_lon, era5_lat, era5_lon):
    """
    GPM-ERA5 匹配入口：从 config 读取 spatial_matching 后调用 spatial_matching。
    返回 (h, w, 447) 的 matched_raw，不返回 quality_flags（保持与现有调用方兼容）。
    """
    config = load_match_config()
    sm = config.get("spatial_matching") or {}
    method = (sm.get("method") or "hybrid").lower()
    threshold_km = sm.get("nearest_threshold") or config["match_rule"].get("dist_threshold_km", 20.0)
    idw_k = sm.get("idw_k", 4)
    idw_power = sm.get("idw_power", 2)

    out = spatial_matching(
        gpm_tensor, era5_tensor, gpm_lat, gpm_lon, era5_lat, era5_lon,
        method=method,
        threshold_km=threshold_km,
        idw_k=idw_k,
        idw_power=idw_power,
        return_quality_flags=False,
    )
    return out