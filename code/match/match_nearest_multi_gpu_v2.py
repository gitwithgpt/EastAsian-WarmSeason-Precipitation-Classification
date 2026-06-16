"""
GPM-ERA5 最近邻匹配 - 多GPU并行版本 (V2改进版)
改进内容：
1. 按月增量保存 - 每月处理完成后立即保存，避免内存堆积
2. 内存优化 - 分块读取GPM大数组，避免一次性加载
3. 更好的错误处理和日志

作者: leaf
日期: 2026-04-24
"""

import os
import sys
import logging
import time
import argparse
import warnings
import gc
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
import torch.multiprocessing as mp
from yaml import safe_load
import h5py
import xarray as xr

from src.match.match_pressure import near_match_era5_pressure_gpm_full
from src.match.match_single import near_match_era5_single_gpm_full
from src.io_utils.log_utils import init_logger
from src.io_utils.path_utils import create_dir, get_gpm_root_path


_CONFIG_CACHE = {}


# ERA5数据基础路径（硬编码）
ERA5_BASE_DIR = r"E:\\ERA5"


def get_era5_pressure_dir(year, config):
    """
    获取ERA5压力层数据目录
    固定路径: E:\\ERA5\\{year}
    """
    era5_dir = os.path.join(ERA5_BASE_DIR, str(year))
    return era5_dir
   


def get_era5_single_dir(year, config):
    """
    获取ERA5单层数据目录
    固定路径: E:\\ERA5\\SingleLevel\\{year}
    """
    era5_dir = os.path.join(ERA5_BASE_DIR, "SingleLevel", str(year))
    return era5_dir   


def load_config():
    """加载项目配置（带缓存）"""
    if 'config' in _CONFIG_CACHE:
        return _CONFIG_CACHE['config']

    config_dir = Path(__file__).parent.parent.parent / "config"

    with open(config_dir / "base_config.yaml", "r", encoding="utf-8") as f:
        base_config = safe_load(f)

    with open(config_dir / "match_config.yaml", "r", encoding="utf-8") as f:
        match_config = safe_load(f)

    with open(config_dir / "multi_gpu_config.yaml", "r", encoding="utf-8") as f:
        gpu_config = safe_load(f)

    config = {**base_config, **match_config, **gpu_config}
    _CONFIG_CACHE['config'] = config
    return config


def setup_logging(gpu_id=None, month_key=None):
    """设置日志"""
    config = load_config()
    log_dir = Path(config["project_root"]) / "temp" / "logs"
    create_dir(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if gpu_id is not None and month_key is not None:
        log_path = log_dir / f"match_v2_gpu{gpu_id}_{month_key}_{timestamp}.log"
    elif gpu_id is not None:
        log_path = log_dir / f"match_v2_gpu{gpu_id}_{timestamp}.log"
    else:
        log_path = log_dir / f"match_v2_main_{timestamp}.log"

    return init_logger(str(log_path))


def get_time_info_from_gpm(gpm_file_path):
    """从GPM HDF5文件中提取时间信息"""
    with h5py.File(gpm_file_path, 'r') as h5f:
        fs_group = h5f['FS']
        scan_time = fs_group['ScanTime']

        year_data = scan_time['Year'][:]
        month_data = scan_time['Month'][:]
        day_data = scan_time['DayOfMonth'][:]
        hour_data = scan_time['Hour'][:]

        valid_idx = None
        for i in range(len(year_data)):
            if (year_data[i] != -9999 and month_data[i] != -99 and
                day_data[i] != -99 and hour_data[i] != -99):
                valid_idx = i
                break

        if valid_idx is None:
            raise ValueError(f"无法从文件 {gpm_file_path} 提取有效时间信息")

        year = int(year_data[valid_idx])
        month = int(month_data[valid_idx])
        day = int(day_data[valid_idx])
        hour = int(hour_data[valid_idx])

        day_unique = day
        hour_unique = hour

    return year, month, day, hour, day_unique, hour_unique


def is_gpm_missing_value(data, dtype='float'):
    """
    判断GPM数据是否为缺失值
    根据DPR L2文档，不同数据类型的缺失值标记：
    - float: -9999.9, -9999.0, -28888.0 (计算错误值)
    - 需要同时检查NaN和Inf
    """
    if dtype == 'float':
        # float类型的缺失值标记
        missing_values = [-9999.9, -9999.0, -28888.0]
        # 使用np.isin进行精确比较，同时检查NaN和Inf
        mask = np.isin(data, missing_values) | np.isnan(data) | np.isinf(data)
        return mask
    elif dtype == 'short':
        missing_values = [-9999, -28888]
        return np.isin(data, missing_values)
    elif dtype == 'byte':
        missing_values = [-99]
        return np.isin(data, missing_values)
    elif dtype == 'int':
        missing_values = [-9999]
        return np.isin(data, missing_values)
    return np.zeros_like(data, dtype=bool)


def safe_nanargmax(arr, axis=1, default_idx=0):
    """安全的nanargmax，处理全NaN的情况"""
    try:
        return np.nanargmax(arr, axis=axis)
    except ValueError:
        n_samples = arr.shape[0] if len(arr.shape) > 1 else 1
        return np.full(n_samples, default_idx, dtype=np.int64)


def safe_nanmax(arr, axis=1, default_val=np.nan):
    """安全的nanmax，处理全NaN的情况"""
    all_nan_mask = np.all(np.isnan(arr), axis=axis)

    if np.all(all_nan_mask):
        n_samples = arr.shape[0] if len(arr.shape) > 1 else 1
        return np.full(n_samples, default_val, dtype=arr.dtype)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmax(arr, axis=axis)

    if np.any(all_nan_mask):
        if isinstance(result, np.ndarray):
            result[all_nan_mask] = default_val
        else:
            if all_nan_mask:
                result = default_val

    return result


def load_gpm_data_memory_efficient(gpm_file_path, lat_range, lon_range, min_rain_rate=0.5, logger=None):
    """
    内存优化版GPM数据加载 - 分块读取大数组，避免内存溢出
    改进：
    1. 先读取小数组确定有效像元索引
    2. 分块读取大数组，只处理有效像元，立即释放内存
    3. 逐个处理特征提取，避免同时持有多个大数组
    """
    fill_value = -9999.9

    try:
        with h5py.File(gpm_file_path, 'r') as h5f:
            fs_group = h5f['FS']

            # 第一步：先读取经纬度和降水率（小数组）确定有效像元
            lat = fs_group['Latitude'][:]
            lon = fs_group['Longitude'][:]
            precip_rate = fs_group['SLV']['precipRateNearSurface'][:]

            # 创建有效像元掩码
            spatial_mask = (
                (lat >= lat_range[0]) & (lat <= lat_range[1]) &
                (lon >= lon_range[0]) & (lon <= lon_range[1]) &
                ~is_gpm_missing_value(lat) & ~is_gpm_missing_value(lon)
            )
            precip_mask = (precip_rate >= min_rain_rate) & ~is_gpm_missing_value(precip_rate)
            valid_mask = spatial_mask & precip_mask

            if not np.any(valid_mask):
                if logger:
                    logger.debug(f"  研究区域内无有效降水数据")
                return None

            valid_scan_idx, valid_ray_idx = np.where(valid_mask)
            n_valid = len(valid_scan_idx)

            if logger:
                logger.info(f"  有效像元数: {n_valid}")

             # 获取维度信息
            nscan, nray = lat.shape
            nbin = fs_group['PRE']['height'].shape[2] if len(fs_group['PRE']['height'].shape) == 3 else 176
      # 预分配GPM特征数组
            gpm_features = np.zeros((n_valid, 47), dtype=np.float32)

            # 0: precipRateNearSurface
            gpm_features[:, 0] = precip_rate[valid_mask]

            # 分块读取辅助函数
            def read_chunks_2d(dataset, scan_idx, ray_idx, chunk_size=1000):
                """分块读取2D数组，只获取指定索引的数据"""
                result = np.zeros(len(scan_idx), dtype=dataset.dtype)
                for i in range(0, len(scan_idx), chunk_size):
                    end = min(i + chunk_size, len(scan_idx))
                    s_idx = scan_idx[i:end]
                    r_idx = ray_idx[i:end]
                    # 使用列表推导式逐个读取，避免大内存分配
                    for j, (s, r) in enumerate(zip(s_idx, r_idx)):
                        result[i + j] = dataset[s, r]
                return result
            def read_chunks_3d(dataset, scan_idx, ray_idx, chunk_size=500):
                """分块读取3D数组 (nscan, nray, nbin)，只获取指定索引的数据"""
                nbin_full = dataset.shape[2]
                result = np.zeros((len(scan_idx), nbin_full), dtype=dataset.dtype)
                for i in range(0, len(scan_idx), chunk_size):
                    end = min(i + chunk_size, len(scan_idx))
                    s_idx = scan_idx[i:end]
                    r_idx = ray_idx[i:end]
                    for j, (s, r) in enumerate(zip(s_idx, r_idx)):
                        result[i + j, :] = dataset[s, r, :]
                return result

            def read_chunks_3d_dual(dataset, scan_idx, ray_idx, chunk_size=500):
                """分块读取3D双频数组 (nscan, nray, 2)"""
                result = np.zeros((len(scan_idx), 2), dtype=dataset.dtype)
                for i in range(0, len(scan_idx), chunk_size):
                    end = min(i + chunk_size, len(scan_idx))
                    s_idx = scan_idx[i:end]
                    r_idx = ray_idx[i:end]
                    for j, (s, r) in enumerate(zip(s_idx, r_idx)):
                        result[i + j, 0] = dataset[s, r, 0]  # Ku
                        result[i + j, 1] = dataset[s, r, 1]  # Ka
                return result

            def read_chunks_4d(dataset, scan_idx, ray_idx, chunk_size=200):
                """分块读取4D数组 (nscan, nray, nbin, 2)，只获取指定索引的数据"""
                nbin_full = dataset.shape[2]
                result = np.zeros((len(scan_idx), nbin_full, 2), dtype=dataset.dtype)
                for i in range(0, len(scan_idx), chunk_size):
                    end = min(i + chunk_size, len(scan_idx))
                    s_idx = scan_idx[i:end]
                    r_idx = ray_idx[i:end]
                    for j, (s, r) in enumerate(zip(s_idx, r_idx)):
                        result[i + j, :, :] = dataset[s, r, :, :]
                return result

            # 读取高度相关变量 (2D)
            heightStormTop = read_chunks_2d(fs_group['PRE']['heightStormTop'], valid_scan_idx, valid_ray_idx)
            binClutterFreeBottom = read_chunks_2d(fs_group['PRE']['binClutterFreeBottom'], valid_scan_idx, valid_ray_idx)

            # heightZeroDeg 在 VER 组中，某些文件可能缺失
            try:
                heightZeroDeg = read_chunks_2d(fs_group['VER']['heightZeroDeg'], valid_scan_idx, valid_ray_idx)
            except (KeyError, ValueError) as e:
                if logger:
                    logger.warning(f"  heightZeroDeg 读取失败，使用NaN填充: {e}")
                heightZeroDeg = np.full(n_valid, np.nan, dtype=np.float32)

            # zFactorFinalNearSurface 是3D双频数组 (nscan, nray, 2)，使用专门的函数读取
            zFactorFinalNS = read_chunks_3d_dual(fs_group['SLV']['zFactorFinalNearSurface'], valid_scan_idx, valid_ray_idx, chunk_size=500)

            # 1: heightStormTop, 2: heightZeroDeg
            gpm_features[:, 1] = heightStormTop
            gpm_features[:, 2] = heightZeroDeg

            # 3: free_bottom_height (需要height数组)
            # 分块读取height并计算free_bottom_height
            height_dataset = fs_group['PRE']['height']
            nbin_actual = height_dataset.shape[2] if height_dataset.ndim >= 3 else nbin
            free_bottom_height = np.zeros(n_valid)

            for i in range(0, n_valid, 500):
                end = min(i + 500, n_valid)
                s_idx = valid_scan_idx[i:end]
                r_idx = valid_ray_idx[i:end]
                for j, (s, r) in enumerate(zip(s_idx, r_idx)):
                    bin_idx = int(binClutterFreeBottom[i + j])
                    if 0 <= bin_idx < nbin_actual:
                        free_bottom_height[i + j] = height_dataset[s, r, bin_idx]
                    else:
                        free_bottom_height[i + j] = np.nan

            gpm_features[:, 3] = free_bottom_height

            # 读取zFactorFinal (4D: nscan, nray, nbin, 2) - 这是大数组，分块处理
            zFactorFinal = read_chunks_4d(fs_group['SLV']['zFactorFinal'], valid_scan_idx, valid_ray_idx, chunk_size=300)
            zku_profile = zFactorFinal[:, :, 0]
            zka_profile = zFactorFinal[:, :, 1] if zFactorFinal.shape[2] > 1 else np.zeros_like(zku_profile)

            # 清理缺失值
            zku_profile = np.where(is_gpm_missing_value(zku_profile), np.nan, zku_profile)
            zka_profile = np.where(is_gpm_missing_value(zka_profile), np.nan, zka_profile)

            # 读取zFactorMeasured (4D: nscan, nray, nbin, 2) - 最大数组，用更小的chunk
            zFactorMeasured = read_chunks_4d(fs_group['PRE']['zFactorMeasured'], valid_scan_idx, valid_ray_idx, chunk_size=100)
            zku_meas_profile = zFactorMeasured[:, :, 0]
            zka_meas_profile = zFactorMeasured[:, :, 1] if zFactorMeasured.shape[2] > 1 else np.zeros_like(zku_meas_profile)

            # 清理缺失值
            zku_meas_profile = np.where(is_gpm_missing_value(zku_meas_profile), np.nan, zku_meas_profile)
            zka_meas_profile = np.where(is_gpm_missing_value(zka_meas_profile), np.nan, zka_meas_profile)

            # 释放大数组内存
            del zFactorFinal, zFactorMeasured
            gc.collect()

            # 近地面值 (zFactorFinalNS 形状: (n_valid, 2))
            zku_ns = zFactorFinalNS[:, 0]  # Ku波段
            zka_ns = zFactorFinalNS[:, 1]  # Ka波段
            zku_ns = np.where(is_gpm_missing_value(zku_ns), np.nan, zku_ns)
            zka_ns = np.where(is_gpm_missing_value(zka_ns), np.nan, zka_ns)
            dfr_ns = np.log10(np.clip(zku_ns, 1e-6, None)) - np.log10(np.clip(zka_ns, 1e-6, None))

            # 4-6: 衰减后近地面
            gpm_features[:, 4] = zku_ns
            gpm_features[:, 5] = zka_ns
            gpm_features[:, 6] = dfr_ns

            # 柱内最大
            zku_max = safe_nanmax(zku_profile, axis=1)
            zka_max = safe_nanmax(zka_profile, axis=1)
            zku_max_idx = safe_nanargmax(zku_profile, axis=1)
            zka_max_idx = safe_nanargmax(zka_profile, axis=1)

            dfr_profile = np.log10(np.clip(zku_profile, 1e-6, None)) - np.log10(np.clip(zka_profile, 1e-6, None))
            dfr_max = safe_nanmax(dfr_profile, axis=1)

            # 7-9: 衰减后柱内最大
            gpm_features[:, 7] = zku_max
            gpm_features[:, 8] = zka_max
            gpm_features[:, 9] = dfr_max

            # 计算最大值高度 - 分块处理
            zku_max_height = np.zeros(n_valid)
            zka_max_height = np.zeros(n_valid)
            height_dataset = fs_group['PRE']['height']

            for i in range(0, n_valid, 500):
                end = min(i + 500, n_valid)
                for j in range(i, end):
                    idx = j - i
                    if not np.isnan(zku_max_idx[j]):
                        bin_idx = int(zku_max_idx[j])
                        if bin_idx < nbin_actual:
                            s, r = valid_scan_idx[j], valid_ray_idx[j]
                            zku_max_height[j] = height_dataset[s, r, bin_idx]
                    if not np.isnan(zka_max_idx[j]):
                        bin_idx = int(zka_max_idx[j])
                        if bin_idx < nbin_actual:
                            s, r = valid_scan_idx[j], valid_ray_idx[j]
                            zka_max_height[j] = height_dataset[s, r, bin_idx]

            # 斜率计算
            slope_zku = np.zeros(n_valid)
            slope_zka = np.zeros(n_valid)
            for i in range(0, n_valid, 500):
                end = min(i + 500, n_valid)
                for j in range(i, end):
                    bin_idx = int(binClutterFreeBottom[j])
                    if 0 <= bin_idx < nbin_actual - 2:
                        zku_2km_idx = min(bin_idx + 20, nbin_actual - 1)
                        if not np.isnan(zku_ns[j]) and not np.isnan(zku_profile[j, zku_2km_idx]):
                            slope_zku[j] = (zku_profile[j, zku_2km_idx] - zku_ns[j]) / 2.0
                        if not np.isnan(zka_ns[j]) and not np.isnan(zka_profile[j, zku_2km_idx]):
                            slope_zka[j] = (zka_profile[j, zku_2km_idx] - zka_ns[j]) / 2.0
            slope_dfr = slope_zku - slope_zka

            # 10-12: 斜率
            gpm_features[:, 10] = slope_zku
            gpm_features[:, 11] = slope_zka
            gpm_features[:, 12] = slope_dfr

            # 13-15: 最大值高度
            gpm_features[:, 13] = zku_max_height
            gpm_features[:, 14] = zka_max_height
            dfr_max_idx = safe_nanargmax(dfr_profile, axis=1)

            dfr_max_height = np.zeros(n_valid)
            for i in range(0, n_valid, 500):
                end = min(i + 500, n_valid)
                for j in range(i, end):
                    if not np.isnan(dfr_max_idx[j]):
                        bin_idx = int(dfr_max_idx[j])
                        if bin_idx < nbin_actual:
                            s, r = valid_scan_idx[j], valid_ray_idx[j]
                            dfr_max_height[j] = height_dataset[s, r, bin_idx]
            gpm_features[:, 15] = dfr_max_height

      # 16-17: 冰相/液相层厚度
            ipl = np.maximum(heightStormTop - heightZeroDeg, 0)
            lpl = np.maximum(heightZeroDeg - free_bottom_height, 0)
            gpm_features[:, 16] = ipl
            gpm_features[:, 17] = lpl

            # 释放profile内存
            del zku_profile, zka_profile, dfr_profile
            gc.collect()

      # 18-20: 衰减前近地面
            zku_ns_meas = zku_meas_profile[:, 0] if zku_meas_profile.ndim == 2 else zku_meas_profile
            zka_ns_meas = zka_meas_profile[:, 1] if zka_meas_profile.ndim == 2 else np.zeros_like(zku_ns_meas)
            zku_ns_meas = np.where(is_gpm_missing_value(zku_ns_meas), np.nan, zku_ns_meas)
            zka_ns_meas = np.where(is_gpm_missing_value(zka_ns_meas), np.nan, zka_ns_meas)
            dfr_ns_meas = np.log10(np.clip(zku_ns_meas, 1e-6, None)) - np.log10(np.clip(zka_ns_meas, 1e-6, None))

            gpm_features[:, 18] = zku_ns_meas
            gpm_features[:, 19] = zka_ns_meas
            gpm_features[:, 20] = dfr_ns_meas

            # 21-23: 衰减前柱内最大
            zku_max_meas = safe_nanmax(zku_meas_profile, axis=1)
            zka_max_meas = safe_nanmax(zka_meas_profile, axis=1)
            dfr_meas_profile = np.log10(np.clip(zku_meas_profile, 1e-6, None)) - np.log10(np.clip(zka_meas_profile, 1e-6, None))
            dfr_max_meas = safe_nanmax(dfr_meas_profile, axis=1)
            gpm_features[:, 21] = zku_max_meas
            gpm_features[:, 22] = zka_max_meas
            gpm_features[:, 23] = dfr_max_meas

            # 24-26: 衰减前最大值高度
            zku_max_meas_idx = safe_nanargmax(zku_meas_profile, axis=1)
            zka_max_meas_idx = safe_nanargmax(zka_meas_profile, axis=1)
            dfr_max_meas_idx = safe_nanargmax(dfr_meas_profile, axis=1)

            zku_max_meas_height = np.zeros(n_valid)
            zka_max_meas_height = np.zeros(n_valid)
            dfr_max_meas_height = np.zeros(n_valid)

            for i in range(0, n_valid, 500):
                end = min(i + 500, n_valid)
                for j in range(i, end):
                    if not np.isnan(zku_max_meas_idx[j]):
                        bin_idx = int(zku_max_meas_idx[j])
                        if bin_idx < nbin_actual:
                            s, r = valid_scan_idx[j], valid_ray_idx[j]
                            zku_max_meas_height[j] = height_dataset[s, r, bin_idx]
                    if not np.isnan(zka_max_meas_idx[j]):
                        bin_idx = int(zka_max_meas_idx[j])
                        if bin_idx < nbin_actual:
                            s, r = valid_scan_idx[j], valid_ray_idx[j]
                            zka_max_meas_height[j] = height_dataset[s, r, bin_idx]
                    if not np.isnan(dfr_max_meas_idx[j]):
                        bin_idx = int(dfr_max_meas_idx[j])
                        if bin_idx < nbin_actual:
                            s, r = valid_scan_idx[j], valid_ray_idx[j]
                            dfr_max_meas_height[j] = height_dataset[s, r, bin_idx]

            gpm_features[:, 24] = zku_max_meas_height
            gpm_features[:, 25] = zka_max_meas_height
            gpm_features[:, 26] = dfr_max_meas_height

            # 释放剩余profile内存
            del zku_meas_profile, zka_meas_profile, dfr_meas_profile
            gc.collect()

            # 27-29: 衰减前斜率
            slope_zku_meas = np.zeros(n_valid)
            slope_zka_meas = np.zeros(n_valid)
            for j in range(n_valid):
                if not np.isnan(zku_ns_meas[j]) and not np.isnan(zku_max_meas[j]):
                    slope_zku_meas[j] = (zku_max_meas[j] - zku_ns_meas[j]) / 2.0
                if not np.isnan(zka_ns_meas[j]) and not np.isnan(zka_max_meas[j]):
                    slope_zka_meas[j] = (zka_max_meas[j] - zka_ns_meas[j]) / 2.0
            slope_dfr_meas = slope_zku_meas - slope_zka_meas
            gpm_features[:, 27] = slope_zku_meas
            gpm_features[:, 28] = slope_zka_meas
            gpm_features[:, 29] = slope_dfr_meas

            # 30-31: 预留
            gpm_features[:, 30:32] = np.nan

            # 读取CSF组变量（某些文件可能缺失）
            try:
                csf_group = fs_group['CSF']
                binBBTop = read_chunks_2d(csf_group['binBBTop'], valid_scan_idx, valid_ray_idx)
                binBBBottom = read_chunks_2d(csf_group['binBBBottom'], valid_scan_idx, valid_ray_idx)
                heightBB = read_chunks_2d(csf_group['heightBB'], valid_scan_idx, valid_ray_idx)

                # 32-33: 融化层参数
                heightBB_km = heightBB / 1000.0
                bb_thickness = (binBBTop - binBBBottom) * 0.125
                gpm_features[:, 32] = heightBB_km
                gpm_features[:, 33] = bb_thickness
            except (KeyError, ValueError) as e:
                if logger:
                    logger.warning(f"  CSF组变量读取失败，使用NaN填充: {e}")
                gpm_features[:, 32] = np.nan
                gpm_features[:, 33] = np.nan

            # 34-37: 雨滴谱参数 - paramDSD是4D数组 (nscan, nray, nbin, 2)
            paramDSD_dataset = fs_group['SLV']['paramDSD']
            dm_2_5km = np.zeros(n_valid)
            nw_2_5km = np.zeros(n_valid)
            dm_column_max = np.zeros(n_valid)
            nw_column_max = np.zeros(n_valid)

            # paramDSD 形状: (nscan, nray, nbin, 2)，第三维是nbin，第四维是2(Nw, Dm)
            nbin_dsd = paramDSD_dataset.shape[2]
            for i in range(0, n_valid, 300):
                end = min(i + 300, n_valid)
                s_idx = valid_scan_idx[i:end]
                r_idx = valid_ray_idx[i:end]
                for j, (s, r) in enumerate(zip(s_idx, r_idx)):
                    idx = i + j
                    bin_2_5km = min(20, nbin_dsd - 1)
                    # 读取该像元的完整DSD数据 (nbin, 2)
                    dsd_data = paramDSD_dataset[s, r, :, :]  # 形状: (nbin, 2)
                    nw_2_5km[idx] = dsd_data[bin_2_5km, 0]  # Nw at 2.5km
                    dm_2_5km[idx] = dsd_data[bin_2_5km, 1]  # Dm at 2.5km
                    nw_column_max[idx] = np.nanmax(dsd_data[:, 0]) if not np.all(np.isnan(dsd_data[:, 0])) else np.nan
                    dm_column_max[idx] = np.nanmax(dsd_data[:, 1]) if not np.all(np.isnan(dsd_data[:, 1])) else np.nan

            gpm_features[:, 34] = dm_2_5km
            gpm_features[:, 35] = nw_2_5km
            gpm_features[:, 36] = dm_column_max
            gpm_features[:, 37] = nw_column_max

            # 38-46: 预留
            gpm_features[:, 38:47] = np.nan

            # 清理缺失值
            gpm_features = np.where(is_gpm_missing_value(gpm_features), np.nan, gpm_features)
            gpm_features = np.where(np.isinf(gpm_features), np.nan, gpm_features)

            # 强制垃圾回收
            gc.collect()

            return {
                'latitude': lat,
                'longitude': lon,
                'precip_rate': precip_rate,
                'valid_mask': valid_mask,
                'gpm_features': gpm_features,
                'n_valid': n_valid,
            }

    except MemoryError as e:
        if logger:
            logger.error(f"  内存不足，无法加载GPM数据: {e}")
        return None
    except Exception as e:
        if logger:
            logger.error(f"  加载GPM数据异常: {e}")
        return None


# 使用新的内存优化版本
load_gpm_data = load_gpm_data_memory_efficient

def find_era5_pressure_file(year, month, day, config, logger=None):
    """
    查找ERA5压力层文件
    2014-2023年: 十天或十一天一个nc文件，命名格式如 2019-04-01-04-10...nc
    2024年: 每天一个nc文件
    """
    import glob
    import re
    import calendar

    era5_dir = get_era5_pressure_dir(year, config)
    if not os.path.exists(era5_dir):
        raise FileNotFoundError(f"ERA5压力层目录不存在: {era5_dir}")

    all_files = glob.glob(os.path.join(era5_dir, "*.nc"))
    if not all_files:
        raise FileNotFoundError(f"ERA5压力层目录中没有nc文件: {era5_dir}")

    if year == 2024:
        # 2024年: 每天一个文件，有两种文件名格式
        # 格式1 (4-6月): "pressure level_YYYYMMDD.nc" (如 pressure level_20240512.nc)
        # 格式2 (7-9月): "2024-MM-DD...nc" (如 2024-07-01...nc)

        # 首先尝试格式1: YYYYMMDD
        date_str = f"{year}{month:02d}{day:02d}"
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            if date_str in file_name:
                if logger:
                    logger.info(f"  找到ERA5压力层文件(2024每日,格式1): {file_name}")
                return file_path, day

        # 如果格式1没找到，尝试格式2: YYYY-MM-DD
        pattern = f"{year}-{month:02d}-{day:02d}"
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            if pattern in file_name:
                if logger:
                    logger.info(f"  找到ERA5压力层文件(2024每日,格式2): {file_name}")
                return file_path, day

        raise FileNotFoundError(f"找不到2024年ERA5压力层文件: {date_str} 或 {pattern}")
    else:
        # 2014-2023年: 十天文件，根据日期范围匹配
        # 计算该日期所在的十天区间
        day_start = ((day - 1) // 10) * 10 + 1
        _, max_day_in_month = calendar.monthrange(year, month)
        day_end = min(day_start + 9, max_day_in_month)

        # 文件名格式: 2019-04-01-04-10...nc 或 2019-07-21-07-31...nc
        # 查找包含该日期范围的文件
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            # 匹配年月和日期范围格式
            date_pattern = rf'{year}-{month:02d}-(\d{{2}})-{month:02d}-(\d{{2}})'
            match = re.search(date_pattern, file_name)
            if match:
                file_day_start = int(match.group(1))
                file_day_end = int(match.group(2))
                # 检查目标日期是否在文件覆盖范围内
                if file_day_start <= day <= file_day_end:
                    if logger:
                        logger.info(f"  找到ERA5压力层文件(十天): {file_name} ({file_day_start}-{file_day_end}日)")
                    return file_path, file_day_start

        raise FileNotFoundError(
            f"找不到ERA5压力层文件: {year}-{month:02d}-{day:02d} "
            f"(预期日期范围: {day_start:02d}-{day_end:02d})"
        )
def find_era5_single_file(year, month, config, logger=None):
    """
    查找ERA5单层文件
    2014-2023年: 每月一个nc文件，命名格式如 2019-04...nc
    2024年: 每月三个nc文件 (accum, avg, instant)，使用instant文件
    """
    import glob
    import re

    era5_dir = get_era5_single_dir(year, config)
    if not os.path.exists(era5_dir):
        raise FileNotFoundError(f"ERA5单层目录不存在: {era5_dir}")

    all_files = glob.glob(os.path.join(era5_dir, "*.nc"))
    if not all_files:
        raise FileNotFoundError(f"ERA5单层目录中没有nc文件: {era5_dir}")

    if year == 2024:
        # 2024年: 每月三个文件 (accum, avg, instant)
        # 使用instant文件（即时数据，适合匹配GPM瞬时观测）
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            # 匹配年月和instant类型
            if (file_name.startswith(f"{year}-{month:02d}") and
                "instant" in file_name.lower()):
                if logger:
                    logger.info(f"  找到ERA5单层文件(2024 instant): {file_name}")
                return file_path
        # 如果没有找到instant文件，尝试avg文件
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            if (file_name.startswith(f"{year}-{month:02d}") and
                "avg" in file_name.lower()):
                if logger:
                    logger.info(f"  找到ERA5单层文件(2024 avg): {file_name}")
                return file_path
        # 如果都没有找到，返回第一个匹配年月的文件
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            if file_name.startswith(f"{year}-{month:02d}"):
                if logger:
                    logger.info(f"  找到ERA5单层文件(2024): {file_name}")
                return file_path
        raise FileNotFoundError(f"找不到2024年ERA5单层文件: {year}-{month:02d}")
    else:
        # 2014-2023年: 每月一个文件
        # 文件名格式: 2019-04...nc
        for file_path in all_files:
            file_name = os.path.basename(file_path)
            # 匹配年月
            if re.match(rf'{year}-{month:02d}\D', file_name):
                if logger:
                    logger.info(f"  找到ERA5单层文件(每月): {file_name}")
                return file_path
        raise FileNotFoundError(f"找不到ERA5单层文件: {year}-{month:02d}")
def load_era5_pressure_for_gpm(year, month, day, hour, config, logger=None):
    """加载ERA5压力层数据对应GPM时间"""
    # 查找ERA5压力层文件
    era5_file, day_start_number = find_era5_pressure_file(year, month, day, config, logger)
    if not os.path.exists(era5_file):
        raise FileNotFoundError(f"ERA5压力层文件不存在: {era5_file}")
    if logger:
        logger.info(f"  ERA5压力层文件: {os.path.basename(era5_file)}")
    # 使用xarray加载数据
    ds = xr.open_dataset(era5_file)
    # 空间裁剪
    lat_range = config["data_range"]["lat_range"]
    lon_range = config["data_range"]["lon_range"]
    ds_slice = ds.sel(
        longitude=slice(lon_range[0], lon_range[1]),
        latitude=slice(lat_range[1], lat_range[0])
    )
    # 14个压力层变量
    var_mapping = {
        'z': 'z', 'o3': 'o3', 'pv': 'pv', 'r': 'r',
        'ciwc': 'ciwc', 'clwc': 'clwc', 'q': 'q', 'crwc': 'crwc',
        'cswc': 'cswc', 't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'vo': 'vo',
    }
    result = {
        'lon': ds_slice.longitude.values,
        'lat': ds_slice.latitude.values,
        'level': ds_slice.level.values if 'level' in ds_slice else np.arange(27),
        'day_start_number': day_start_number,
    }
    n_lon = len(result['lon'])
    n_lat = len(result['lat'])
    n_level = len(result['level'])
    time_size = ds_slice.dims.get('time', 240)
    for key, var_name in var_mapping.items():
        if var_name in ds_slice:
            data = ds_slice[var_name].values
            if data.ndim == 4:
                data = np.transpose(data, (3, 2, 1, 0))
            result[key] = data
        else:
            result[key] = np.full((n_lon, n_lat, n_level, time_size), np.nan)
    ds.close()
    return result
def load_era5_single_for_gpm(year, month, day, hour, config, logger=None):
    """加载ERA5单层数据对应GPM时间"""
    era5_file = find_era5_single_file(year, month, config, logger)
    if not os.path.exists(era5_file):
        raise FileNotFoundError(f"ERA5单层文件不存在: {era5_file}")
    ds = xr.open_dataset(era5_file)
    lat_range = config["data_range"]["lat_range"]
    lon_range = config["data_range"]["lon_range"]
    ds_slice = ds.sel(
        longitude=slice(lon_range[0], lon_range[1]),
        latitude=slice(lat_range[1], lat_range[0])
    )
    n_lon = len(ds_slice.longitude.values)
    n_lat = len(ds_slice.latitude.values)
    time_size = ds_slice.dims.get('time', 720)
    var_mapping = {
        'u10': 'u10', 'v10': 'v10', 'd2m': 'd2m', 't2m': 't2m', 'sp': 'sp',
        'cbh': 'cbh', 'tcc': 'tcc', 'tciw': 'tciw', 'tclw': 'tclw', 'crr': 'crr',
        'ptype': 'ptype', 'tcrw': 'tcrw', 'blh': 'blh', 'cape': 'cape', 'cin': 'cin',
        'z_surf': 'z', 'kx': 'kx', 'tcw': 'tcw', 'tcwv': 'tcwv', 'deg0l': 'deg0l',
        'rh850': 'rh850', 'tp': 'tp',
    }
    result = {
        'lon': ds_slice.longitude.values,
        'lat': ds_slice.latitude.values,
    }
    for key, var_name in var_mapping.items():
        if var_name in ds_slice:
            data = ds_slice[var_name].values
            if data.ndim == 3:
                data = np.transpose(data, (2, 1, 0))
            result[key] = data
        else:
            result[key] = np.full((n_lon, n_lat, time_size), np.nan)
    ds.close()
    return result
def process_single_gpm_file(gpm_file, gpm_dir, config, logger):
    """
    处理单个GPM文件：加载GPM数据，匹配ERA5，返回匹配结果
    改进：更好的错误处理和内存管理
    返回: (result_dict, error_type) 元组，成功时error_type为None
    """
    gpm_file_path = os.path.join(gpm_dir, gpm_file)
    logger.info(f"处理GPM文件: {gpm_file}")
    # 1. 提取时间信息
    try:
        year, month, day, hour, day_unique, hour_unique = get_time_info_from_gpm(gpm_file_path)
        logger.info(f"  时间: {year}-{month:02d}-{day:02d} {hour:02d}:00, DayUnique={day_unique}, HourUnique={hour_unique}")
    except Exception as e:
        logger.error(f"  无法提取时间信息: {e}")
        return None, 'time_extraction'
    # 2. 加载GPM数据（内存优化版）
    try:
        lat_range = config["data_range"]["lat_range"]
        lon_range = config["data_range"]["lon_range"]
        min_rain_rate = config.get("spatial_matching", {}).get("quality_control", {}).get("min_rain_rate", 0.5)
        gpm_data = load_gpm_data(gpm_file_path, lat_range, lon_range, min_rain_rate, logger)
        if gpm_data is None:
            logger.warning(f"  GPM数据加载返回None（研究区域内无有效降水），跳过文件")
            return None, 'gpm_no_precipitation'
        valid_mask = gpm_data['valid_mask']
        n_valid = gpm_data['n_valid']
        if n_valid == 0:
            logger.warning(f"  没有有效像元（空间范围内有数据但无有效降水），跳过")
            return None, 'gpm_no_valid_pixels'
    except MemoryError as e:
        logger.error(f"  加载GPM数据内存不足: {e}")
        return None, 'gpm_load_memory'
    except Exception as e:
        logger.error(f"  加载GPM数据失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 'gpm_load_other'
    # 3. 加载ERA5压力层数据
    try:
        era5_pressure = load_era5_pressure_for_gpm(year, month, day, hour, config, logger)
        logger.info(f"  ERA5压力层: shape={era5_pressure['r'].shape}")
    except FileNotFoundError as e:
        logger.error(f"  找不到ERA5压力层文件: {e}")
        return None, 'era5_pressure_not_found'
    except Exception as e:
        logger.error(f"  加载ERA5压力层失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 'era5_pressure_load'
    # 4. 加载ERA5单层数据
    try:
        era5_single = load_era5_single_for_gpm(year, month, day, hour, config, logger)
        logger.info(f"  ERA5单层: shape={era5_single['cape'].shape}")
    except FileNotFoundError as e:
        logger.error(f"  找不到ERA5单层文件: {e}")
        return None, 'era5_single_not_found'
    except Exception as e:
        logger.error(f"  加载ERA5单层失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 'era5_single_load'
    # 5. 执行最近邻匹配 - 压力层
    try:
        day_start_number_era5 = era5_pressure['day_start_number']
        logger.info(f"  开始压力层匹配: DayUnique={day_unique}, HourUnique={hour_unique}, ERA5起始日={day_start_number_era5}")
        Z, O3, PV, R, CIWC, CLWC, Q, CRWC, CSWC, T, U, V_var, W, VO = near_match_era5_pressure_gpm_full(
            latitude=gpm_data['latitude'],
            longitude=gpm_data['longitude'],
            day_unique=day_unique,
            hour_unique=hour_unique,
            day_start_number_era5_pressure=day_start_number_era5,
            era5_lon=era5_pressure['lon'],
            era5_lat=era5_pressure['lat'],
            era5_level=era5_pressure['level'],
            era5_z=era5_pressure['z'],
            era5_o3=era5_pressure['o3'],
            era5_pv=era5_pressure['pv'],
            era5_r=era5_pressure['r'],
            era5_ciwc=era5_pressure['ciwc'],
            era5_clwc=era5_pressure['clwc'],
            era5_q=era5_pressure['q'],
            era5_crwc=era5_pressure['crwc'],
            era5_cswc=era5_pressure['cswc'],
            era5_t=era5_pressure['t'],
            era5_u=era5_pressure['u'],
            era5_v=era5_pressure['v'],
            era5_w=era5_pressure['w'],
            era5_vo=era5_pressure['vo'],
        )
        logger.info(f"  压力层匹配完成: 14变量, Z.shape={Z.shape}")
    except Exception as e:
        logger.error(f"  压力层匹配失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 'pressure_match'
    # 6. 执行最近邻匹配 - 单层
    logger.info(f"  开始单层匹配...")
    try:
        (U10, V10, D2M, T2M, SP, CBH, TCC, TCIW_s, TCLW_s, CRR, PTYPE, TCRW,
         BLH, CAPE, CIN, Z_SURF, KX, TCW, TCWV, DEG0L, RH850, TP) = near_match_era5_single_gpm_full(
            latitude=gpm_data['latitude'],
            longitude=gpm_data['longitude'],
            day_unique=day_unique,
            hour_unique=hour_unique,
            era5_lon=era5_single['lon'],
            era5_lat=era5_single['lat'],
            era5_u10=era5_single['u10'],
            era5_v10=era5_single['v10'],
            era5_d2m=era5_single['d2m'],
            era5_t2m=era5_single['t2m'],
            era5_sp=era5_single['sp'],
            era5_cbh=era5_single['cbh'],
            era5_tcc=era5_single['tcc'],
            era5_tciw=era5_single['tciw'],
            era5_tclw_s=era5_single['tclw'],
            era5_crr=era5_single['crr'],
            era5_ptype=era5_single['ptype'],
            era5_tcrw=era5_single['tcrw'],
            era5_blh=era5_single['blh'],
            era5_cape=era5_single['cape'],
            era5_cin=era5_single['cin'],
            era5_z_surf=era5_single['z_surf'],
            era5_kx=era5_single['kx'],
            era5_tcw=era5_single['tcw'],
            era5_tcwv=era5_single['tcwv'],
            era5_deg0l=era5_single['deg0l'],
            era5_rh850=era5_single['rh850'],
            era5_tp=era5_single['tp'],
        )
        logger.info(f"  单层匹配完成: 22变量, U10.shape={U10.shape}")
    except Exception as e:
        logger.error(f"  单层匹配失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, 'single_match'
    # 7. 构建完整的447维matched_raw
    p, q = gpm_data['latitude'].shape
    n_pressure_levels = 27
    n_pressure_vars = 14
    n_single_vars = 22
    matched_raw = np.zeros((p, q, 447))
    # 填充GPM特征
    valid_indices = np.where(valid_mask)
    gpm_features = gpm_data['gpm_features']
    matched_raw[valid_indices[0], valid_indices[1], 0:47] = gpm_features
    # 填充压力层数据
    pressure_vars = [Z, O3, PV, R, CIWC, CLWC, Q, CRWC, CSWC, T, U, V_var, W, VO]
    for level_idx in range(n_pressure_levels):
        base_idx = 47 + level_idx * n_pressure_vars
        for var_idx, var_data in enumerate(pressure_vars):
            matched_raw[:, :, base_idx + var_idx] = var_data[:, :, level_idx]
    # 填充单层数据
    single_vars = [U10, V10, D2M, T2M, SP, CBH, TCC, TCIW_s, TCLW_s, CRR, PTYPE, TCRW,
                   BLH, CAPE, CIN, Z_SURF, KX, TCW, TCWV, DEG0L, RH850, TP]
    base_idx = 47 + n_pressure_levels * n_pressure_vars
    for var_idx, var_data in enumerate(single_vars):
        matched_raw[:, :, base_idx + var_idx] = var_data
    logger.info(f"  构建matched_raw完成: shape={matched_raw.shape}, 447维")
    # 创建时间戳数组
    timestamp = np.full((p, q), f"{year}{month:02d}{day:02d}{hour:02d}", dtype='U10')
    # 释放中间变量内存
    del Z, O3, PV, R, CIWC, CLWC, Q, CRWC, CSWC, T, U, V_var, W, VO
    del U10, V10, D2M, T2M, SP, CBH, TCC, TCIW_s, TCLW_s, CRR, PTYPE, TCRW
    del BLH, CAPE, CIN, Z_SURF, KX, TCW, TCWV, DEG0L, RH850, TP
    gc.collect()
    return {
        'matched_raw': matched_raw,
        'lat': gpm_data['latitude'],
        'lon': gpm_data['longitude'],
        'valid_mask': valid_mask,
        'n_valid': n_valid,
        'year': year,
        'month': month,
        'day': day,
        'hour': hour,
        'timestamp': timestamp,
    }, None
def save_monthly_results_incremental(year, month, results, output_dir, logger, config):
    """
    按月增量保存匹配结果
    改进：直接追加到月度文件，不保留所有结果在内存中
    """
    from pathlib import Path
    create_dir(output_dir)
    if not results:
        logger.warning(f"{year}-{month:02d} 没有有效结果，跳过保存")
        return
    output_file = os.path.join(output_dir, f"matched_nearest_{year}_{month:02d}.npz")
    # 构造数据字典
    data_dict = {}
    block_idx = 0
    total_samples = 0
    for r in results:
        # 展平空间维度
        p, q = r['lat'].shape
        flat_matched_raw = r['matched_raw'].reshape(-1, r['matched_raw'].shape[-1])
        flat_lat = r['lat'].ravel()
        flat_lon = r['lon'].ravel()
        flat_timestamp = r['timestamp'].ravel()
        flat_valid_mask = r['valid_mask'].ravel()
        # 只保留有效像元
        valid_indices = np.where(flat_valid_mask)[0]
        if len(valid_indices) > 0:
            data_dict[f'block_{block_idx}'] = flat_matched_raw[valid_indices]
            data_dict[f'lat_block_{block_idx}'] = flat_lat[valid_indices]
            data_dict[f'lon_block_{block_idx}'] = flat_lon[valid_indices]
            data_dict[f'timestamp_block_{block_idx}'] = flat_timestamp[valid_indices]
            data_dict[f'meta_block_{block_idx}'] = np.array([
                block_idx,
                len(valid_indices),
                r['year'],
                r['month']
            ], dtype=np.int32)
            total_samples += len(valid_indices)
            block_idx += 1
    if block_idx == 0:
        logger.warning(f"{year}-{month:02d} 没有有效数据块，跳过保存")
        return
    # 添加全局元数据
    data_dict['meta_year'] = year
    data_dict['meta_month'] = month
    data_dict['meta_n_blocks'] = block_idx
    data_dict['meta_total_samples'] = total_samples
    # 追加模式保存
    if os.path.exists(output_file):
        try:
            existing_data = np.load(output_file, allow_pickle=True)
            existing_data = dict(existing_data)
            # 找出已有最大block编号
            max_block_num = -1
            for key in existing_data.keys():
                if key.startswith('block_') and key[6:].isdigit():
                    block_num = int(key[6:])
                    max_block_num = max(max_block_num, block_num)
            next_block_num = max_block_num + 1
            new_data_dict = {}
            n_new_blocks = 0
            for key, value in data_dict.items():
                if key.startswith('block_') and key[6:].isdigit():
                    old_block_num = int(key[6:])
                    new_block_num = next_block_num + old_block_num
                    new_key = f'block_{new_block_num}'
                    new_data_dict[new_key] = value
                    n_new_blocks += 1
                    # 同步更新相关数组
                    old_lat_key = f'lat_block_{old_block_num}'
                    old_lon_key = f'lon_block_{old_block_num}'
                    old_ts_key = f'timestamp_block_{old_block_num}'
                    old_meta_key = f'meta_block_{old_block_num}'
                    if old_lat_key in data_dict:
                        new_data_dict[f'lat_block_{new_block_num}'] = data_dict[old_lat_key]
                    if old_lon_key in data_dict:
                        new_data_dict[f'lon_block_{new_block_num}'] = data_dict[old_lon_key]
                    if old_ts_key in data_dict:
                        new_data_dict[f'timestamp_block_{new_block_num}'] = data_dict[old_ts_key]
                    if old_meta_key in data_dict:
                        new_data_dict[f'meta_block_{new_block_num}'] = data_dict[old_meta_key]
                elif key.startswith('meta_') and key not in ('meta_year', 'meta_month'):
                    pass
                elif key not in ('meta_year', 'meta_month'):
                    new_data_dict[key] = value
            # 合并数据
            existing_data.update(new_data_dict)
            # 更新全局元数据
            total_blocks = len([k for k in existing_data.keys() if k.startswith('block_')])
            existing_data['meta_n_blocks'] = total_blocks
            existing_data['meta_year'] = year
            existing_data['meta_month'] = month
            if 'meta_total_samples' not in existing_data:
                existing_data['meta_total_samples'] = 0
            existing_data['meta_total_samples'] += data_dict.get('meta_total_samples', 0)
            np.savez_compressed(output_file, **existing_data)
            logger.info(f"[追加模式] 保存 {year}-{month:02d} 结果: {output_file} (新增 {n_new_blocks} 个块, 总计 {total_blocks} 个块, {existing_data['meta_total_samples']} 样本)")
        except Exception as e:
            logger.error(f"追加保存失败，覆盖写入: {e}")
            np.savez_compressed(output_file, **data_dict)
            logger.info(f"[覆盖模式] 保存 {year}-{month:02d} 结果: {output_file} ({block_idx} 个数据块)")
    else:
        np.savez_compressed(output_file, **data_dict)
        logger.info(f"[新建] 保存 {year}-{month:02d} 结果: {output_file} ({block_idx} 个数据块, {total_samples} 样本)")

def gpu_worker_monthly(gpu_id, month_key, gpm_files_with_dirs, config, output_dir):
    """
    GPU工作进程 - 按月处理
    改进：处理完一个月后立即保存，不保留所有结果在内存中
    添加：详细的错误分类统计，帮助诊断问题
    """
    logger = setup_logging(gpu_id, month_key)
    logger.info(f"GPU {gpu_id} 开始处理 {month_key}")
    # 设置CUDA设备
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    processed_count = 0
    error_count = 0
    skip_no_valid_pixels = 0
    skip_no_precipitation = 0
    # 详细的错误统计
    error_types = {
        'time_extraction': 0,
        'gpm_load_memory': 0,
        'gpm_load_other': 0,
        'gpm_no_valid_pixels': 0,  # 有数据但空间范围内无有效像元
        'gpm_no_precipitation': 0,   # 空间范围内无降水
        'era5_pressure_not_found': 0,
        'era5_pressure_load': 0,
        'era5_single_not_found': 0,
        'era5_single_load': 0,
        'pressure_match': 0,
        'single_match': 0,
    }
    # 记录第一个成功的文件信息，用于调试
    first_success_file = None
    first_error_file = None
    sample_errors = []  # 记录前几个错误详情
    monthly_results = []
    for idx, (gpm_file, gpm_dir) in enumerate(gpm_files_with_dirs):
        logger.info(f"[{idx+1}/{len(gpm_files_with_dirs)}] 处理 {gpm_file}")
        result, error_type = process_single_gpm_file(gpm_file, gpm_dir, config, logger)
        if result is not None:
            monthly_results.append(result)
            processed_count += 1
            if first_success_file is None:
                first_success_file = gpm_file
                logger.info(f"  ✓ 第一个成功处理的文件: {gpm_file}, 有效像元数: {result['n_valid']}")
        else:
            error_count += 1
            if error_type:
                error_types[error_type] = error_types.get(error_type, 0) + 1
            if first_error_file is None:
                first_error_file = gpm_file
                logger.warning(f"  ✗ 第一个失败的文件: {gpm_file}, 错误类型: {error_type}")
        # 每处理10个文件或处理完所有文件时，保存一次结果
        if len(monthly_results) >= 10 or idx == len(gpm_files_with_dirs) - 1:
            if monthly_results:
                year = monthly_results[0]['year']
                month = monthly_results[0]['month']
                save_monthly_results_incremental(year, month, monthly_results, output_dir, logger, config)
                # 清空已保存的结果
                monthly_results = []
                # 强制垃圾回收
                gc.collect()
                torch.cuda.empty_cache()
    # 详细的统计报告
    logger.info("=" * 60)
    logger.info(f"GPU {gpu_id} 处理 {month_key} 统计报告:")
    logger.info(f"  总文件数: {len(gpm_files_with_dirs)}")
    logger.info(f"  成功处理: {processed_count}")
    logger.info(f"  失败/跳过: {error_count}")
    logger.info(f"  成功率: {processed_count / len(gpm_files_with_dirs) * 100:.1f}%")
    if first_success_file:
        logger.info(f"  首个成功文件: {first_success_file}")
    if first_error_file:
        logger.info(f"  首个失败文件: {first_error_file}")
    logger.info("  错误分类统计:")
    for error_type, count in error_types.items():
        if count > 0:
            logger.info(f"    - {error_type}: {count}")
    logger.info("=" * 60)
    return {
        'gpu_id': gpu_id,
        'month_key': month_key,
        'processed': processed_count,
        'failed': error_count,
        'total': len(gpm_files_with_dirs),
        'error_breakdown': error_types,
    }
def gpu_worker_with_queue(gpu_id, month_key, gpm_files_with_dirs, config, result_queue):
    """
    GPU工作进程 - 使用队列实时传递结果给主进程
    改进：处理完一个文件就通过队列发送结果，不保留所有结果在内存中
    """
    logger = setup_logging(gpu_id, month_key)
    logger.info(f"GPU {gpu_id} 开始处理 {month_key}，共 {len(gpm_files_with_dirs)} 个文件")
    # 设置CUDA设备
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    processed_count = 0
    error_count = 0
    # 错误类型统计
    error_types = {
        'time_extraction': 0,
        'gpm_load_memory': 0,
        'gpm_load_other': 0,
        'gpm_no_valid_pixels': 0,
        'gpm_no_precipitation': 0,
        'era5_pressure_not_found': 0,
        'era5_pressure_load': 0,
        'era5_single_not_found': 0,
        'era5_single_load': 0,
        'pressure_match': 0,
        'single_match': 0,
    }
    try:
        for idx, (gpm_file, gpm_dir) in enumerate(gpm_files_with_dirs):
            logger.info(f"[GPU{gpu_id} {idx+1}/{len(gpm_files_with_dirs)}] 开始处理 {gpm_file}")
            result, error_type = process_single_gpm_file(gpm_file, gpm_dir, config, logger)
            if result is not None:
                try:
                    result_queue.put({
                        'type': 'result',
                        'data': result,
                        'gpu_id': gpu_id
                    })
                    processed_count += 1
                    logger.info(
                        f"[GPU{gpu_id} {idx+1}/{len(gpm_files_with_dirs)}] ✓ 完成: {gpm_file} "
                        f"(有效像元: {result['n_valid']})"
                    )
                except Exception as e:
                    logger.error(
                        f"[GPU{gpu_id} {idx+1}/{len(gpm_files_with_dirs)}] ✗ 发送结果到队列失败: {e}"
                    )
            else:
                error_count += 1
                if error_type:
                    error_types[error_type] = error_types.get(error_type, 0) + 1
                logger.warning(
                    f"[GPU{gpu_id} {idx+1}/{len(gpm_files_with_dirs)}] ✗ 失败: {gpm_file} (错误: {error_type})"
                )
            if (idx + 1) % 10 == 0:
                logger.info(
                    f"[GPU{gpu_id}] 进度: {idx+1}/{len(gpm_files_with_dirs)} 完成, "
                    f"成功: {processed_count}, 失败: {error_count}"
                )
            if idx % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    except Exception as e:
        logger.exception(f"GPU {gpu_id} 处理 {month_key} 时发生未捕获异常: {e}")
    finally:
        try:
            result_queue.put({'type': 'error_count', 'count': error_count})
        except Exception:
            pass
        try:
            result_queue.put(None)
        except Exception:
            pass
        logger.info(f"GPU {gpu_id} 处理 {month_key} 完成: 成功 {processed_count}, 失败 {error_count}")
def main():
    """主函数 - 按月增量处理"""
    parser = argparse.ArgumentParser(description='GPM-ERA5最近邻匹配 - 按月增量保存版本')
    parser.add_argument('--years', type=int, nargs='+', default=None,
                        help='指定运行的年份，如: --years 2014')
    parser.add_argument('--months', type=int, nargs='+', default=None,
                        help='指定运行的月份，如: --months 8 9')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='指定输出目录')
    parser.add_argument('--test', action='store_true',
                        help='测试模式：只处理前5个文件')
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='指定使用的GPU，如: --gpus 0 1')
    args = parser.parse_args()
    start_time = time.time()
    # 加载配置
    config = load_config()
    logger = setup_logging()
    # 覆盖配置
    if args.years:
        config["data_range"]["years"] = args.years
    if args.months:
        config["data_range"]["months"] = args.months
    # 确定输出目录
    if args.output_dir:
        output_dir = os.path.join(args.output_dir, "results", "matched_nearest")
    else:
        output_dir = os.path.join(config["project_root"], "results", "matched_nearest")
    logger.info("=" * 60)
    logger.info("GPM-ERA5 最近邻匹配 - 按月增量保存版本 (V2)")
    logger.info(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"运行年份: {config['data_range']['years']}")
    logger.info(f"运行月份: {config['data_range']['months']}")
    logger.info("=" * 60)
    # 确定GPU
    available_gpus = torch.cuda.device_count()
    if args.gpus:
        gpu_ids = [g for g in args.gpus if g < available_gpus]
    else:
        gpu_ids = list(range(min(config["gpu"]["num_gpus"], available_gpus)))
    if not gpu_ids:
        logger.error("没有可用的GPU，退出")
        return
    logger.info(f"使用GPU: {gpu_ids}")
    # 准备文件列表（按年月分组）
    years = config["data_range"].get("years", [2024])
    months = config["data_range"]["months"]
    monthly_files = {}
    for year in years:
        gpm_dir = get_gpm_root_path(year, config)
        if not os.path.exists(gpm_dir):
            logger.warning(f"GPM目录不存在: {gpm_dir}")
            continue
        import re
        for f in os.listdir(gpm_dir):
            if f.lower().endswith(('.hdf5', '.hdf')):
                date_match = re.search(r'\.(\d{8})-', f)
                if date_match:
                    date_str = date_match.group(1)
                    file_year = int(date_str[:4])
                    file_month = int(date_str[4:6])
                    if file_year == year and file_month in months:
                        month_key = f"{year}_{file_month:02d}"
                        if month_key not in monthly_files:
                            monthly_files[month_key] = []
                        monthly_files[month_key].append((f, gpm_dir))
    if not monthly_files:
        logger.error("没有找到GPM文件，退出")
        return
    # 测试模式
    if args.test:
        for key in monthly_files:
            monthly_files[key] = monthly_files[key][:5]
        logger.info("⚠️ 测试模式：每个月只处理前5个文件")
    # 打印统计信息
    total_files = sum(len(files) for files in monthly_files.values())
    logger.info(f"找到 {len(monthly_files)} 个月的数据，共 {total_files} 个文件")
    for month_key, files in sorted(monthly_files.items()):
        logger.info(f"  {month_key}: {len(files)} 个文件")
    # 逐月顺序处理（参考task_distribute_spatial的实现方式）
    sorted_months = sorted(monthly_files.keys())
    mp.set_start_method('spawn', force=True)
    # 逐月处理：每个月内部使用多GPU并行，处理完立即保存
    for month_key in sorted_months:
        files = monthly_files[month_key]
        year, month = int(month_key.split('_')[0]), int(month_key.split('_')[1])
        logger.info("\n" + "=" * 60)
        logger.info(f"开始处理 {month_key}，共 {len(files)} 个文件")
        logger.info(f"使用 {len(gpu_ids)} 个GPU并行处理")
        logger.info("=" * 60)
        # 将文件分配给各个GPU
        gpu_files = {gpu_id: [] for gpu_id in gpu_ids}
        for idx, file_info in enumerate(files):
            gpu_id = gpu_ids[idx % len(gpu_ids)]
            gpu_files[gpu_id].append(file_info)
        for gpu_id in gpu_ids:
            logger.info(f"  GPU {gpu_id}: 分配 {len(gpu_files[gpu_id])} 个文件")
        # 创建结果队列（用于worker传递结果给主进程）
        result_queue = mp.Queue(maxsize=20)
        # 启动GPU worker进程
        processes = []
        for gpu_id in gpu_ids:
            if gpu_files[gpu_id]:  # 只启动有任务的GPU
                p = mp.Process(
                    target=gpu_worker_with_queue,
                    args=(gpu_id, month_key, gpu_files[gpu_id], config, result_queue)
                )
                p.daemon = True
                p.start()
                processes.append(p)
                logger.info(f"[SPATIAL] Started process for GPU {gpu_id}")
        # 收集结果（整月完成后一次性保存）
        n_workers = len(processes)
        monthly_results = []
        finished_signals = 0
        total_results = 0
        total_failed = 0
        # 每个 worker 结束时向队列放入一个 None；完成条件必须用固定的 n_workers，
        # 不能用 len(processes) 且在循环里 remove 进程，否则会出现「只剩最后一个子进程时
        # finished_procs >= len(processes)」而提前退出、主进程结束并杀死 daemon 子进程的问题。
        while finished_signals < n_workers:
            try:
                res = result_queue.get(timeout=1.0)
                if res is None:
                    finished_signals += 1
                    logger.info(f"[SPATIAL] 收到 worker 结束信号 {finished_signals}/{n_workers}")
                    continue
                if res.get('type') == 'result':
                    monthly_results.append(res['data'])
                    total_results += 1
                    if total_results % 100 == 0:
                        logger.info(
                            f"[SPATIAL] 已处理 {total_results}/{len(files)} 个文件，"
                            f"内存中暂存 {len(monthly_results)} 个结果"
                        )
                    if len(monthly_results) % 50 == 0:
                        gc.collect()
                elif res.get('type') == 'error_count':
                    total_failed += res.get('count', 0)
            except Exception:
                pass
        for p in processes:
            p.join(timeout=120)
            if p.is_alive():
                logger.warning(f"[SPATIAL] 子进程 pid={p.pid} join 超时，仍在运行")
        # 整月完成后一次性保存所有结果
        if monthly_results:
            logger.info(f"[SPATIAL] 整月处理完成，开始保存 {len(monthly_results)} 个结果...")
            save_monthly_results_incremental(year, month, monthly_results, output_dir, logger, config)
            logger.info(f"[SPATIAL] 保存完成: {len(monthly_results)} 个结果")
        else:
            logger.warning(f"[SPATIAL] {month_key} 没有有效结果需要保存")
        logger.info(f"{month_key} 处理完成: 成功 {total_results}, 失败 {total_failed}")
        # 强制垃圾回收，准备下一个月
        gc.collect()
        torch.cuda.empty_cache()
    # 统计
    elapsed_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("所有月份处理完成")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()