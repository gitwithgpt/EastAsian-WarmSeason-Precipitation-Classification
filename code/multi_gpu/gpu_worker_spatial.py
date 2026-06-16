import os
import torch
import logging
import numpy as np
import h5py
import xarray as xr
from yaml import safe_load

from src.data_loader.gpm_loader import load_gpm_data, extract_gpm_time
from src.data_loader.era5_loader import load_era5_hour_slice
from src.match.match_core import load_match_config, spatial_matching  # 改用 spatial_matching 以获取 quality_flags
from src.feature.feature_core import extract_all_features
from .memory_control import check_gpu_memory


def load_config():
    """加载基础配置（与原 gpu_worker 相同）"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "base_config.yaml",
    )
    with open(config_path, "r", encoding="utf-8") as f:
        return safe_load(f)


def get_logger():
    return logging.getLogger(__name__)


from .gpu_worker import find_era5_files, find_era5_file_by_date  # 直接复用原函数


def single_gpu_worker_spatial(gpu_id, data_blocks, result_queue):
    """
    带经纬度输出的单GPU处理函数。

    与 src.multi_gpu.gpu_worker.single_gpu_worker 的逻辑基本一致，
    但在结果中额外附加每个样本的纬度/经度（lat, lon），以便后续空间分布分析。
    """
    from datetime import datetime
    from src.io_utils.path_utils import create_dir

    # 加载配置并初始化日志
    config = load_config()
    project_root = config.get("project_root", ".")
    log_dir = os.path.join(project_root, "temp", "logs")
    create_dir(log_dir)
    log_path = os.path.join(
        log_dir, f"gpu_{gpu_id}_worker_spatial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    logger = get_logger()

    # 绑定 GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"[SPATIAL] GPU {gpu_id} worker started, device: {device}")
    logger.info(f"[SPATIAL] log file: {log_path}")

    months = config["data_range"]["months"]

    # ERA5 缓存（与原逻辑一致）
    era5_pressure_paths_cache = {}
    era5_single_paths_cache = {}
    era5_lat = None
    era5_lon = None
    lon_range = config["data_range"]["lon_range"]
    lat_range = config["data_range"]["lat_range"]

    for block in data_blocks:
        if block.get("is_empty", False):
            continue

        block_id = block["block_id"]
        gpm_file_list = block.get("gpm_files", [])

        if not gpm_file_list:
            logger.warning(f"[SPATIAL] GPU {gpu_id}: Block {block_id} has no GPM files, skip")
            continue

        try:
            check_gpu_memory(gpu_id)

            for gpm_file_info in gpm_file_list:
                # 处理 (文件名, 目录) 或旧格式
                if isinstance(gpm_file_info, tuple):
                    gpm_file, gpm_dir = gpm_file_info
                    gpm_file_path = os.path.join(gpm_dir, gpm_file)
                else:
                    gpm_file = gpm_file_info
                    import re
                    from src.io_utils.path_utils import get_gpm_root_path

                    date_match = re.search(r"\.(\d{8})-", gpm_file)
                    if not date_match:
                        logger.debug(
                            f"[SPATIAL] GPU {gpu_id}: cannot parse year from file name, skip: {gpm_file}"
                        )
                        continue
                    date_str = date_match.group(1)
                    file_year = int(date_str[:4])
                    gpm_dir = get_gpm_root_path(file_year, config)
                    gpm_file_path = os.path.join(gpm_dir, gpm_file)

                if not os.path.exists(gpm_file_path):
                    logger.warning(
                        f"[SPATIAL] GPU {gpu_id}: GPM file not found, skip: {gpm_file_path}"
                    )
                    continue

                try:
                    # 时间信息
                    year, DayUnique, HourUnique = extract_gpm_time(gpm_file_path)
                    logger.debug(
                        f"[SPATIAL] GPU {gpu_id}: time for {gpm_file} -> year={year}, DayUnique={DayUnique}, HourUnique={HourUnique}"
                    )

                    # ERA5 文件路径缓存
                    if year not in era5_pressure_paths_cache:
                        logger.info(f"[SPATIAL] GPU {gpu_id}: scanning ERA5 paths for year {year} ...")
                        era5_pressure_paths_dict, era5_single_paths_dict = find_era5_files(
                            year, months, config
                        )
                        era5_pressure_paths_cache[year] = era5_pressure_paths_dict
                        era5_single_paths_cache[year] = era5_single_paths_dict
                    else:
                        era5_pressure_paths_dict = era5_pressure_paths_cache[year]
                        era5_single_paths_dict = era5_single_paths_cache[year]

                    if not era5_pressure_paths_dict and not era5_single_paths_dict:
                        logger.warning(
                            f"[SPATIAL] GPU {gpu_id}: no ERA5 files for year {year}, skip {gpm_file}"
                        )
                        continue

                    # 首次读取 ERA5 经纬度
                    if era5_lat is None or era5_lon is None:
                        era5_file_for_coords = None
                        if era5_pressure_paths_dict:
                            era5_file_for_coords = list(era5_pressure_paths_dict.values())[0]
                        elif era5_single_paths_dict:
                            any_val = list(era5_single_paths_dict.values())[0]
                            if isinstance(any_val, tuple):
                                era5_file_for_coords = any_val[0]
                            else:
                                era5_file_for_coords = any_val

                        if era5_file_for_coords and os.path.exists(era5_file_for_coords):
                            with xr.open_dataset(era5_file_for_coords) as ds:
                                if "latitude" not in ds.coords or "longitude" not in ds.coords:
                                    raise ValueError("ERA5文件缺少经纬度坐标")
                                lat_data = ds.coords["latitude"].values
                                if len(lat_data) > 1 and lat_data[0] > lat_data[-1]:
                                    lat_data = lat_data[::-1]
                                lon_data = ds.coords["longitude"].values

                            lat_mask = (lat_data >= lat_range[0]) & (lat_data <= lat_range[1])
                            lon_mask = (lon_data >= lon_range[0]) & (lon_data <= lon_range[1])
                            lat_c = lat_data[lat_mask]
                            lon_c = lon_data[lon_mask]
                            lat_grid, lon_grid = np.meshgrid(lat_c, lon_c, indexing="ij")
                            era5_lat = torch.from_numpy(lat_grid).float().to(device)
                            era5_lon = torch.from_numpy(lon_grid).float().to(device)
                            logger.info(
                                f"[SPATIAL] GPU {gpu_id}: ERA5 lat/lon grid loaded, shape: {era5_lat.shape}"
                            )
                        else:
                            logger.error(
                                f"[SPATIAL] GPU {gpu_id}: cannot find ERA5 file for lat/lon"
                            )
                            continue

                    # 加载 GPM 数据（含 lat/lon）
                    gpm_tensor, gpm_lat, gpm_lon, valid_indices = load_gpm_data(
                        gpm_file_path, device
                    )
                    logger.debug(
                        f"[SPATIAL] GPU {gpu_id}: loaded GPM data, shape={gpm_tensor.shape}, n_valid={len(gpm_lat)}"
                    )

                    # 计算目标日期和小时
                    import datetime

                    # 固定使用4月1日作为基准日期，确保与extract_gpm_time中的DayUnique计算一致
                    # 无论处理哪个月份，DayUnique都是从4月1日开始计算的
                    start_date = datetime.date(year, 4, 1)  # 固定为4月1日
                    target_date = start_date + datetime.timedelta(days=DayUnique - 1)
                    date_str = target_date.strftime("%Y%m%d")
                    month_str = target_date.strftime("%Y%m")

                    hour_clamped = max(0, min(23, HourUnique))
                    time_num = hour_clamped + 1
                    obs_dt = datetime.datetime(
                        target_date.year, target_date.month, target_date.day, hour_clamped, 0, 0
                    )

                    era5_pressure_path = era5_pressure_paths_dict.get(date_str)
                    era5_single_info = era5_single_paths_dict.get(month_str)
                    if isinstance(era5_single_info, tuple):
                        single_instant_path, single_accum_path = era5_single_info
                    else:
                        single_instant_path = era5_single_info
                        single_accum_path = None

                    # 加载 ERA5 pressure
                    era5_pressure_2d = None
                    if era5_pressure_path:
                        era5_p = load_era5_hour_slice(
                            era5_pressure_path, time_num, data_type="pressure", device=device
                        )
                        lat_size, lon_size, level_size, var_size = era5_p.shape
                        era5_pressure_2d = era5_p.reshape(
                            lat_size, lon_size, level_size * var_size
                        )

                    # 加载 ERA5 single
                    era5_tensor_2d = None
                    if single_instant_path:
                        era5_single_2d = load_era5_hour_slice(
                            single_instant_path,
                            time_num,
                            data_type="single",
                            device=device,
                            accum_file_path=single_accum_path,
                            pressure_file_path=era5_pressure_path,
                        )
                        if era5_pressure_2d is not None:
                            era5_tensor_2d = torch.cat(
                                [era5_pressure_2d, era5_single_2d], dim=-1
                            )
                        else:
                            # 仅 single level 时，用零数组占位 pressure
                            p_placeholder = torch.zeros(
                                era5_single_2d.shape[0],
                                era5_single_2d.shape[1],
                                162,
                                dtype=era5_single_2d.dtype,
                                device=era5_single_2d.device,
                            )
                            era5_tensor_2d = torch.cat([p_placeholder, era5_single_2d], dim=-1)
                    else:
                        if era5_pressure_2d is not None:
                            s_placeholder = torch.zeros(
                                era5_pressure_2d.shape[0],
                                era5_pressure_2d.shape[1],
                                13,
                                dtype=era5_pressure_2d.dtype,
                                device=era5_pressure_2d.device,
                            )
                            era5_tensor_2d = torch.cat([era5_pressure_2d, s_placeholder], dim=-1)

                    if era5_tensor_2d is None:
                        logger.error(
                            f"[SPATIAL] GPU {gpu_id}: no usable ERA5 data (date={date_str}, month={month_str}), skip {gpm_file}"
                        )
                        continue

                    # 匹配（使用 hybrid 方法，强制返回 quality_flags 用于审计）
                    gpm_tensor_2d = gpm_tensor.unsqueeze(1)
                    gpm_lat_2d = gpm_lat.unsqueeze(1)
                    gpm_lon_2d = gpm_lon.unsqueeze(1)

                    # 读取配置
                    match_cfg = load_match_config()
                    sm = match_cfg.get("spatial_matching") or {}
                    method = (sm.get("method") or "hybrid").lower()
                    threshold_km = sm.get("nearest_threshold") or match_cfg["match_rule"].get("dist_threshold_km", 20.0)
                    idw_k = sm.get("idw_k", 4)
                    idw_power = sm.get("idw_power", 2)

                    # 调用 spatial_matching 并获取 quality_flags
                    matched_raw, quality_flags = spatial_matching(
                        gpm_tensor_2d,
                        era5_tensor_2d,
                        gpm_lat_2d,
                        gpm_lon_2d,
                        era5_lat,
                        era5_lon,
                        method=method,
                        threshold_km=threshold_km,
                        idw_k=idw_k,
                        idw_power=idw_power,
                        return_quality_flags=True,  # 强制返回质量标记
                    )
                    matched_raw = matched_raw.squeeze(1)
                    # quality_flags 形状 (h, w) -> 展平为 (n_pts,) 以便逐样本追踪
                    quality_flags_flat = quality_flags.ravel()

                    # 统计 far_match / no_match 比例（用于日志审计）
                    n_total = quality_flags_flat.size
                    n_ok = np.sum(quality_flags_flat == "ok")
                    n_far = np.sum(quality_flags_flat == "far_match")
                    n_no = np.sum(quality_flags_flat == "no_match")
                    logger.info(
                        f"[SPATIAL] GPU {gpu_id}: matching done, shape={matched_raw.shape}, "
                        f"quality(ok/far/no)=({n_ok}/{n_far}/{n_no})"
                    )

                    # 特征提取
                    feature_set = extract_all_features(matched_raw)
                    logger.debug(
                        f"[SPATIAL] GPU {gpu_id}: feature extraction done, feature_set shape={feature_set.shape}"
                    )

                    # 唯一 block_id（与原逻辑一致）
                    file_idx = 0
                    for idx2, fi in enumerate(gpm_file_list):
                        if isinstance(fi, tuple):
                            if fi[0] == gpm_file:
                                file_idx = idx2
                                break
                        elif fi == gpm_file:
                            file_idx = idx2
                            break
                    unique_block_id = block_id * 10000 + file_idx

                    # 转 CPU + numpy，同时附加 lat/lon
                    n_pts = int(gpm_lat.shape[0])
                    # Use file-level real observation hour as base time, and add per-sample
                    # seconds to keep strictly increasing timestamps within file.
                    # This avoids zero-delta issues in downstream pair building.
                    base_t = np.datetime64(obs_dt, "s")
                    time_arr = base_t + np.arange(n_pts, dtype="timedelta64[s]")
                    result_queue.put(
                        {
                            "block_id": unique_block_id,
                            "matched_raw": matched_raw.cpu().numpy(),
                            "feature_set": feature_set.cpu().numpy(),
                            "lat": gpm_lat.cpu().numpy().astype(np.float32),
                            "lon": gpm_lon.cpu().numpy().astype(np.float32),
                            "time": time_arr,
                            "quality_flags": quality_flags_flat,  # 新增：质量标记（"ok"/"far_match"/"no_match"）
                        }
                    )

                    logger.info(
                        f"[SPATIAL] GPU {gpu_id}: finished file {gpm_file} (block_id={unique_block_id})"
                    )

                    # 释放显存
                    del gpm_tensor, gpm_tensor_2d, matched_raw, feature_set
                    torch.cuda.empty_cache()

                except Exception as e:
                    logger.error(
                        f"[SPATIAL] GPU {gpu_id}: failed to process GPM file {gpm_file}: {str(e)}",
                        exc_info=True,
                    )
                    continue

            logger.info(f"[SPATIAL] GPU {gpu_id}: finished Block {block_id}")

        except Exception as e:
            logger.error(
                f"[SPATIAL] GPU {gpu_id}: failed to process Block {block_id}: {str(e)}",
                exc_info=True,
            )
            continue

    logger.info(f"[SPATIAL] GPU {gpu_id}: all tasks finished")


