import os
import torch
import logging
import torch.multiprocessing as mp
import numpy as np
from yaml import safe_load
from .gpu_worker import single_gpu_worker
from src.io_utils.npz_writer import batch_write_npz
from src.io_utils.log_utils import get_unfinished_blocks


def load_config():
    """加载配置文件"""
    base_config = safe_load(
        open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "base_config.yaml"),
             "r", encoding="utf-8"))
    gpu_config = safe_load(open(
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "multi_gpu_config.yaml"),
        "r", encoding="utf-8"))
    return {**base_config, **gpu_config}


def get_logger():
    """获取日志器"""
    return logging.getLogger(__name__)


def split_data_into_blocks():
    """
    生成数据分块
    根据run_single_gpu.py的逻辑，按GPM文件列表分块，而不是按空间分块
    支持多年份（2014-2024年）的GPM文件查找
    """
    import h5py
    import re
    from src.io_utils.path_utils import get_gpm_root_path
    
    config = load_config()
    logger = get_logger()
    
    years = config["data_range"].get("years", [2024])  # 支持多年份
    months = config["data_range"]["months"]
    lon_range = config["data_range"]["lon_range"]
    lat_range = config["data_range"]["lat_range"]
    fill_value = -9999.9
    
    # 构建月份字符串列表（用于文件名匹配）
    month_strs = [f"{month:02d}" for month in months]  # ['04', '05', '06', '07', '08', '09']
    
    # 1. 遍历所有年份，获取所有GPM文件
    all_candidate_gpm_files = []
    
    for year in years:
        # 根据年份获取GPM数据目录
        gpm_dir = get_gpm_root_path(year, config)
        
        if not os.path.exists(gpm_dir):
            logger.warning(f"GPM数据目录不存在，跳过年份 {year}: {gpm_dir}")
            continue
        
        # 获取该年份的所有GPM文件
        try:
            # 支持多种扩展名：.HDF5, .hdf5, .hdf, .HDF
            all_gpm_files = [f for f in os.listdir(gpm_dir) 
                           if f.lower().endswith(('.hdf5', '.hdf'))]
            
            # 筛选文件名含目标月份的文件（匹配格式：YYYYMMDD，其中MM在months中）
            for f in all_gpm_files:
                # 从文件名提取日期（格式：.YYYYMMDD-）
                date_match = re.search(r'\.(\d{8})-', f)
                if date_match:
                    date_str = date_match.group(1)
                    file_year = int(date_str[:4])
                    file_month = int(date_str[4:6])
                    
                    # 检查是否在目标年份和月份范围内
                    if file_year == year and file_month in months:
                        all_candidate_gpm_files.append((f, gpm_dir))  # 保存文件名和目录路径
        except (PermissionError, OSError) as e:
            logger.warning(f"无法访问GPM数据目录 {gpm_dir}，错误: {str(e)}")
            continue
    
    candidate_gpm_files = [f[0] for f in all_candidate_gpm_files]  # 仅文件名列表，用于后续处理
    
    logger.info(f"找到候选GPM文件（{years}年{months}月）: {len(candidate_gpm_files)}个，开始筛选有效文件...")
    
    # 2. 对每个候选文件，快速判断是否含研究区有效像元
    valid_gpm_files = []
    for gpm_file, gpm_dir in all_candidate_gpm_files:
        gpm_file_path = os.path.join(gpm_dir, gpm_file)
        try:
            with h5py.File(gpm_file_path, 'r') as h5f:
                if 'FS' not in h5f:
                    continue
                
                fs_group = h5f['FS']
                if 'Latitude' not in fs_group or 'Longitude' not in fs_group:
                    continue
                
                # 读取经纬度和降水率（用于快速筛选，与2024年测试代码一致）
                lat_data = fs_group['Latitude'][:]
                lon_data = fs_group['Longitude'][:]
                
                # 读取降水率（如果存在，用于阈值筛选）
                precip_data = None
                if 'SLV' in fs_group and 'precipRateNearSurface' in fs_group['SLV']:
                    precip_data = fs_group['SLV']['precipRateNearSurface'][:]
                
                if not isinstance(lat_data, np.ndarray):
                    lat_data = np.array(lat_data)
                if not isinstance(lon_data, np.ndarray):
                    lon_data = np.array(lon_data)
                
                # 空间范围筛选
                spatial_mask = (
                    (lat_data >= lat_range[0]) & (lat_data <= lat_range[1]) &
                    (lon_data >= lon_range[0]) & (lon_data <= lon_range[1]) &
                    (lat_data != fill_value) & (lon_data != fill_value)
                )
                
                # 如果有降水率数据，添加降水率阈值筛选（precipRateNearSurface >= 0.5 mm/hr）
                if precip_data is not None:
                    if not isinstance(precip_data, np.ndarray):
                        precip_data = np.array(precip_data)
                    precip_threshold = 0.5  # mm/hr，与Matlab代码一致
                    precip_mask = (precip_data >= precip_threshold) & (precip_data != fill_value)
                    valid_mask = spatial_mask & precip_mask
                else:
                    # 如果没有降水率数据，只使用空间范围筛选（实际筛选会在load_gpm_data中进行）
                    valid_mask = spatial_mask
                
                if np.any(valid_mask):
                    # 保存完整路径（目录+文件名），以便后续处理
                    valid_gpm_files.append((gpm_file, gpm_dir))
        except Exception:
            continue
    
    if not valid_gpm_files:
        raise FileNotFoundError(f"未找到含研究区有效像元的GPM文件")
    
    logger.info(f"筛选完成：找到 {len(valid_gpm_files)} 个有效GPM文件")
    
    # 3. 将GPM文件列表分块（每个块包含多个文件，便于并行处理）
    # 根据配置的block大小，将文件列表分成多个块
    # 为了适配4卡，每个块包含的文件数应该合理分配
    num_gpus = config.get("gpu", {}).get("num_gpus", 4)
    files_per_block = max(1, len(valid_gpm_files) // (num_gpus * 4))  # 每个块的文件数
    
    data_blocks = []
    block_id = 0
    
    for i in range(0, len(valid_gpm_files), files_per_block):
        block_files = valid_gpm_files[i:i + files_per_block]
        # block_files是元组列表：(文件名, 目录路径)
        data_blocks.append({
            "block_id": block_id,
            "gpm_files": block_files,  # 保存(文件名, 目录路径)元组列表
            "lon_range": lon_range,  # 保留用于兼容性
            "lat_range": lat_range   # 保留用于兼容性
        })
        block_id += 1
    
    # 补全为4的整数倍（适配4卡）
    if len(data_blocks) % 4 != 0:
        pad_num = 4 - (len(data_blocks) % 4)
        for i in range(pad_num):
            data_blocks.append({"block_id": block_id + i, "is_empty": True})
    
    logger.info(f"生成 {len(data_blocks)} 个数据块（每个块约 {files_per_block} 个文件）")
    
    return data_blocks


def multi_gpu_task_distribute():
    """多GPU任务分发主函数"""
    config = load_config()
    logger = get_logger()
    num_gpus = config["gpu"]["num_gpus"]
    batch_size = config["gpu"]["batch_write_size"]

    # 设置多进程启动方式
    mp.set_start_method(config["gpu"]["start_method"], force=True)

    # 生成数据块
    all_blocks = split_data_into_blocks()
    # 分配数据块到各GPU
    gpu_blocks = {i: [] for i in range(num_gpus)}
    for idx, block in enumerate(all_blocks):
        gpu_blocks[idx % num_gpus].append(block)

    # 检查断点
    unfinished_blocks = get_unfinished_blocks()
    if unfinished_blocks is not None:
        for gpu_id in gpu_blocks:
            gpu_blocks[gpu_id] = [b for b in gpu_blocks[gpu_id] if b["block_id"] not in unfinished_blocks]

    # 创建结果队列
    result_queue = mp.Queue(maxsize=batch_size * 2)

    # 启动GPU进程
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=single_gpu_worker, args=(gpu_id, gpu_blocks[gpu_id], result_queue))
        p.daemon = True
        p.start()
        processes.append(p)
        logger.info(f"Started process for GPU {gpu_id}")

    # 结果收集与写入
    raw_buffer = []
    feat_buffer = []
    finished_procs = 0
    total_results = 0

    while finished_procs < num_gpus:
        # 读取队列结果（使用超时避免无限等待）
        try:
            res = result_queue.get(timeout=1.0)
            # batch_write_npz期望的数据结构：{"block_id": ..., "matched_raw": ...} 或 {"block_id": ..., "feature_set": ...}
            raw_buffer.append({"block_id": res["block_id"], "matched_raw": res["matched_raw"]})
            feat_buffer.append({"block_id": res["block_id"], "feature_set": res["feature_set"]})
            total_results += 1

            # 批量写入
            if len(raw_buffer) >= batch_size:
                batch_write_npz(raw_buffer, "matched_raw", config)
                batch_write_npz(feat_buffer, "feature_set", config)
                logger.info(f"已写入 {len(raw_buffer)} 个结果块，累计: {total_results} 个")
                raw_buffer.clear()
                feat_buffer.clear()
        except:
            # 队列为空或超时，继续检查进程状态
            pass

        # 检查进程状态
        for p in list(processes):  # 使用list副本避免迭代时修改
            if not p.is_alive():
                finished_procs += 1
                processes.remove(p)
                logger.info(f"GPU进程已完成，剩余: {num_gpus - finished_procs} 个进程")

    # 写入剩余数据
    if len(raw_buffer) > 0:
        batch_write_npz(raw_buffer, "matched_raw", config)
        batch_write_npz(feat_buffer, "feature_set", config)
        logger.info(f"写入最后 {len(raw_buffer)} 个结果块")

    logger.info(f"All GPU processes finished, 总共处理 {total_results} 个结果，results saved")