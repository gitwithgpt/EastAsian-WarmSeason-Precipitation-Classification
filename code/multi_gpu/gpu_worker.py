import os
import torch
import logging
import numpy as np
import h5py
import xarray as xr
from yaml import safe_load
from src.data_loader.gpm_loader import load_gpm_data, extract_gpm_time
from src.data_loader.era5_loader import load_era5_data, load_era5_hour_slice
from src.match.match_core import match_gpm_era5
from src.feature.feature_core import extract_all_features
from .memory_control import check_gpu_memory


def load_config():
    """加载基础配置"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config",
                               "base_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return safe_load(f)


def get_logger():
    """获取日志器"""
    return logging.getLogger(__name__)


def find_era5_files(year, months, config=None):
    """
    查找ERA5文件（pressure level和single level）
    支持多年份和多种文件命名格式：
    - 压力层：E:/ERA5/年份/ 目录下的文件
      - 2014-2023年格式: 2022-04 adaptor.mars.internal-xxx.nc（一个月一个文件）
      - 2024年格式: 2024-09-30-xxxx.nc 或 pressure level_20240501.nc（一天一个文件）
    - 单层：E:/ERA5/SingleLevel/年份/ 目录下的文件
      - 2014-2023年格式: 2023-04-01-04-10 adaptor.mars.internal-xxx.nc（十天一个文件）
      - 2024年格式: 2024-04 data_stream-oper_stepType-instant.nc（一个月一个文件）
    :param year: 年份（2014-2024）
    :param months: 月份列表，如[4, 5, 6, 7, 8, 9]
    :param config: 配置字典（如果为None则从文件加载）
    :return: (era5_pressure_paths_dict, era5_single_paths_dict)
        - era5_pressure_paths_dict: {日期字符串: 文件路径}，如{"20240401": "path/to/file.nc"}
        - era5_single_paths_dict: {月份字符串: (instant路径, accum路径)}，如{"202404": ("instant.nc", "accum.nc")}
    """
    import re
    import datetime
    import calendar
    from src.data_loader.era5_loader import load_config
    from src.io_utils.path_utils import get_era5_pressure_dir, get_era5_single_dir
    
    logger = get_logger()
    if config is None:
        config = load_config()
    
    era5_pressure_paths_dict = {}  # {日期: 文件路径}
    era5_single_paths_dict = {}    # {月份: (instant路径, accum路径)}
    
    # 从配置文件读取路径（根据年份动态构建）
    era5_pressure_dir = get_era5_pressure_dir(year, config)
    era5_single_dir = get_era5_single_dir(year, config)
    
    # 构建月份字符串列表和目标日期范围
    month_strs = [f"{year}{month:02d}" for month in months]  # ['202404', '202405', ...]
    
    # 计算目标日期范围（用于验证提取的日期是否在范围内）
    start_date = datetime.date(year, months[0], 1)
    # 使用当月的实际最后一天，避免 9 月 29、30 号被错误截断
    last_day = calendar.monthrange(year, months[-1])[1]
    end_date = datetime.date(year, months[-1], last_day)
    
    # ========== 查找压力层文件 ==========
    if os.path.exists(era5_pressure_dir):
        try:
            pressure_files = [f for f in os.listdir(era5_pressure_dir) if f.endswith('.nc')]
            logger.info(f"在压力层目录 {era5_pressure_dir} 中找到 {len(pressure_files)} 个.nc文件")
            
            for f in pressure_files:
                full_path = os.path.join(era5_pressure_dir, f)
                
                # 根据年份选择不同的解析策略
                if year >= 2024:
                    # 2024年格式：一天一个文件
                    # 模式1: 2024-09-30-xxxx.nc 或 2024-9-30-xxxx.nc 格式
                    date_str = None
                    match1 = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', f)
                    if match1:
                        year_str, month_str_raw, day_str_raw = match1.groups()
                        # 补零到两位，构造标准 YYYYMMDD
                        month_str = f"{int(month_str_raw):02d}"
                        day_str = f"{int(day_str_raw):02d}"
                        date_str = f"{year_str}{month_str}{day_str}"
                    
                    # 模式2: pressure level_20240501.nc 或类似格式（包含8位数字日期）
                    if not date_str:
                        match2 = re.search(r'(\d{8})', f)
                        if match2:
                            date_str = match2.group(1)
                    
                    # 验证日期是否在目标范围内
                    if date_str:
                        try:
                            file_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
                            if start_date <= file_date <= end_date:
                                era5_pressure_paths_dict[date_str] = full_path
                                logger.debug(f"找到压力层文件: {f} -> 日期: {date_str}")
                        except ValueError:
                            logger.warning(f"无法解析日期格式: {date_str} (文件: {f})")
                    else:
                        logger.warning(f"无法从文件名提取日期: {f}")
                else:
                    # 2014-2023年格式：十天一个文件，格式如：2022-04-01-04-10 adaptor.mars.internal-xxx.nc
                    # 匹配格式：YYYY-MM-DD-MM-DD（十天范围）
                    match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})', f)
                    if match:
                        file_year = int(match.group(1))
                        start_month = int(match.group(2))
                        start_day = int(match.group(3))
                        end_month = int(match.group(4))
                        end_day = int(match.group(5))
                        
                        # 检查是否在目标年份和月份范围内
                        if file_year == year and start_month in months:
                            # 计算该十天周期的所有日期
                            start_date_obj = datetime.date(file_year, start_month, start_day)
                            # 结束日期（假设是十天周期，结束日期应该是start_day+9，但也要考虑跨月情况）
                            # 根据文件名中的end_month和end_day确定结束日期
                            if end_month == start_month:
                                # 同一个月内
                                end_date_obj = datetime.date(file_year, end_month, end_day)
                            else:
                                # 跨月情况（较少见，但需要处理）
                                end_date_obj = datetime.date(file_year, end_month, end_day)
                            
                            # 为该十天周期内的所有日期创建映射
                            current_date = start_date_obj
                            while current_date <= end_date_obj:
                                # 只处理目标月份范围内的日期
                                if current_date.year == year and current_date.month in months:
                                    date_str = current_date.strftime("%Y%m%d")
                                    era5_pressure_paths_dict[date_str] = full_path
                                # 移动到下一天
                                current_date += datetime.timedelta(days=1)
                            
                            logger.debug(f"找到压力层文件: {f} -> 日期范围: {start_date_obj} 到 {end_date_obj}")
                    else:
                        # 如果没有匹配到十天格式，尝试匹配月份格式（向后兼容）
                        match_month = re.search(r'(\d{4})-(\d{1,2})', f)
                        if match_month:
                            file_year = int(match_month.group(1))
                            file_month = int(match_month.group(2))
                            
                            # 检查是否在目标年份和月份范围内
                            if file_year == year and file_month in months:
                                # 为该月的所有日期创建映射（使用该文件）
                                for month in months:
                                    if month == file_month:
                                        month_str = f"{year}{month:02d}"
                                        # 获取该月的所有日期
                                        days_in_month = calendar.monthrange(year, month)[1]
                                        for day in range(1, days_in_month + 1):
                                            date_str = f"{year}{month:02d}{day:02d}"
                                            era5_pressure_paths_dict[date_str] = full_path
                                        logger.info(f"找到压力层文件: {f} -> 月份: {month_str} (覆盖该月所有日期)")
                                        break
        except (PermissionError, OSError) as e:
            logger.error(f"无法访问压力层目录: {era5_pressure_dir}, 错误: {str(e)}")
    else:
        logger.warning(f"压力层目录不存在: {era5_pressure_dir}")
    
    # ========== 查找单层文件 ==========
    if os.path.exists(era5_single_dir):
        try:
            single_files = [f for f in os.listdir(era5_single_dir) if f.endswith('.nc')]
            logger.info(f"在单层目录 {era5_single_dir} 中找到 {len(single_files)} 个.nc文件")
            
            if year >= 2024:
                # 2024年格式：一个月一个文件
                # 分离instant和accum文件
                instant_files = [f for f in single_files if 'instant' in f.lower()]
                accum_files = [f for f in single_files if 'accum' in f.lower()]
                
                # 为每个目标月份查找对应的文件
                for month_str in month_strs:
                    month_instant = []
                    month_accum = []
                    
                    # 查找包含该月份的文件
                    for f in instant_files:
                        # 检查文件名是否包含月份（如 2024-04 或 202404）
                        if month_str in f or f"{year}-{month_str[-2:]}" in f:
                            month_instant.append(f)
                    
                    for f in accum_files:
                        if month_str in f or f"{year}-{month_str[-2:]}" in f:
                            month_accum.append(f)
                    
                    # 如果找到instant文件，记录路径
                    if month_instant:
                        instant_path = os.path.join(era5_single_dir, month_instant[0])
                        # 查找对应的accum文件（优先使用同月份的文件）
                        accum_path = None
                        if month_accum:
                            accum_path = os.path.join(era5_single_dir, month_accum[0])
                        
                        if os.path.exists(instant_path):
                            if accum_path and os.path.exists(accum_path):
                                era5_single_paths_dict[month_str] = (instant_path, accum_path)
                                logger.info(f"找到单层文件 {month_str}: instant={os.path.basename(instant_path)}, accum={os.path.basename(accum_path)}")
                            else:
                                era5_single_paths_dict[month_str] = (instant_path, None)
                                logger.warning(f"找到instant文件但未找到accum文件: {os.path.basename(instant_path)}")
                    else:
                        logger.warning(f"未找到月份 {month_str} 的单层文件")
            else:
                # 2014-2023年格式：一个月一个文件，格式如：2014-04 adaptor.mars.internal-xxx.nc
                # 注意：根据用户说明，2014-2023年的单层数据是一个月一个文件，不是十天一个文件
                # 文件命名格式：年份-月份 adaptor.mars.internal-xxx.nc
                
                # 为每个目标月份查找对应的文件
                for month_str in month_strs:
                    month_num = int(month_str[-2:])  # 提取月份数字，如 "04" -> 4
                    month_pattern = f"{year}-{month_num:02d}"  # 如 "2014-04"
                    
                    # 查找包含该年份-月份模式的文件
                    month_files = []
                    for f in single_files:
                        # 检查文件名是否以年份-月份开头（如 2014-04）
                        # 匹配格式：YYYY-MM 开头的文件名
                        if f.startswith(month_pattern) or f"{year}-{month_num}" in f:
                            month_files.append(f)
                    
                    if month_files:
                        # 如果找到文件，使用第一个文件（通常一个月只有一个文件）
                        # 注意：2014-2023年的单层文件可能不区分instant和accum，需要根据实际情况处理
                        file_path = os.path.join(era5_single_dir, month_files[0])
                        
                        if os.path.exists(file_path):
                            # 对于2014-2023年，单层文件可能不区分instant和accum
                            # 如果文件名中包含instant或accum，则分别处理；否则使用同一个文件
                            instant_file = None
                            accum_file = None
                            
                            for f in month_files:
                                if 'instant' in f.lower():
                                    instant_file = f
                                elif 'accum' in f.lower():
                                    accum_file = f
                            
                            if instant_file:
                                instant_path = os.path.join(era5_single_dir, instant_file)
                                accum_path = os.path.join(era5_single_dir, accum_file) if accum_file else None
                            else:
                                # 没有明确的instant/accum区分，使用同一个文件
                                instant_path = file_path
                                accum_path = file_path  # 使用同一个文件
                            
                            if os.path.exists(instant_path):
                                era5_single_paths_dict[month_str] = (instant_path, accum_path if accum_path and os.path.exists(accum_path) else None)
                                logger.info(f"找到单层文件 {month_str}: {os.path.basename(instant_path)}" + 
                                          (f", accum={os.path.basename(accum_path)}" if accum_path and os.path.exists(accum_path) else ""))
                            else:
                                logger.warning(f"单层文件路径不存在: {instant_path}")
                        else:
                            logger.warning(f"单层文件路径不存在: {file_path}")
                    else:
                        logger.warning(f"未找到月份 {month_str} 的单层文件（在目录 {era5_single_dir} 中查找模式 {month_pattern}）")
        except (PermissionError, OSError) as e:
            logger.error(f"无法访问单层目录: {era5_single_dir}, 错误: {str(e)}")
    else:
        logger.warning(f"单层目录不存在: {era5_single_dir}")
    
    logger.info(f"找到ERA5 pressure level文件: {len(era5_pressure_paths_dict)}个（按日期索引）")
    logger.info(f"找到ERA5 single level文件: {len(era5_single_paths_dict)}个（按月份索引）")
    
    return era5_pressure_paths_dict, era5_single_paths_dict


def find_era5_file_by_date(era5_pressure_paths_dict, era5_single_paths_dict, DayUnique, year, months):
    """
    根据DayUnique查找对应的ERA5文件路径
    :param era5_pressure_paths_dict: {日期: 文件路径}字典
    :param era5_single_paths_dict: {月份: (instant_path, accum_path)}字典
    :param DayUnique: 日期唯一标识（从1开始，对应4月1日为1）
    :param year: 年份
    :param months: 月份列表，如[4, 5, 6, 7, 8, 9]
    :return: (pressure_file_path, (instant_path, accum_path))
    """
    import datetime
    # 固定使用4月1日作为基准日期，确保与extract_gpm_time中的DayUnique计算一致
    # 无论处理哪个月份，DayUnique都是从4月1日开始计算的
    start_date = datetime.date(year, 4, 1)  # 固定为4月1日
    target_date = start_date + datetime.timedelta(days=DayUnique - 1)
    date_str = target_date.strftime("%Y%m%d")
    month_str = target_date.strftime("%Y%m")
    
    pressure_path = era5_pressure_paths_dict.get(date_str)
    single_path_info = era5_single_paths_dict.get(month_str)
    
    # 确保返回格式一致：single_path_info 可能是元组或None
    if single_path_info is None:
        single_path_info = (None, None)
    elif not isinstance(single_path_info, tuple):
        # 兼容旧格式（单个路径）
        single_path_info = (single_path_info, None)
    
    return pressure_path, single_path_info


def single_gpu_worker(gpu_id, data_blocks, result_queue):
    """
    单GPU处理函数：加载数据→匹配→特征提取
    根据run_single_gpu.py的逻辑，适配新的处理方式
    :param gpu_id: GPU编号
    :param data_blocks: 分配的数据集块列表（每个块包含多个GPM文件）
    :param result_queue: 结果队列
    """
    # 在spawn模式下，子进程需要重新初始化日志配置
    import logging
    from datetime import datetime
    from src.io_utils.path_utils import create_dir
    
    # 加载配置以获取日志路径
    config = load_config()
    project_root = config.get("project_root", ".")
    log_dir = os.path.join(project_root, "temp", "logs")
    create_dir(log_dir)
    log_path = os.path.join(log_dir, f"gpu_{gpu_id}_worker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 初始化日志配置（每个worker进程都需要）
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(process)d - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True  # 强制重新配置，避免重复配置错误
    )
    
    logger = get_logger()
    
    # 绑定GPU
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"GPU {gpu_id} worker started, device: {device}")
    logger.info(f"GPU {gpu_id} worker日志文件: {log_path}")
    import numpy as np

    # 配置已加载，继续使用
    months = config["data_range"]["months"]
    
    # ERA5文件路径字典（按年份缓存，按需加载）
    era5_pressure_paths_cache = {}  # {年份: {日期: 文件路径}}
    era5_single_paths_cache = {}    # {年份: {月份: (instant_path, accum_path)}}
    
    # ERA5经纬度（延迟加载，在处理第一个GPM文件时读取）
    era5_lat = None
    era5_lon = None
    lon_range = config["data_range"]["lon_range"]
    lat_range = config["data_range"]["lat_range"]
    
    # 维护single level文件缓存（按月份，因为single level文件包含整个月份的数据）
    era5_single_cache = {}  # {月份字符串: Tensor}
    
    # 处理每个数据块
    for block in data_blocks:
        if block.get("is_empty", False):
            continue
        
        block_id = block["block_id"]
        gpm_file_list = block.get("gpm_files", [])
        
        if not gpm_file_list:
            logger.warning(f"GPU {gpu_id}: Block {block_id} 没有GPM文件，跳过")
            continue
        
        try:
            # 检查显存
            check_gpu_memory(gpu_id)
            
            # 循环处理每个GPM文件
            for gpm_file_info in gpm_file_list:
                # gpm_file_info应该是(文件名, 目录路径)元组
                if isinstance(gpm_file_info, tuple):
                    gpm_file, gpm_file_dir = gpm_file_info
                    gpm_file_path = os.path.join(gpm_file_dir, gpm_file)
                else:
                    # 兼容旧格式（仅文件名），需要从配置中获取GPM根目录
                    # 注意：旧格式无法确定年份，需要从文件名提取
                    gpm_file = gpm_file_info
                    # 尝试从文件名提取年份
                    import re
                    date_match = re.search(r'\.(\d{8})-', gpm_file)
                    if date_match:
                        date_str = date_match.group(1)
                        file_year = int(date_str[:4])
                        from src.io_utils.path_utils import get_gpm_root_path
                        gpm_file_dir = get_gpm_root_path(file_year, config)
                        gpm_file_path = os.path.join(gpm_file_dir, gpm_file)
                    else:
                        logger.warning(f"GPU {gpu_id}: 无法从文件名提取年份，跳过: {gpm_file}")
                        continue
                
                if not os.path.exists(gpm_file_path):
                    logger.warning(f"GPU {gpu_id}: GPM文件不存在，跳过: {gpm_file_path}")
                    continue
                
                try:
                    # 从GPM文件名提取时间信息（包含年份）
                    year, DayUnique, HourUnique = extract_gpm_time(gpm_file_path)
                    logger.info(f"GPU {gpu_id}: GPM文件 {gpm_file} 时间信息: year={year}, DayUnique={DayUnique}, HourUnique={HourUnique}")
                    
                    # 按需加载该年份的ERA5文件路径字典
                    if year not in era5_pressure_paths_cache:
                        logger.info(f"GPU {gpu_id}: 查找年份 {year} 的ERA5文件路径...")
                        era5_pressure_paths_dict, era5_single_paths_dict = find_era5_files(year, months, config)
                        era5_pressure_paths_cache[year] = era5_pressure_paths_dict
                        era5_single_paths_cache[year] = era5_single_paths_dict
                    else:
                        era5_pressure_paths_dict = era5_pressure_paths_cache[year]
                        era5_single_paths_dict = era5_single_paths_cache[year]
                    
                    if not era5_pressure_paths_dict and not era5_single_paths_dict:
                        logger.warning(f"GPU {gpu_id}: 年份 {year} 未找到任何ERA5文件，跳过GPM文件 {gpm_file}")
                        continue
                    
                    # 延迟加载ERA5经纬度（在处理第一个GPM文件时读取）
                    if era5_lat is None or era5_lon is None:
                        logger.info(f"GPU {gpu_id}: 从ERA5文件中读取真实经纬度...")
                        # 获取一个ERA5文件路径用于读取经纬度
                        era5_file_for_coords = None
                        if era5_pressure_paths_dict:
                            era5_file_for_coords = list(era5_pressure_paths_dict.values())[0]
                        elif era5_single_paths_dict:
                            # era5_single_paths_dict的值可能是元组 (instant_path, accum_path)
                            single_path_info = list(era5_single_paths_dict.values())[0]
                            if isinstance(single_path_info, tuple):
                                era5_file_for_coords = single_path_info[0]  # 使用instant文件路径
                            else:
                                era5_file_for_coords = single_path_info
                        
                        if era5_file_for_coords and os.path.exists(era5_file_for_coords):
                            try:
                                with xr.open_dataset(era5_file_for_coords) as era5_ds:
                                    # 读取经纬度坐标
                                    if 'latitude' in era5_ds.coords:
                                        era5_lat_data = era5_ds.coords['latitude'].values
                                        # ERA5纬度默认从90°N到90°S，需要反转
                                        if len(era5_lat_data) > 1 and era5_lat_data[0] > era5_lat_data[-1]:
                                            era5_lat_data = era5_lat_data[::-1]
                                    else:
                                        raise ValueError("ERA5文件缺少latitude坐标")
                                    
                                    if 'longitude' in era5_ds.coords:
                                        era5_lon_data = era5_ds.coords['longitude'].values
                                    else:
                                        raise ValueError("ERA5文件缺少longitude坐标")
                                    
                                    # 裁剪到研究区范围
                                    era5_lat_mask = (era5_lat_data >= lat_range[0]) & (era5_lat_data <= lat_range[1])
                                    era5_lon_mask = (era5_lon_data >= lon_range[0]) & (era5_lon_data <= lon_range[1])
                                    
                                    era5_lat_cropped = era5_lat_data[era5_lat_mask]
                                    era5_lon_cropped = era5_lon_data[era5_lon_mask]
                                    
                                    # 生成网格
                                    era5_lat_grid, era5_lon_grid = np.meshgrid(era5_lat_cropped, era5_lon_cropped, indexing='ij')
                                    
                                    era5_lat = torch.from_numpy(era5_lat_grid).float().to(device)
                                    era5_lon = torch.from_numpy(era5_lon_grid).float().to(device)
                                    
                                    logger.info(f"GPU {gpu_id}: ERA5经纬度提取完成，网格形状: {era5_lat.shape}")
                            except Exception as e:
                                logger.error(f"GPU {gpu_id}: 读取ERA5经纬度失败: {era5_file_for_coords}, 错误: {str(e)}", exc_info=True)
                                continue
                        else:
                            logger.error(f"GPU {gpu_id}: 无法找到ERA5文件用于读取经纬度")
                            continue
                    
                    # 加载单个GPM文件的数据和经纬度
                    gpm_tensor, gpm_lat, gpm_lon, valid_indices = load_gpm_data(gpm_file_path, device)
                    logger.info(f"GPU {gpu_id}: GPM数据加载完成，形状: {gpm_tensor.shape}, 有效像元: {len(gpm_lat)}")
                    
                    # 根据GPM时间信息，按需加载对应的ERA5数据
                    DayStartNumberERA5pressure = 1  # 默认值
                    
                    # 查找对应的ERA5文件路径（根据DayUnique确定目标日期）
                    import datetime
                    # 固定使用4月1日作为基准日期，确保与extract_gpm_time中的DayUnique计算一致
                    # 无论处理哪个月份，DayUnique都是从4月1日开始计算的
                    start_date = datetime.date(year, 4, 1)  # 固定为4月1日
                    target_date = start_date + datetime.timedelta(days=DayUnique - 1)
                    date_str = target_date.strftime("%Y%m%d")
                    month_str = target_date.strftime("%Y%m")
                    
                    # 计算TimeNum：对于按天存储的ERA5文件，TimeNum应该只基于HourUnique，范围[1, 24]
                    # 注意：ERA5文件按天存储，每个文件包含24小时数据，所以TimeNum应该在[1, 24]范围内
                    # 如果HourUnique超出[0, 23]范围，需要修正
                    hour_unique_clamped = max(0, min(23, HourUnique))
                    time_num_pressure = hour_unique_clamped + 1  # TimeNum = HourUnique + 1，范围[1, 24]
                    time_num_single = hour_unique_clamped + 1  # 单层数据同样处理
                    
                    if HourUnique != hour_unique_clamped:
                        logger.warning(f"GPU {gpu_id}: HourUnique={HourUnique}超出范围[0, 23]，已修正为{hour_unique_clamped}")
                    
                    logger.info(f"GPU {gpu_id}: 计算得到目标日期: {date_str}, 月份: {month_str}, "
                              f"TimeNum_pressure={time_num_pressure}, TimeNum_single={time_num_single} "
                              f"(DayUnique={DayUnique}, HourUnique={HourUnique})")
                    
                    era5_pressure_path = era5_pressure_paths_dict.get(date_str)
                    era5_single_path_info = era5_single_paths_dict.get(month_str)
                    
                    # 处理single level路径（可能是元组 (instant_path, accum_path) 或单个路径）
                    if isinstance(era5_single_path_info, tuple):
                        era5_single_instant_path, era5_single_accum_path = era5_single_path_info
                    else:
                        era5_single_instant_path = era5_single_path_info
                        era5_single_accum_path = None
                    
                    if not era5_pressure_path:
                        logger.warning(f"GPU {gpu_id}: 未找到日期 {date_str} 的ERA5 pressure level文件。"
                                     f"可用日期: {sorted(list(era5_pressure_paths_dict.keys()))[:10]}...")
                    if not era5_single_instant_path:
                        logger.warning(f"GPU {gpu_id}: 未找到月份 {month_str} 的ERA5 single level文件。"
                                     f"可用月份: {sorted(list(era5_single_paths_dict.keys()))}")
                    if era5_single_instant_path and not era5_single_accum_path:
                        logger.warning(f"GPU {gpu_id}: 未找到月份 {month_str} 的ERA5 accum文件（tp变量将缺失）")
                    
                    r_850hpa_2d = None
                    era5_pressure_2d = None
                    era5_tensor_2d = None
                    
                    # 按需加载pressure level数据
                    if era5_pressure_path:
                        # 验证TimeNum是否在合理范围内（单天24小时）
                        if time_num_pressure < 1 or time_num_pressure > 24:
                            logger.error(f"GPU {gpu_id}: TimeNum_pressure={time_num_pressure}超出单天24小时范围[1, 24]。"
                                       f"可能原因：GPM文件跨越午夜导致DayUnique计算错误。"
                                       f"DayUnique={DayUnique}, HourUnique={HourUnique}, "
                                       f"目标日期={date_str}。跳过pressure level数据加载，使用零数组占位")
                            # 创建零数组占位
                            if era5_lat is not None and era5_lon is not None:
                                era5_pressure_2d = torch.zeros((era5_lat.shape[0], era5_lon.shape[1], 6), 
                                                               dtype=torch.float32, device=device)
                                logger.warning(f"GPU {gpu_id}: 使用零数组占位ERA5 pressure level数据（6维）")
                            else:
                                logger.error(f"GPU {gpu_id}: 无法创建ERA5 pressure level占位数组（缺少经纬度信息）")
                        else:
                            try:
                                # 使用load_era5_hour_slice按需加载对应时间切片
                                logger.info(f"GPU {gpu_id}: 加载ERA5 pressure level文件: {os.path.basename(era5_pressure_path)}, "
                                          f"TimeNum={time_num_pressure}")
                                era5_pressure_data = load_era5_hour_slice(era5_pressure_path, time_num_pressure, 
                                                                          data_type='pressure', device=device)
                                # era5_pressure_data形状: (lat, lon, level, vars) = (157, 121, 27, 14)
                                # 根据MATLAB代码，匹配阶段应保留完整的27层数据，而不是求平均
                                # 将 (lat, lon, level, vars) 重塑为 (lat, lon, level*vars) = (157, 121, 378)
                                # 这样每个GPM像元匹配后会得到完整的27层×14变量=378维数据
                                lat_size, lon_size, level_size, var_size = era5_pressure_data.shape
                                era5_pressure_2d = era5_pressure_data.reshape(lat_size, lon_size, level_size * var_size)  # (157, 121, 378)
                                
                                # 提取850hPa相对湿度（从压力层数据中提取）
                                # 查找850hPa对应的level索引
                                with xr.open_dataset(era5_pressure_path) as ds:
                                    if 'pressure_level' in ds.coords:
                                        pressure_levels = ds.coords['pressure_level'].values
                                        level_850_idx = np.where(np.abs(pressure_levels - 850) < 1)[0]
                                        if len(level_850_idx) > 0:
                                            level_850_idx = level_850_idx[0]
                                        else:
                                            level_850_idx = 2  # 默认索引
                                    else:
                                        level_850_idx = 2
                                
                                if era5_pressure_data.shape[2] > level_850_idx:
                                    # 提取850hPa相对湿度（r变量在索引0）
                                    r_850hpa = era5_pressure_data[:, :, level_850_idx, 0]  # (157, 121) - 相对湿度
                                    r_850hpa_2d = r_850hpa.unsqueeze(-1)  # (157, 121, 1)
                                    logger.info(f"GPU {gpu_id}: 成功提取850hPa相对湿度，压力层索引: {level_850_idx}")
                                else:
                                    logger.warning(f"GPU {gpu_id}: 850hPa压力层索引{level_850_idx}超出范围")
                            except Exception as e:
                                error_msg = str(e)
                                logger.error(f"GPU {gpu_id}: 加载ERA5 pressure level数据失败: {error_msg}")
                                
                                # 分析失败原因
                                if "TimeNum" in error_msg and "超出" in error_msg:
                                    logger.error(f"GPU {gpu_id}: ⚠️ TimeNum超出范围错误！"
                                               f"文件: {os.path.basename(era5_pressure_path)}, "
                                               f"TimeNum={time_num_pressure}, "
                                               f"DayUnique={DayUnique}, HourUnique={HourUnique}, "
                                               f"目标日期={date_str}。"
                                               f"可能原因：GPM文件跨越午夜导致时间计算错误")
                                elif "文件不存在" in error_msg:
                                    logger.error(f"GPU {gpu_id}: ⚠️ ERA5文件不存在: {era5_pressure_path}")
                                elif "时间维度" in error_msg:
                                    logger.error(f"GPU {gpu_id}: ⚠️ ERA5文件缺少时间维度: {era5_pressure_path}")
                                else:
                                    logger.error(f"GPU {gpu_id}: ⚠️ 其他错误: {error_msg}", exc_info=True)
                                
                                # 创建零数组占位，确保维度一致（378维：27层×14变量）
                                if era5_lat is not None and era5_lon is not None:
                                    era5_pressure_2d = torch.zeros((era5_lat.shape[0], era5_lon.shape[1], 378), 
                                                                   dtype=torch.float32, device=device)
                                    logger.warning(f"GPU {gpu_id}: 使用零数组占位ERA5 pressure level数据（378维：27层×14变量）")
                                else:
                                    logger.error(f"GPU {gpu_id}: 无法创建ERA5 pressure level占位数组（缺少经纬度信息）")
                    else:
                        logger.warning(f"GPU {gpu_id}: 未找到日期 {date_str} 的ERA5 pressure level文件路径")
                    
                    # 按需加载single level数据（使用load_era5_hour_slice按需加载）
                    if era5_single_instant_path:
                        try:
                            # 使用load_era5_hour_slice按需加载对应时间切片
                            # 传递pressure_file_path参数，以便从压力层数据提取850hPa相对湿度
                            logger.info(f"GPU {gpu_id}: 加载ERA5 single level文件: {os.path.basename(era5_single_instant_path)}, "
                                      f"TimeNum={time_num_single}, accum文件: {os.path.basename(era5_single_accum_path) if era5_single_accum_path else 'None'}, "
                                      f"pressure文件: {os.path.basename(era5_pressure_path) if era5_pressure_path else 'None'}")
                            era5_single_2d = load_era5_hour_slice(era5_single_instant_path, time_num_single, 
                                                                   data_type='single', device=device,
                                                                   accum_file_path=era5_single_accum_path,
                                                                   pressure_file_path=era5_pressure_path)
                            # era5_single_2d形状: (lat, lon, vars) = (157, 121, 13)
                            # 850hPa相对湿度已从压力层数据提取并包含在era5_single_2d中（索引10）
                            logger.info(f"GPU {gpu_id}: ERA5 single level时间切片提取完成，形状: {era5_single_2d.shape}（已包含850hPa相对湿度）")
                            
                                # 合并pressure和single level数据
                            if era5_pressure_2d is not None:
                                # 同时有pressure和single level数据
                                # 根据MATLAB代码，pressure level保留完整的27层×6变量=162维
                                era5_tensor_2d = torch.cat([era5_pressure_2d, era5_single_2d], dim=-1)  # (157, 121, 162+13=175)
                                logger.info(f"GPU {gpu_id}: ERA5数据合并完成，维度: {era5_tensor_2d.shape} (pressure: 162 + single: 13 = 175)")
                            else:
                                # 只有single level数据，创建pressure level零数组占位（162维：27层×6变量）
                                if era5_lat is not None and era5_lon is not None:
                                    era5_pressure_placeholder = torch.zeros((era5_single_2d.shape[0], era5_single_2d.shape[1], 162), 
                                                                          dtype=era5_single_2d.dtype, device=era5_single_2d.device)
                                    era5_tensor_2d = torch.cat([era5_pressure_placeholder, era5_single_2d], dim=-1)  # (157, 121, 162+13=175)
                                    logger.warning(f"GPU {gpu_id}: ERA5 pressure level数据缺失，使用零数组占位，最终维度: {era5_tensor_2d.shape}")
                                else:
                                    # 如果连经纬度都没有，只能使用single level数据（不推荐）
                                    era5_tensor_2d = era5_single_2d
                                    logger.warning(f"GPU {gpu_id}: ERA5数据只有single level（13维），缺少pressure level数据，最终维度: {era5_tensor_2d.shape}")
                        except Exception as e:
                            logger.error(f"GPU {gpu_id}: 加载ERA5 single level数据失败: {str(e)}", exc_info=True)
                            # 如果single level也加载失败，但pressure level成功，则只使用pressure level数据
                            if era5_pressure_2d is not None:
                                # 创建single level零数组占位（13维）
                                if era5_lat is not None and era5_lon is not None:
                                    era5_single_placeholder = torch.zeros((era5_pressure_2d.shape[0], era5_pressure_2d.shape[1], 13), 
                                                                         dtype=era5_pressure_2d.dtype, device=era5_pressure_2d.device)
                                    era5_tensor_2d = torch.cat([era5_pressure_2d, era5_single_placeholder], dim=-1)  # (157, 121, 162+13=175)
                                    logger.warning(f"GPU {gpu_id}: ERA5 single level数据缺失，使用零数组占位，最终维度: {era5_tensor_2d.shape}")
                                else:
                                    era5_tensor_2d = era5_pressure_2d
                                    logger.warning(f"GPU {gpu_id}: ERA5数据只有pressure level（162维），缺少single level数据，最终维度: {era5_tensor_2d.shape}")
                    
                    if era5_tensor_2d is None:
                        logger.error(f"GPU {gpu_id}: 未找到可用的ERA5数据（日期: {date_str}, 月份: {month_str}），跳过该GPM文件")
                        continue
                    
                    # 验证ERA5数据维度（应该是400维：378+22，其中378=27层×14变量）
                    if era5_tensor_2d.shape[-1] != 400:
                        logger.warning(f"GPU {gpu_id}: ERA5数据维度异常，期望400维（pressure: 378 + single: 22），实际{era5_tensor_2d.shape[-1]}维")
                    
                    # 适配nearest_neighbor_match函数：GPM数据是1D数组，需要转换为2D格式
                    n_valid = gpm_tensor.shape[0]
                    gpm_tensor_2d = gpm_tensor.unsqueeze(1)  # (n_valid, 1, feat)
                    gpm_lat_2d = gpm_lat.unsqueeze(1)  # (n_valid, 1)
                    gpm_lon_2d = gpm_lon.unsqueeze(1)  # (n_valid, 1)
                    
                    # 匹配计算（使用GPM真实经纬度和ERA5对应时间切片）
                    logger.info(f"GPU {gpu_id}: 开始匹配计算...")
                    matched_raw = match_gpm_era5(gpm_tensor_2d, era5_tensor_2d, gpm_lat_2d, gpm_lon_2d, era5_lat, era5_lon)
                    
                    # 将匹配结果转换回1D格式
                    matched_raw = matched_raw.squeeze(1)  # (n_valid, feat)
                    logger.info(f"GPU {gpu_id}: 匹配完成 - 匹配数据形状: {matched_raw.shape}")
                    
                    # 验证matched_raw的维度（应该是447维：GPM(47) + ERA5_pressure(378) + ERA5_single(22)）
                    # 其中378=27层×14变量（z, o3, pv, r, ciwc, clwc, q, crwc, cswc, t, u, v, w, vo），保留完整的压力层数据
                    expected_dim = 447
                    actual_dim = matched_raw.shape[-1]
                    if actual_dim != expected_dim:
                        logger.error(f"GPU {gpu_id}: matched_raw维度错误！期望{expected_dim}维（GPM: 47 + ERA5_pressure: 378 + ERA5_single: 22），"
                                   f"实际{actual_dim}维。可能原因：ERA5数据加载不完整。跳过特征提取")
                        continue
                    
                    # 特征提取
                    logger.info(f"GPU {gpu_id}: 开始特征提取...")
                    
                    feature_set = extract_all_features(matched_raw)
                    
                    # 验证feature_set的维度（应该是51维）
                    if feature_set.shape[-1] != 51:
                        logger.error(f"GPU {gpu_id}: feature_set维度错误！期望51维，实际{feature_set.shape[-1]}维，跳过保存")
                        continue
                    
                    logger.info(f"GPU {gpu_id}: 特征提取完成 - 特征集形状: {feature_set.shape}")
                    
                    # 结果转CPU并放入队列（使用文件索引作为block_id的一部分）
                    # 为了区分同一block中的不同文件，使用block_id * 10000 + file_idx
                    # 注意：gpm_file_list包含(文件名, 目录路径)元组，需要正确提取索引
                    file_idx = 0
                    for idx, file_info in enumerate(gpm_file_list):
                        if isinstance(file_info, tuple):
                            file_name, _ = file_info
                            if file_name == gpm_file:
                                file_idx = idx
                                break
                        elif file_info == gpm_file:
                            file_idx = idx
                            break
                    unique_block_id = block_id * 10000 + file_idx
                    
                    result_queue.put({
                        "block_id": unique_block_id,
                        "matched_raw": matched_raw.cpu().numpy(),
                        "feature_set": feature_set.cpu().numpy()
                    })
                    
                    logger.info(f"GPU {gpu_id}: 完成GPM文件 {gpm_file} (block_id={unique_block_id})")
                    
                    # 释放显存
                    del gpm_tensor, gpm_tensor_2d, matched_raw, feature_set
                    torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"GPU {gpu_id}: 处理GPM文件 {gpm_file} 失败: {str(e)}", exc_info=True)
                    continue
            
            logger.info(f"GPU {gpu_id}: 完成Block {block_id}")
        
        except Exception as e:
            logger.error(f"GPU {gpu_id}: 处理Block {block_id} 失败: {str(e)}", exc_info=True)
            continue
    
    logger.info(f"GPU {gpu_id}: 所有任务完成")
