"""
GPM-ERA5压力层匹配函数
"""
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def haversine_distance_km(lat1, lon1, lat2, lon2, earth_radius=6371.4):
    """
    计算两点间的大圆距离（单位：km）
    对应Matlab的distance函数，使用6371.4km作为地球半径
    """
    # 转换为弧度
    lat1_rad = np.deg2rad(lat1)
    lon1_rad = np.deg2rad(lon1)
    lat2_rad = np.deg2rad(lat2)
    lon2_rad = np.deg2rad(lon2)
    
    # Haversine公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = earth_radius * c
    
    return distance


def near_match_era5_pressure_gpm(latitude, longitude, day_unique, hour_unique, 
                                  day_start_number_era5_pressure,
                                  era5_lon, era5_lat, era5_level,
                                  era5_rh, era5_sh, era5_air_te, era5_vv, era5_vu, era5_vv_var):
    """
    最邻近匹配插值 - 压力层数据

    
    :param latitude: GPM纬度数组 (p, q) - p行q列
    :param longitude: GPM经度数组 (p, q)
    :param day_unique: 日（如1表示第1天）
    :param hour_unique: 小时（0-23）
    :param day_start_number_era5_pressure: ERA5压力层数据起始日编号（默认1）
    :param era5_lon: ERA5经度数组 (C,) - C是经度维度大小
    :param era5_lat: ERA5纬度数组 (R,) - R是纬度维度大小
    :param era5_level: ERA5压力层数组 (k,) - k=27层
    :param era5_rh: ERA5相对湿度 (C, R, k, TimeNum)
    :param era5_sh: ERA5比湿 (C, R, k, TimeNum)
    :param era5_air_te: ERA5温度 (C, R, k, TimeNum)
    :param era5_vv: ERA5垂直风速 (C, R, k, TimeNum)
    :param era5_vu: ERA5 u风分量 (C, R, k, TimeNum)
    :param era5_vv_var: ERA5 v风分量 (C, R, k, TimeNum)
    :return: RH, SH, AirTe, VV, Vu, Vv (都是 p, q, k)
    """
    # TimeNum计算
    time_num = (day_unique - day_start_number_era5_pressure) * 24 + hour_unique + 1
    
    # 检查TimeNum是否有效
    if time_num < 1 or time_num > era5_rh.shape[3]:
        raise ValueError(f"TimeNum={time_num}超出ERA5数据时间范围 [1, {era5_rh.shape[3]}]")
    
    k = len(era5_level)  # 27层
    p, q = latitude.shape  # GPM数据的行和列

    # 建立和DPR数据相同的数据结构（初始化为NaN，而非0）
    # 这样无效值和未匹配点都保持NaN，不会与有效值0混淆
    RH = np.full((p, q, k), np.nan)
    SH = np.full((p, q, k), np.nan)
    AirTe = np.full((p, q, k), np.nan)
    VV = np.full((p, q, k), np.nan)
    Vu = np.full((p, q, k), np.nan)
    Vv = np.full((p, q, k), np.nan)
    
    # 将ERA5经纬度转换为numpy数组（如果是Tensor）
    if isinstance(era5_lat, torch.Tensor):
        era5_lat = era5_lat.cpu().numpy()
    if isinstance(era5_lon, torch.Tensor):
        era5_lon = era5_lon.cpu().numpy()
    if isinstance(latitude, torch.Tensor):
        latitude = latitude.cpu().numpy()
    if isinstance(longitude, torch.Tensor):
        longitude = longitude.cpu().numpy()
    
    # 将ERA5数据转换为numpy（如果是Tensor）
    era5_rh_np = era5_rh.cpu().numpy() if isinstance(era5_rh, torch.Tensor) else era5_rh
    era5_sh_np = era5_sh.cpu().numpy() if isinstance(era5_sh, torch.Tensor) else era5_sh
    era5_air_te_np = era5_air_te.cpu().numpy() if isinstance(era5_air_te, torch.Tensor) else era5_air_te
    era5_vv_np = era5_vv.cpu().numpy() if isinstance(era5_vv, torch.Tensor) else era5_vv
    era5_vu_np = era5_vu.cpu().numpy() if isinstance(era5_vu, torch.Tensor) else era5_vu
    era5_vv_var_np = era5_vv_var.cpu().numpy() if isinstance(era5_vv_var, torch.Tensor) else era5_vv_var
    
    logger.info(f"压力层匹配：TimeNum={time_num}, GPM数据形状=({p},{q}), ERA5层数={k}")
    
    # 严格按照Matlab代码的双重循环
    for i in range(p):
        for j in range(q):  # GPM位置循环
            tempM = []
            tempN = []
            
            # 快速寻找到DPR像元5km范围内的所有ERA5像元
            MaxLat = latitude[i, j] + 0.2
            MinLat = latitude[i, j] - 0.2
            MaxLon = longitude[i, j] + 0.2
            MinLon = longitude[i, j] - 0.2
            
            # 找到满足条件的ERA5像元索引
            r = np.where(np.abs(latitude[i, j] - era5_lat) < 0.2)[0]
            c = np.where(np.abs(longitude[i, j] - era5_lon) < 0.2)[0]
            
            # 进一步筛选：在±0.2°范围内且满足边界条件
            for m in range(len(r)):
                for n in range(len(c)):
                    if (era5_lat[r[m]] < MaxLat and era5_lat[r[m]] > MinLat and 
                        era5_lon[c[n]] < MaxLon and era5_lon[c[n]] > MinLon):
                        tempM.append(r[m])
                        tempN.append(c[n])
            
            # 将同一DPR像元周围所有ERA5像元进行最邻近匹配
            if len(tempM) > 0 and len(tempN) > 0:
                if len(tempM) == len(tempN):
                    aa = len(tempM)  # 范围内ERA5像元的个数
                    Distmp = np.zeros(aa)
                    
                    # 计算距离
                    for ii in range(aa):
                        Distmp[ii] = haversine_distance_km(
                            latitude[i, j], longitude[i, j],
                            era5_lat[tempM[ii]], era5_lon[tempN[ii]],
                            6371.4  # 地球半径，单位：km
                        )
                    
                    # 找到最近的点
                    point_num = np.where(Distmp == np.min(Distmp))[0]
                    if len(point_num) > 0:
                        R = tempM[point_num[0]]  # 纬度索引
                        C = tempN[point_num[0]]  # 经度索引
                        
                        # 赋值给输出矩阵（严格按照Matlab索引顺序：C,R,Level,TimeNum）
                        # 注意：Matlab是1-based，Python是0-based，但TimeNum已经是1-based，需要-1
                        RH[i, j, :] = era5_rh_np[C, R, :, time_num - 1]
                        SH[i, j, :] = era5_sh_np[C, R, :, time_num - 1]
                        AirTe[i, j, :] = era5_air_te_np[C, R, :, time_num - 1]
                        VV[i, j, :] = era5_vv_np[C, R, :, time_num - 1]
                        Vu[i, j, :] = era5_vu_np[C, R, :, time_num - 1]
                        Vv[i, j, :] = era5_vv_var_np[C, R, :, time_num - 1]
    
    return RH, SH, AirTe, VV, Vu, Vv


def near_match_era5_pressure_gpm_full(latitude, longitude, day_unique, hour_unique,
                                       day_start_number_era5_pressure,
                                       era5_lon, era5_lat, era5_level,
                                       era5_z, era5_o3, era5_pv, era5_r, era5_ciwc, era5_clwc,
                                       era5_q, era5_crwc, era5_cswc, era5_t, era5_u, era5_v, era5_w, era5_vo):
    """
    最邻近匹配插值 - 压力层数据（完整14变量版本）

    返回14个变量的匹配结果：(Z, O3, PV, R, CIWC, CLWC, Q, CRWC, CSWC, T, U, V, W, VO)
    每个都是 (p, q, k) 形状
    """
    time_num = (day_unique - day_start_number_era5_pressure) * 24 + hour_unique + 1

    if time_num < 1 or time_num > era5_r.shape[3]:
        raise ValueError(f"TimeNum={time_num}超出ERA5数据时间范围 [1, {era5_r.shape[3]}]")

    k = len(era5_level)
    p, q = latitude.shape

    # 初始化14个输出数组为NaN（而非0），确保无效值和未匹配点不会与有效值0混淆
    Z = np.full((p, q, k), np.nan)
    O3 = np.full((p, q, k), np.nan)
    PV = np.full((p, q, k), np.nan)
    R = np.full((p, q, k), np.nan)
    CIWC = np.full((p, q, k), np.nan)
    CLWC = np.full((p, q, k), np.nan)
    Q = np.full((p, q, k), np.nan)
    CRWC = np.full((p, q, k), np.nan)
    CSWC = np.full((p, q, k), np.nan)
    T = np.full((p, q, k), np.nan)
    U = np.full((p, q, k), np.nan)
    V_var = np.full((p, q, k), np.nan)
    W = np.full((p, q, k), np.nan)
    VO = np.full((p, q, k), np.nan)

    # 转换为numpy
    if isinstance(era5_lat, torch.Tensor):
        era5_lat = era5_lat.cpu().numpy()
    if isinstance(era5_lon, torch.Tensor):
        era5_lon = era5_lon.cpu().numpy()

    era5_vars_np = [
        era5_z.cpu().numpy() if isinstance(era5_z, torch.Tensor) else era5_z,
        era5_o3.cpu().numpy() if isinstance(era5_o3, torch.Tensor) else era5_o3,
        era5_pv.cpu().numpy() if isinstance(era5_pv, torch.Tensor) else era5_pv,
        era5_r.cpu().numpy() if isinstance(era5_r, torch.Tensor) else era5_r,
        era5_ciwc.cpu().numpy() if isinstance(era5_ciwc, torch.Tensor) else era5_ciwc,
        era5_clwc.cpu().numpy() if isinstance(era5_clwc, torch.Tensor) else era5_clwc,
        era5_q.cpu().numpy() if isinstance(era5_q, torch.Tensor) else era5_q,
        era5_crwc.cpu().numpy() if isinstance(era5_crwc, torch.Tensor) else era5_crwc,
        era5_cswc.cpu().numpy() if isinstance(era5_cswc, torch.Tensor) else era5_cswc,
        era5_t.cpu().numpy() if isinstance(era5_t, torch.Tensor) else era5_t,
        era5_u.cpu().numpy() if isinstance(era5_u, torch.Tensor) else era5_u,
        era5_v.cpu().numpy() if isinstance(era5_v, torch.Tensor) else era5_v,
        era5_w.cpu().numpy() if isinstance(era5_w, torch.Tensor) else era5_w,
        era5_vo.cpu().numpy() if isinstance(era5_vo, torch.Tensor) else era5_vo,
    ]

    # 双重循环匹配（添加进度日志）
    total_pixels = p * q
    logger.info(f"开始匹配: 总像元数={total_pixels} ({p}×{q}), ERA5网格=({len(era5_lat)}, {len(era5_lon)})")
    
    for i in range(p):
        # 每100行打印一次进度
        if i % 100 == 0:
            progress = (i * q) / total_pixels * 100
            logger.info(f"匹配进度: {i}/{p} 行 ({progress:.1f}%)")
        
        for j in range(q):
            MaxLat = latitude[i, j] + 0.2
            MinLat = latitude[i, j] - 0.2
            MaxLon = longitude[i, j] + 0.2
            MinLon = longitude[i, j] - 0.2

            r_idx = np.where(np.abs(latitude[i, j] - era5_lat) < 0.2)[0]
            c_idx = np.where(np.abs(longitude[i, j] - era5_lon) < 0.2)[0]

            tempM = []
            tempN = []
            for m in range(len(r_idx)):
                for n in range(len(c_idx)):
                    if (era5_lat[r_idx[m]] < MaxLat and era5_lat[r_idx[m]] > MinLat and
                        era5_lon[c_idx[n]] < MaxLon and era5_lon[c_idx[n]] > MinLon):
                        tempM.append(r_idx[m])
                        tempN.append(c_idx[n])

            if len(tempM) > 0 and len(tempN) > 0:
                aa = len(tempM)
                Distmp = np.zeros(aa)
                for ii in range(aa):
                    Distmp[ii] = haversine_distance_km(
                        latitude[i, j], longitude[i, j],
                        era5_lat[tempM[ii]], era5_lon[tempN[ii]],
                        6371.4
                    )

                point_num = np.where(Distmp == np.min(Distmp))[0]
                if len(point_num) > 0:
                    R_idx = tempM[point_num[0]]
                    C_idx = tempN[point_num[0]]

                    # 赋值14个变量
                    Z[i, j, :] = era5_vars_np[0][C_idx, R_idx, :, time_num - 1]
                    O3[i, j, :] = era5_vars_np[1][C_idx, R_idx, :, time_num - 1]
                    PV[i, j, :] = era5_vars_np[2][C_idx, R_idx, :, time_num - 1]
                    R[i, j, :] = era5_vars_np[3][C_idx, R_idx, :, time_num - 1]
                    CIWC[i, j, :] = era5_vars_np[4][C_idx, R_idx, :, time_num - 1]
                    CLWC[i, j, :] = era5_vars_np[5][C_idx, R_idx, :, time_num - 1]
                    Q[i, j, :] = era5_vars_np[6][C_idx, R_idx, :, time_num - 1]
                    CRWC[i, j, :] = era5_vars_np[7][C_idx, R_idx, :, time_num - 1]
                    CSWC[i, j, :] = era5_vars_np[8][C_idx, R_idx, :, time_num - 1]
                    T[i, j, :] = era5_vars_np[9][C_idx, R_idx, :, time_num - 1]
                    U[i, j, :] = era5_vars_np[10][C_idx, R_idx, :, time_num - 1]
                    V_var[i, j, :] = era5_vars_np[11][C_idx, R_idx, :, time_num - 1]
                    W[i, j, :] = era5_vars_np[12][C_idx, R_idx, :, time_num - 1]
                    VO[i, j, :] = era5_vars_np[13][C_idx, R_idx, :, time_num - 1]

    logger.info(f"匹配完成: 处理了 {total_pixels} 个像元")
    return Z, O3, PV, R, CIWC, CLWC, Q, CRWC, CSWC, T, U, V_var, W, VO


