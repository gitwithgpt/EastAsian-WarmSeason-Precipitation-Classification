"""
GPM-ERA5单层匹配函数

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


def near_match_era5_single_gpm(latitude, longitude, day_unique, hour_unique,
                                era5_lon, era5_lat,
                                era5_cape, era5_cin, era5_kindex, era5_te2m, era5_sp, era5_tp,
                                era5_zero_h, era5_tclw, era5_tciw, era5_tcw, era5_tcwv):
    """
    最邻近匹配插值 - 单层数据
    
    :param latitude: GPM纬度数组 (p, q)
    :param longitude: GPM经度数组 (p, q)
    :param day_unique: 日（如1表示第1天）
    :param hour_unique: 小时（0-23）
    :param era5_lon: ERA5经度数组 (C,)
    :param era5_lat: ERA5纬度数组 (R,)
    :param era5_cape: ERA5 CAPE (C, R, TimeNum)
    :param era5_cin: ERA5 CIN (C, R, TimeNum)
    :param era5_kindex: ERA5 Kindex (C, R, TimeNum)
    :param era5_te2m: ERA5 2m温度 (C, R, TimeNum)
    :param era5_sp: ERA5 表面气压 (C, R, TimeNum)
    :param era5_tp: ERA5 总降水 (C, R, TimeNum)
    :param era5_zero_h: ERA5 ZeroH (C, R, TimeNum)
    :param era5_tclw: ERA5 TCLW (C, R, TimeNum)
    :param era5_tciw: ERA5 TCIW (C, R, TimeNum)
    :param era5_tcw: ERA5 TCW (C, R, TimeNum)
    :param era5_tcwv: ERA5 TCWV (C, R, TimeNum)
    :return: CAPE, CIN, Kindex, Te2m, SP, TP, ZeroH, TCLW, TCIW, TCW, TCWV (都是 p, q)
    """
    # TimeNum计算：严格按照Matlab代码
    time_num = (day_unique - 1) * 24 + hour_unique + 1
    
    # 检查TimeNum是否有效
    if time_num < 1 or time_num > era5_cape.shape[2]:
        raise ValueError(f"TimeNum={time_num}超出ERA5数据时间范围 [1, {era5_cape.shape[2]}]")
    
    p, q = latitude.shape  # GPM数据的行和列

    # 建立和DPR数据相同的数据结构（初始化为NaN，而非0）
    # 这样无效值和未匹配点都保持NaN，不会与有效值0混淆
    CAPE = np.full((p, q), np.nan)
    CIN = np.full((p, q), np.nan)
    Kindex = np.full((p, q), np.nan)
    Te2m = np.full((p, q), np.nan)
    SP = np.full((p, q), np.nan)
    TP = np.full((p, q), np.nan)
    ZeroH = np.full((p, q), np.nan)
    TCLW = np.full((p, q), np.nan)
    TCIW = np.full((p, q), np.nan)
    TCW = np.full((p, q), np.nan)
    TCWV = np.full((p, q), np.nan)
    
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
    era5_cape_np = era5_cape.cpu().numpy() if isinstance(era5_cape, torch.Tensor) else era5_cape
    era5_cin_np = era5_cin.cpu().numpy() if isinstance(era5_cin, torch.Tensor) else era5_cin
    era5_kindex_np = era5_kindex.cpu().numpy() if isinstance(era5_kindex, torch.Tensor) else era5_kindex
    era5_te2m_np = era5_te2m.cpu().numpy() if isinstance(era5_te2m, torch.Tensor) else era5_te2m
    era5_sp_np = era5_sp.cpu().numpy() if isinstance(era5_sp, torch.Tensor) else era5_sp
    era5_tp_np = era5_tp.cpu().numpy() if isinstance(era5_tp, torch.Tensor) else era5_tp
    era5_zero_h_np = era5_zero_h.cpu().numpy() if isinstance(era5_zero_h, torch.Tensor) else era5_zero_h
    era5_tclw_np = era5_tclw.cpu().numpy() if isinstance(era5_tclw, torch.Tensor) else era5_tclw
    era5_tciw_np = era5_tciw.cpu().numpy() if isinstance(era5_tciw, torch.Tensor) else era5_tciw
    era5_tcw_np = era5_tcw.cpu().numpy() if isinstance(era5_tcw, torch.Tensor) else era5_tcw
    era5_tcwv_np = era5_tcwv.cpu().numpy() if isinstance(era5_tcwv, torch.Tensor) else era5_tcwv
    
    logger.info(f"单层匹配：TimeNum={time_num}, GPM数据形状=({p},{q})")
    
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
            
            # 将同一DPR像元周围最近的ERA5像元进行赋值
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
                        
                        # 赋值给输出矩阵（严格按照Matlab索引顺序：C,R,TimeNum）
                        # 注意：Matlab是1-based，Python是0-based，但TimeNum已经是1-based，需要-1
                        CAPE[i, j] = era5_cape_np[C, R, time_num - 1]
                        CIN[i, j] = era5_cin_np[C, R, time_num - 1]
                        Kindex[i, j] = era5_kindex_np[C, R, time_num - 1]
                        Te2m[i, j] = era5_te2m_np[C, R, time_num - 1]
                        SP[i, j] = era5_sp_np[C, R, time_num - 1]
                        TP[i, j] = era5_tp_np[C, R, time_num - 1]
                        ZeroH[i, j] = era5_zero_h_np[C, R, time_num - 1]
                        TCLW[i, j] = era5_tclw_np[C, R, time_num - 1]
                        TCIW[i, j] = era5_tciw_np[C, R, time_num - 1]
                        TCW[i, j] = era5_tcw_np[C, R, time_num - 1]
                        TCWV[i, j] = era5_tcwv_np[C, R, time_num - 1]
    
    return CAPE, CIN, Kindex, Te2m, SP, TP, ZeroH, TCLW, TCIW, TCW, TCWV


def near_match_era5_single_gpm_full(latitude, longitude, day_unique, hour_unique,
                                     era5_lon, era5_lat,
                                     era5_u10, era5_v10, era5_d2m, era5_t2m, era5_sp, era5_cbh,
                                     era5_tcc, era5_tciw, era5_tclw_s, era5_crr, era5_ptype, era5_tcrw,
                                     era5_blh, era5_cape, era5_cin, era5_z_surf, era5_kx, era5_tcw, era5_tcwv,
                                     era5_deg0l, era5_rh850, era5_tp):
    """
    最邻近匹配插值 - 单层数据（完整22变量版本）

    返回22个变量的匹配结果：(U10, V10, D2M, T2M, SP, CBH, TCC, TCIW, TCLW, CRR, PTYPE, TCRW,
                              BLH, CAPE, CIN, Z_SURF, KX, TCW, TCWV, DEG0L, RH850, TP)
    每个都是 (p, q) 形状
    """
    time_num = (day_unique - 1) * 24 + hour_unique + 1

    if time_num < 1 or time_num > era5_cape.shape[2]:
        raise ValueError(f"TimeNum={time_num}超出ERA5数据时间范围 [1, {era5_cape.shape[2]}]")

    p, q = latitude.shape

    # 初始化22个输出数组为NaN（而非0），确保无效值和未匹配点不会与有效值0混淆
    U10 = np.full((p, q), np.nan)
    V10 = np.full((p, q), np.nan)
    D2M = np.full((p, q), np.nan)
    T2M = np.full((p, q), np.nan)
    SP = np.full((p, q), np.nan)
    CBH = np.full((p, q), np.nan)
    TCC = np.full((p, q), np.nan)
    TCIW = np.full((p, q), np.nan)
    TCLW = np.full((p, q), np.nan)
    CRR = np.full((p, q), np.nan)
    PTYPE = np.full((p, q), np.nan)
    TCRW = np.full((p, q), np.nan)
    BLH = np.full((p, q), np.nan)
    CAPE = np.full((p, q), np.nan)
    CIN = np.full((p, q), np.nan)
    Z_SURF = np.full((p, q), np.nan)
    KX = np.full((p, q), np.nan)
    TCW = np.full((p, q), np.nan)
    TCWV = np.full((p, q), np.nan)
    DEG0L = np.full((p, q), np.nan)
    RH850 = np.full((p, q), np.nan)
    TP = np.full((p, q), np.nan)

    # 转换为numpy
    if isinstance(era5_lat, torch.Tensor):
        era5_lat = era5_lat.cpu().numpy()
    if isinstance(era5_lon, torch.Tensor):
        era5_lon = era5_lon.cpu().numpy()

    era5_vars_np = [
        era5_u10.cpu().numpy() if isinstance(era5_u10, torch.Tensor) else era5_u10,
        era5_v10.cpu().numpy() if isinstance(era5_v10, torch.Tensor) else era5_v10,
        era5_d2m.cpu().numpy() if isinstance(era5_d2m, torch.Tensor) else era5_d2m,
        era5_t2m.cpu().numpy() if isinstance(era5_t2m, torch.Tensor) else era5_t2m,
        era5_sp.cpu().numpy() if isinstance(era5_sp, torch.Tensor) else era5_sp,
        era5_cbh.cpu().numpy() if isinstance(era5_cbh, torch.Tensor) else era5_cbh,
        era5_tcc.cpu().numpy() if isinstance(era5_tcc, torch.Tensor) else era5_tcc,
        era5_tciw.cpu().numpy() if isinstance(era5_tciw, torch.Tensor) else era5_tciw,
        era5_tclw_s.cpu().numpy() if isinstance(era5_tclw_s, torch.Tensor) else era5_tclw_s,
        era5_crr.cpu().numpy() if isinstance(era5_crr, torch.Tensor) else era5_crr,
        era5_ptype.cpu().numpy() if isinstance(era5_ptype, torch.Tensor) else era5_ptype,
        era5_tcrw.cpu().numpy() if isinstance(era5_tcrw, torch.Tensor) else era5_tcrw,
        era5_blh.cpu().numpy() if isinstance(era5_blh, torch.Tensor) else era5_blh,
        era5_cape.cpu().numpy() if isinstance(era5_cape, torch.Tensor) else era5_cape,
        era5_cin.cpu().numpy() if isinstance(era5_cin, torch.Tensor) else era5_cin,
        era5_z_surf.cpu().numpy() if isinstance(era5_z_surf, torch.Tensor) else era5_z_surf,
        era5_kx.cpu().numpy() if isinstance(era5_kx, torch.Tensor) else era5_kx,
        era5_tcw.cpu().numpy() if isinstance(era5_tcw, torch.Tensor) else era5_tcw,
        era5_tcwv.cpu().numpy() if isinstance(era5_tcwv, torch.Tensor) else era5_tcwv,
        era5_deg0l.cpu().numpy() if isinstance(era5_deg0l, torch.Tensor) else era5_deg0l,
        era5_rh850.cpu().numpy() if isinstance(era5_rh850, torch.Tensor) else era5_rh850,
        era5_tp.cpu().numpy() if isinstance(era5_tp, torch.Tensor) else era5_tp,
    ]

    # 双重循环匹配（添加进度日志）
    total_pixels = p * q
    logger.info(f"单层匹配: 总像元数={total_pixels} ({p}×{q})")
    
    for i in range(p):
        # 每100行打印一次进度
        if i % 100 == 0:
            progress = (i * q) / total_pixels * 100
            logger.info(f"单层匹配进度: {i}/{p} 行 ({progress:.1f}%)")
        
        for j in range(q):
            MaxLat = latitude[i, j] + 0.2
            MinLat = latitude[i, j] - 0.2
            MaxLon = longitude[i, j] + 0.2
            MinLon = longitude[i, j] - 0.2

            r = np.where(np.abs(latitude[i, j] - era5_lat) < 0.2)[0]
            c = np.where(np.abs(longitude[i, j] - era5_lon) < 0.2)[0]

            tempM = []
            tempN = []
            for m in range(len(r)):
                for n in range(len(c)):
                    if (era5_lat[r[m]] < MaxLat and era5_lat[r[m]] > MinLat and
                        era5_lon[c[n]] < MaxLon and era5_lon[c[n]] > MinLon):
                        tempM.append(r[m])
                        tempN.append(c[n])

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
                    R = tempM[point_num[0]]
                    C = tempN[point_num[0]]

                    # 赋值22个变量
                    U10[i, j] = era5_vars_np[0][C, R, time_num - 1]
                    V10[i, j] = era5_vars_np[1][C, R, time_num - 1]
                    D2M[i, j] = era5_vars_np[2][C, R, time_num - 1]
                    T2M[i, j] = era5_vars_np[3][C, R, time_num - 1]
                    SP[i, j] = era5_vars_np[4][C, R, time_num - 1]
                    CBH[i, j] = era5_vars_np[5][C, R, time_num - 1]
                    TCC[i, j] = era5_vars_np[6][C, R, time_num - 1]
                    TCIW[i, j] = era5_vars_np[7][C, R, time_num - 1]
                    TCLW[i, j] = era5_vars_np[8][C, R, time_num - 1]
                    CRR[i, j] = era5_vars_np[9][C, R, time_num - 1]
                    PTYPE[i, j] = era5_vars_np[10][C, R, time_num - 1]
                    TCRW[i, j] = era5_vars_np[11][C, R, time_num - 1]
                    BLH[i, j] = era5_vars_np[12][C, R, time_num - 1]
                    CAPE[i, j] = era5_vars_np[13][C, R, time_num - 1]
                    CIN[i, j] = era5_vars_np[14][C, R, time_num - 1]
                    Z_SURF[i, j] = era5_vars_np[15][C, R, time_num - 1]
                    KX[i, j] = era5_vars_np[16][C, R, time_num - 1]
                    TCW[i, j] = era5_vars_np[17][C, R, time_num - 1]
                    TCWV[i, j] = era5_vars_np[18][C, R, time_num - 1]
                    DEG0L[i, j] = era5_vars_np[19][C, R, time_num - 1]
                    RH850[i, j] = era5_vars_np[20][C, R, time_num - 1]
                    TP[i, j] = era5_vars_np[21][C, R, time_num - 1]

    logger.info(f"单层匹配完成: 处理了 {total_pixels} 个像元")
    return (U10, V10, D2M, T2M, SP, CBH, TCC, TCIW, TCLW, CRR, PTYPE, TCRW,
            BLH, CAPE, CIN, Z_SURF, KX, TCW, TCWV, DEG0L, RH850, TP)


