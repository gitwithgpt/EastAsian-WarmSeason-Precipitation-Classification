import torch


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算球面哈维正弦距离（km）
    :param lat1: 纬度1 (Tensor)
    :param lon1: 经度1 (Tensor)
    :param lat2: 纬度2 (Tensor)
    :param lon2: 经度2 (Tensor)
    :return: 距离数组 (km)
    """
    # 转换为弧度
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)

    # 哈维正弦公式
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    r = 6371.0  # 地球半径(km)
    return r * c