import torch
import logging
import gc


def get_logger():
    """获取日志器"""
    return logging.getLogger(__name__)


def check_gpu_memory(gpu_id, max_mem_GB=18.0):
    """
    检查GPU显存占用，超过阈值则释放
    :param gpu_id: GPU编号
    :param max_mem_GB: 最大允许显存占用(GB)
    """
    torch.cuda.set_device(gpu_id)
    max_mem = max_mem_GB * 1024 ** 3  # 转换为字节
    used_mem = torch.cuda.memory_allocated()

    logger = get_logger()
    if used_mem > max_mem:
        logger.warning(f"GPU {gpu_id} memory usage ({used_mem / 1024 ** 3:.1f}GB) exceeds threshold ({max_mem_GB}GB)")
        torch.cuda.empty_cache()
        gc.collect()

    return used_mem