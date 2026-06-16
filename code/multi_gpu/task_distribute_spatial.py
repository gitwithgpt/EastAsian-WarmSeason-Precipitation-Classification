import os
import torch
import logging
import torch.multiprocessing as mp
import numpy as np
from yaml import safe_load

from .gpu_worker_spatial import single_gpu_worker_spatial
from src.io_utils.npz_writer_spatial import batch_write_npz_spatial
from src.io_utils.log_utils import get_unfinished_blocks


def load_config():
    """加载配置文件（与原 task_distribute 一致）"""
    base_config = safe_load(
        open(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "base_config.yaml",
            ),
            "r",
            encoding="utf-8",
        )
    )
    gpu_config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "config",
        "multi_gpu_config.yaml",
    )
    if os.path.exists(gpu_config_path):
        gpu_config = safe_load(open(gpu_config_path, "r", encoding="utf-8"))
        base_config.update(gpu_config)
    return base_config


def get_logger():
    return logging.getLogger(__name__)


from .task_distribute import split_data_into_blocks  # 直接复用原分块逻辑


def multi_gpu_task_distribute_spatial():
    """
    多GPU任务分发（带经纬度输出版本）。
    逻辑与 src.multi_gpu.task_distribute.multi_gpu_task_distribute 类似，
    但结果写入使用 batch_write_npz_spatial，并附加经纬度。
    """
    config = load_config()
    logger = get_logger()
    num_gpus = config["gpu"]["num_gpus"]
    batch_size = config["gpu"]["batch_write_size"]

    mp.set_start_method(config["gpu"]["start_method"], force=True)

    # 生成数据块并分配
    all_blocks = split_data_into_blocks()
    gpu_blocks = {i: [] for i in range(num_gpus)}
    for idx, block in enumerate(all_blocks):
        gpu_blocks[idx % num_gpus].append(block)

    # 断点恢复（沿用原逻辑）
    unfinished_blocks = get_unfinished_blocks()
    if unfinished_blocks is not None:
        for gid in gpu_blocks:
            gpu_blocks[gid] = [
                b for b in gpu_blocks[gid] if b["block_id"] not in unfinished_blocks
            ]

    # 结果队列
    result_queue = mp.Queue(maxsize=batch_size * 2)

    # 启动 worker
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=single_gpu_worker_spatial,
            args=(gpu_id, gpu_blocks[gpu_id], result_queue),
        )
        p.daemon = True
        p.start()
        processes.append(p)
        logger.info(f"[SPATIAL] Started process for GPU {gpu_id}")

    raw_buffer = []
    feat_buffer = []
    finished_procs = 0
    total_results = 0

    while finished_procs < num_gpus:
        try:
            res = result_queue.get(timeout=1.0)
            # res 已经包含 block_id, matched_raw, feature_set, lat, lon
            raw_buffer.append(res)
            feat_buffer.append(res)
            total_results += 1

            if len(raw_buffer) >= batch_size:
                batch_write_npz_spatial(raw_buffer, "matched_raw", config)
                batch_write_npz_spatial(feat_buffer, "feature_set", config)
                logger.info(
                    f"[SPATIAL] 已写入 {len(raw_buffer)} 个结果块（带经纬度），累计: {total_results} 个"
                )
                raw_buffer.clear()
                feat_buffer.clear()
        except Exception:
            # 队列超时，检查进程状态
            pass

        # 检查进程状态
        for p in list(processes):
            if not p.is_alive():
                finished_procs += 1
                processes.remove(p)
                logger.info(
                    f"[SPATIAL] GPU 进程已完成，剩余: {num_gpus - finished_procs} 个进程"
                )

    # 写入剩余
    if raw_buffer:
        batch_write_npz_spatial(raw_buffer, "matched_raw", config)
        batch_write_npz_spatial(feat_buffer, "feature_set", config)
        logger.info(f"[SPATIAL] 写入最后 {len(raw_buffer)} 个结果块")

    logger.info(
        f"[SPATIAL] 所有 GPU 进程结束，总共处理 {total_results} 个结果块，Spatial 结果已保存。"
    )


