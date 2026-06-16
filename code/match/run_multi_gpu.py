import os
import sys
import logging
from datetime import datetime

# 将项目根目录加入Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.multi_gpu.task_distribute import multi_gpu_task_distribute
from src.io_utils.log_utils import init_logger
from src.io_utils.path_utils import create_dir
from src.multi_gpu.memory_control import check_gpu_memory
import torch


def load_project_config():
    """加载项目基础配置"""
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "base_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    """多GPU并行运行主入口"""
    # 初始化配置
    config = load_project_config()
    project_root = config["project_root"]
    log_dir = os.path.join(project_root, "temp", "logs")
    create_dir(log_dir)
    log_path = os.path.join(log_dir, f"multi_gpu_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # 初始化日志
    logger = init_logger(log_path)

    # 环境自检
    logger.info("=" * 50)
    logger.info(f"项目启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA版本: {torch.version.cuda}")
    logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"GPU{i}名称: {torch.cuda.get_device_name(i)}")
        check_gpu_memory(i)
    logger.info("=" * 50)

    # 启动多GPU任务分发
    try:
        multi_gpu_task_distribute()
        logger.info("多GPU任务执行完成！")
    except Exception as e:
        logger.error(f"多GPU任务执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()