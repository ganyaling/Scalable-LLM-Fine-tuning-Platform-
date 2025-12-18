#!/usr/bin/env python
"""
Celery Worker 启动脚本
在单独的终端中运行此脚本以启动 Celery Worker
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

from celery_app import app

if __name__ == '__main__':
    app.worker_main([
        'worker',
        '--loglevel=info',
        '--concurrency=1',  # 单个 Worker，避免并行训练导致 OOM
        '-Q', 'training,default',  # 监听的队列（改为 -Q）
        '--max-tasks-per-child=1',  # 每个任务后重启 Worker，防止内存泄漏
    ])
