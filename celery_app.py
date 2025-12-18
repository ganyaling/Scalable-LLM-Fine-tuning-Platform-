"""
Celery 应用配置
用于管理异步训练任务
"""
from celery import Celery
from kombu import Queue, Exchange
import os

# 创建 Celery 应用实例
app = Celery('mini_llm_studio')

# 配置 Celery
app.conf.update(
    # Broker 配置（使用 Redis）
    broker_url=os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    # Result backend 配置
    result_backend=os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'),
    # 任务序列化配置
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    # 任务配置
    task_track_started=True,
    task_time_limit=24 * 3600,  # 24小时硬超时
    task_soft_time_limit=23 * 3600,  # 23小时软超时
    # Worker 配置
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1,
    # 队列配置
    task_queues=(
        Queue('training', Exchange('training'), routing_key='training'),
        Queue('default', Exchange('default'), routing_key='default'),
    ),
    task_default_queue='default',
    task_routes={
        'tasks.train_task': {'queue': 'training'},
    }
)

if __name__ == '__main__':
    app.start()
