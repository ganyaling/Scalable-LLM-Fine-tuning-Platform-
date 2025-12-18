"""
GPU 资源锁定机制
支持通过 Redis 或文件锁实现同一时间只有固定数量的任务运行
"""
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import os

# 尝试导入 Redis，如果不可用则使用文件锁
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class GPUResourceLock:
    """GPU 资源锁定管理器"""
    
    def __init__(self, max_concurrent_tasks: int = 1, lock_dir: str = "./data/locks"):
        """
        初始化 GPU 资源锁
        
        Args:
            max_concurrent_tasks: 最大并发任务数（默认 1，确保同时只有 1 个任务用 GPU）
            lock_dir: 锁文件目录
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.lock_dir = Path(lock_dir)
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 Redis 客户端（如果可用）
        self.redis_client = None
        self.use_redis = False
        self._init_redis()
    
    def _init_redis(self):
        """初始化 Redis 连接"""
        if not REDIS_AVAILABLE:
            print("⚠️  Redis 不可用，使用文件锁方式")
            return
        
        try:
            redis_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url)
            # 测试连接
            self.redis_client.ping()
            self.use_redis = True
            print("✓ GPU 资源锁使用 Redis 方式")
        except Exception as e:
            print(f"⚠️  Redis 连接失败: {e}，使用文件锁方式")
            self.use_redis = False
    
    def acquire_lock(self, run_id: str, timeout: int = 86400) -> bool:
        """
        获取 GPU 资源锁
        
        Args:
            run_id: 任务 ID
            timeout: 锁超时时间（秒），默认 24 小时
        
        Returns:
            是否成功获取锁
        """
        if self.use_redis:
            return self._acquire_redis_lock(run_id, timeout)
        else:
            return self._acquire_file_lock(run_id, timeout)
    
    def _acquire_redis_lock(self, run_id: str, timeout: int) -> bool:
        """使用 Redis 获取锁"""
        try:
            lock_key = f"gpu:lock:{run_id}"
            queue_key = "gpu:queue"
            
            # 检查当前运行的任务数
            running_tasks = self.redis_client.smembers("gpu:running")
            
            if len(running_tasks) < self.max_concurrent_tasks:
                # 可以立即获取锁
                self.redis_client.setex(lock_key, timeout, "running")
                self.redis_client.sadd("gpu:running", run_id)
                self.redis_client.hset("gpu:lock_info", run_id, json.dumps({
                    'acquired_at': datetime.now().isoformat(),
                    'timeout': timeout
                }))
                return True
            else:
                # 加入等待队列
                self.redis_client.lpush(queue_key, run_id)
                return False
        except Exception as e:
            print(f"Redis 锁获取错误: {e}")
            return False
    
    def _acquire_file_lock(self, run_id: str, timeout: int) -> bool:
        """使用文件锁获取锁"""
        # 检查已过期的锁
        self._cleanup_expired_locks()
        
        # 统计当前运行的任务
        running_tasks = list(self.lock_dir.glob("*.lock"))
        
        if len(running_tasks) < self.max_concurrent_tasks:
            # 可以获取锁
            lock_file = self.lock_dir / f"{run_id}.lock"
            lock_data = {
                'run_id': run_id,
                'acquired_at': datetime.now().isoformat(),
                'timeout': timeout
            }
            with open(lock_file, 'w') as f:
                json.dump(lock_data, f)
            return True
        else:
            # 加入等待队列
            queue_file = self.lock_dir / "queue.json"
            queue = []
            if queue_file.exists():
                with open(queue_file, 'r') as f:
                    queue = json.load(f)
            
            if run_id not in queue:
                queue.append(run_id)
                with open(queue_file, 'w') as f:
                    json.dump(queue, f)
            return False
    
    def release_lock(self, run_id: str) -> bool:
        """
        释放 GPU 资源锁
        
        Args:
            run_id: 任务 ID
        
        Returns:
            是否成功释放锁
        """
        if self.use_redis:
            return self._release_redis_lock(run_id)
        else:
            return self._release_file_lock(run_id)
    
    def _release_redis_lock(self, run_id: str) -> bool:
        """使用 Redis 释放锁"""
        try:
            lock_key = f"gpu:lock:{run_id}"
            self.redis_client.delete(lock_key)
            self.redis_client.srem("gpu:running", run_id)
            self.redis_client.hdel("gpu:lock_info", run_id)
            
            # 从队列中获取下一个等待的任务
            queue_key = "gpu:queue"
            next_task = self.redis_client.rpop(queue_key)
            if next_task:
                # 递归尝试为下一个任务获取锁
                self.acquire_lock(next_task.decode() if isinstance(next_task, bytes) else next_task)
            
            return True
        except Exception as e:
            print(f"Redis 锁释放错误: {e}")
            return False
    
    def _release_file_lock(self, run_id: str) -> bool:
        """使用文件锁释放锁"""
        lock_file = self.lock_dir / f"{run_id}.lock"
        if lock_file.exists():
            lock_file.unlink()
        
        # 尝试为队列中的下一个任务获取锁
        queue_file = self.lock_dir / "queue.json"
        if queue_file.exists():
            with open(queue_file, 'r') as f:
                queue = json.load(f)
            
            if queue:
                next_task = queue.pop(0)
                if self.acquire_lock(next_task):
                    with open(queue_file, 'w') as f:
                        json.dump(queue, f)
        
        return True
    
    def get_lock_status(self, run_id: str) -> dict:
        """
        获取任务的锁状态
        
        Returns:
            {
                'has_lock': bool,           # 是否持有锁
                'position_in_queue': int,   # 队列中的位置（-1 表示已持有锁）
                'estimated_wait_time': int  # 预计等待时间（秒）
            }
        """
        if self.use_redis:
            return self._get_redis_lock_status(run_id)
        else:
            return self._get_file_lock_status(run_id)
    
    def _get_redis_lock_status(self, run_id: str) -> dict:
        """获取 Redis 锁状态"""
        try:
            lock_key = f"gpu:lock:{run_id}"
            has_lock = self.redis_client.exists(lock_key) > 0
            
            if has_lock:
                return {
                    'has_lock': True,
                    'position_in_queue': -1,
                    'estimated_wait_time': 0
                }
            else:
                # 检查队列位置
                queue_key = "gpu:queue"
                queue = self.redis_client.lrange(queue_key, 0, -1)
                queue = [task.decode() if isinstance(task, bytes) else task for task in queue]
                
                position = queue.index(run_id) if run_id in queue else -1
                
                # 预计等待时间（假设每个任务平均 1 小时）
                estimated_wait = position * 3600 if position >= 0 else 0
                
                return {
                    'has_lock': False,
                    'position_in_queue': position,
                    'estimated_wait_time': estimated_wait
                }
        except Exception as e:
            print(f"Redis 状态查询错误: {e}")
            return {'has_lock': False, 'position_in_queue': -1, 'estimated_wait_time': 0}
    
    def _get_file_lock_status(self, run_id: str) -> dict:
        """获取文件锁状态"""
        lock_file = self.lock_dir / f"{run_id}.lock"
        has_lock = lock_file.exists()
        
        if has_lock:
            return {
                'has_lock': True,
                'position_in_queue': -1,
                'estimated_wait_time': 0
            }
        else:
            # 检查队列位置
            queue_file = self.lock_dir / "queue.json"
            position = -1
            if queue_file.exists():
                with open(queue_file, 'r') as f:
                    queue = json.load(f)
                position = queue.index(run_id) if run_id in queue else -1
            
            estimated_wait = position * 3600 if position >= 0 else 0
            
            return {
                'has_lock': False,
                'position_in_queue': position,
                'estimated_wait_time': estimated_wait
            }
    
    def get_running_tasks(self) -> List[str]:
        """获取当前运行的所有任务 ID"""
        if self.use_redis:
            return self._get_redis_running_tasks()
        else:
            return self._get_file_running_tasks()
    
    def _get_redis_running_tasks(self) -> List[str]:
        """获取 Redis 中运行的任务"""
        try:
            running = self.redis_client.smembers("gpu:running")
            return [task.decode() if isinstance(task, bytes) else task for task in running]
        except:
            return []
    
    def _get_file_running_tasks(self) -> List[str]:
        """获取文件系统中运行的任务"""
        lock_files = list(self.lock_dir.glob("*.lock"))
        running_tasks = []
        for lock_file in lock_files:
            try:
                with open(lock_file, 'r') as f:
                    data = json.load(f)
                    running_tasks.append(data.get('run_id'))
            except:
                pass
        return running_tasks
    
    def get_queue(self) -> List[str]:
        """获取等待队列中的所有任务 ID"""
        if self.use_redis:
            return self._get_redis_queue()
        else:
            return self._get_file_queue()
    
    def _get_redis_queue(self) -> List[str]:
        """获取 Redis 队列"""
        try:
            queue = self.redis_client.lrange("gpu:queue", 0, -1)
            return [task.decode() if isinstance(task, bytes) else task for task in queue]
        except:
            return []
    
    def _get_file_queue(self) -> List[str]:
        """获取文件系统队列"""
        queue_file = self.lock_dir / "queue.json"
        if queue_file.exists():
            try:
                with open(queue_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _cleanup_expired_locks(self):
        """清理已过期的锁"""
        for lock_file in self.lock_dir.glob("*.lock"):
            try:
                with open(lock_file, 'r') as f:
                    data = json.load(f)
                    acquired_at = datetime.fromisoformat(data['acquired_at'])
                    timeout = data['timeout']
                    
                    if datetime.now() > acquired_at + timedelta(seconds=timeout):
                        lock_file.unlink()
            except:
                pass


# 全局 GPU 资源锁实例（最多同时 1 个任务）
gpu_lock = GPUResourceLock(max_concurrent_tasks=1)
