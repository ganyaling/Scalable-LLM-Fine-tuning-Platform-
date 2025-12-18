"""
Celery Task definitions for asynchronous training jobs
"""
import subprocess
import json
from pathlib import Path
from datetime import datetime
from celery import Task
from celery.signals import task_prerun, task_postrun
from celery_app import app


# 导入 GPU 锁定管理器和 WebSocket 日志管理器
import sys
sys.path.insert(0, str(Path(__file__).parent / 'backend'))
try:
    from backend import gpu_lock
except ImportError:
    gpu_lock = None

try:
    from backend.websocket_logs import log_manager
except ImportError:
    log_manager = None


class TrainingTask(Task):
    """Custom training task class for handling status and errors"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Callback when task fails"""
        run_id = kwargs.get('run_id')
        if run_id:
            status_file = Path("./data/status") / f"{run_id}.json"
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
                status['status'] = 'failed'
                status['error'] = str(exc)
                status['celery_task_id'] = task_id
                status['gpu_lock_released_at'] = datetime.now().isoformat()
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)
            
            # 释放 GPU 锁
            if gpu_lock:
                gpu_lock.release_lock(run_id)
    
    def on_success(self, result, task_id, args, kwargs):
        """Callback when task succeeds"""
        run_id = kwargs.get('run_id')
        if run_id:
            status_file = Path("./data/status") / f"{run_id}.json"
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
                status['status'] = 'completed'
                status['progress'] = 100
                status['celery_task_id'] = task_id
                status['completed_at'] = datetime.now().isoformat()
                status['gpu_lock_released_at'] = datetime.now().isoformat()
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)
            
            # 释放 GPU 锁
            if gpu_lock:
                gpu_lock.release_lock(run_id)


@app.task(base=TrainingTask, bind=True, name='tasks.train_task')
def train_task(
    self,
    run_id: str,
    base_model: str,
    data_path: str = None,
    dataset_name: str = None,
    dataset_split: str = "train",
    num_samples: int = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    **kwargs
):
    """
    后台训练任务
    
    Args:
        run_id: 任务唯一 ID
        base_model: 基础模型名称
        data_path: 本地数据文件路径
        dataset_name: HuggingFace 数据集名称
        dataset_split: 数据集分割
        num_samples: 采样数量
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        num_epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
    """
    import threading
    import psutil
    
    try:
        # 更新任务状态为运行中
        status_file = Path("./data/status") / f"{run_id}.json"
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            status['status'] = 'running'
            status['celery_task_id'] = self.request.id
            status['celery_task_started'] = datetime.now().isoformat()
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        
        # 构建训练命令
        cmd = [
            "python",
            "trainer/main.py",
            f"--base_model={base_model}",
            f"--lora_r={lora_r}",
            f"--lora_alpha={lora_alpha}",
            f"--num_epochs={num_epochs}",
            f"--batch_size={batch_size}",
            f"--learning_rate={learning_rate}",
            f"--run_id={run_id}",
        ]
        
        # 添加数据源参数
        if data_path:
            cmd.append(f"--data_path={data_path}")
        if dataset_name:
            cmd.append(f"--dataset_name={dataset_name}")
            cmd.append(f"--dataset_split={dataset_split}")
        if num_samples:
            cmd.append(f"--num_samples={num_samples}")
        
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path.cwd()
        )
        
        # 保存进程 ID
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            status['pid'] = process.pid
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        
        # 后台线程收集日志
        def stream_logs():
            log_lines = []
            line_number = 0
            try:
                for line in process.stdout:
                    line_number += 1
                    log_lines.append(line)
                    # 限制日志大小
                    if len(log_lines) > 200:
                        log_lines = log_lines[-200:]
                    
                    # 通过 WebSocket 广播日志（异步调用）
                    if log_manager:
                        try:
                            # 使用 asyncio 的线程安全方式广播日志
                            import asyncio
                            # 不阻塞日志流，创建后台任务
                            try:
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                loop.run_until_complete(
                                    log_manager.broadcast_log(run_id, line.strip(), "INFO", line_number)
                                )
                                loop.close()
                            except RuntimeError:
                                # 如果事件循环已关闭，尝试使用现有循环
                                pass
                            except Exception as e:
                                print(f"日志广播错误: {e}")
                        except Exception as e:
                            print(f"WebSocket 日志广播异常: {e}")
                    
                    # 实时写入日志（作为备份/审计日志）
                    if status_file.exists():
                        with open(status_file, 'r') as f:
                            current_status = json.load(f)
                        current_status.setdefault("logs", [])
                        current_status["logs"] = log_lines
                        with open(status_file, 'w') as f:
                            json.dump(current_status, f, indent=2)
                    
                    # 更新 Celery 任务状态
                    self.update_state(
                        state='PROGRESS',
                        meta={'progress': 'Training...', 'log_count': len(log_lines)}
                    )
            except Exception as e:
                print(f"日志收集错误: {e}")
        
        # 启动日志收集线程
        log_thread = threading.Thread(target=stream_logs, daemon=True)
        log_thread.start()
        
        # 等待进程完成
        return_code = process.wait()
        
        # 检查进程是否成功
        if return_code != 0:
            raise Exception(f"训练进程以代码 {return_code} 退出")
        
        return {
            'status': 'success',
            'run_id': run_id,
            'message': '训练成功完成'
        }
    
    except Exception as e:
        # 错误已在 on_failure 中处理
        raise


@app.task(name='tasks.cleanup_task')
def cleanup_task(run_id: str):
    """
    清理任务（删除临时文件等）
    
    Args:
        run_id: 任务 ID
    """
    try:
        data_file = Path("./data/uploads") / f"{run_id}.jsonl"
        if data_file.exists():
            data_file.unlink()
        return {'status': 'cleaned', 'run_id': run_id}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


@app.task(bind=True, name='tasks.check_task_status')
def check_task_status(self, celery_task_id: str):
    """
    检查 Celery 任务状态
    
    Args:
        celery_task_id: Celery 任务 ID
    """
    from celery.result import AsyncResult
    
    result = AsyncResult(celery_task_id, app=app)
    return {
        'celery_task_id': celery_task_id,
        'status': result.status,
        'result': result.result if result.ready() else None,
    }
