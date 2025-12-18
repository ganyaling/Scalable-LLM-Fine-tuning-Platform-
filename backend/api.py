from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uuid
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import psutil
import sys

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tasks import train_task, check_task_status
from celery.result import AsyncResult
from gpu_lock import gpu_lock
from websocket_logs import log_manager

app = FastAPI(title="LLM Fine-tuning API")

# cors settings(for frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# directory configuration
DATA_DIR = Path("./data/uploads")
STATUS_DIR = Path("./data/status")
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATUS_DIR.mkdir(parents=True, exist_ok=True)

# task status storage (should use a database in production)
tasks: Dict[str, dict] = {}


def save_task_status(run_id: str, status: dict):
    """Save task status to file"""
    status_file = STATUS_DIR / f"{run_id}.json"
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    tasks[run_id] = status


def load_task_status(run_id: str) -> Optional[dict]:
    """Load task status from file"""
    status_file = STATUS_DIR / f"{run_id}.json"
    if status_file.exists():
        with open(status_file, "r") as f:
            return json.load(f)
    return tasks.get(run_id)


# ============ WebSocket 实时日志端点 ============

@app.websocket("/ws/logs/{run_id}")
async def websocket_logs(websocket: WebSocket, run_id: str):
    """
    WebSocket 端点：实时推送训练日志
    
    连接方式: ws://localhost:8000/ws/logs/{run_id}
    
    消息格式:
    {
        "type": "log" | "progress" | "status",
        "data": {...}
    }
    """
    await websocket.accept()
    log_manager.register_connection(run_id, websocket)
    
    try:
        # 首先发送日志历史
        history = log_manager.get_log_history(run_id)
        await websocket.send_json({
            "type": "history",
            "data": {
                "run_id": run_id,
                "logs": history,
                "count": len(history)
            }
        })
        
        # 发送当前任务状态
        task_info = load_task_status(run_id)
        if task_info:
            await websocket.send_json({
                "type": "status",
                "data": {
                    "run_id": run_id,
                    "status": task_info.get("status", "unknown"),
                    "progress": task_info.get("progress", 0),
                    "timestamp": datetime.now().isoformat()
                }
            })
        
        # 保持连接，等待来自后端的日志推送
        while True:
            # 客户端可以发送 ping 来保活连接
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        log_manager.unregister_connection(run_id, websocket)
    except Exception as e:
        print(f"WebSocket 错误: {e}")
        log_manager.unregister_connection(run_id, websocket)


@app.get("/logs/{run_id}/history")
async def get_logs_history(run_id: str):
    """获取日志历史（HTTP 备选方案）"""
    history = log_manager.get_log_history(run_id)
    return {
        "run_id": run_id,
        "logs": history,
        "count": len(history),
        "connections": log_manager.get_connection_count(run_id)
    }


@app.delete("/logs/{run_id}")
async def clear_logs(run_id: str):
    """清除日志历史"""
    log_manager.clear_logs(run_id)
    return {"status": "cleared", "run_id": run_id}


@app.get("/logs/status")
async def get_logs_status():
    """获取全局日志状态"""
    return {
        "total_connections": log_manager.get_all_connections_count(),
        "active_tasks": list(log_manager.connections.keys()),
        "tasks_with_logs": list(log_manager.log_history.keys())
    }


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "message": "LLM Fine-tuning API",
        "endpoints": {
            "start_finetune": "POST /start_finetune",
            "get_status": "GET /status/{run_id}",
            "list_tasks": "GET /tasks",
            "stop_task": "POST /stop/{run_id}"
        }
    }


@app.post("/start_finetune")
async def start_finetune(
    data_file: UploadFile = File(None),  # Optional
    dataset_name: str = Form(None),  
    dataset_split: str = Form("train"),  
    num_samples: int = Form(None),  
    base_model: str = Form("Qwen/Qwen2-1.5B-Instruct"),
    lora_r: int = Form(16),
    lora_alpha: int = Form(32),
    num_epochs: int = Form(3),
    batch_size: int = Form(4),
    learning_rate: float = Form(2e-4),
):
    """Start fine-tuning task"""
    run_id = str(uuid.uuid4())
    
    # Validate: must provide data source
    if not data_file and not dataset_name:
        raise HTTPException(
            status_code=400, 
            detail="Must provide either data_file or dataset_name"
        )
    
    data_path = None
    
    # 1. If a file is uploaded, save it
    if data_file:
        if not data_file.filename.endswith('.jsonl'):
            raise HTTPException(status_code=400, detail="Only .jsonl files are supported")
        
        data_path = DATA_DIR / f"{run_id}.jsonl"
        try:
            with data_path.open("wb") as f:
                shutil.copyfileobj(data_file.file, f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # 2. Initialize task status
    task_info = {
        "run_id": run_id,
        "status": "starting",
        "base_model": base_model,
        "data_source": dataset_name if dataset_name else data_file.filename,
        "params": {
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_samples": num_samples,
        },
        "created_at": datetime.now().isoformat(),
        "pid": None,
        "progress": 0,
        "logs": []
    }
    save_task_status(run_id, task_info)
    
    # 2.5. 尝试获取 GPU 资源锁
    lock_acquired = gpu_lock.acquire_lock(run_id)
    
    if lock_acquired:
        task_info["gpu_lock_status"] = "acquired"
        task_info["gpu_lock_acquired_at"] = datetime.now().isoformat()
        gpu_status = "✓ GPU 资源已获取，任务将立即开始"
    else:
        task_info["gpu_lock_status"] = "waiting"
        task_info["gpu_lock_waiting_since"] = datetime.now().isoformat()
        lock_info = gpu_lock.get_lock_status(run_id)
        queue_position = lock_info.get('position_in_queue', -1)
        gpu_status = f"⏳ 等待 GPU 资源，队列位置: {queue_position + 1}"
    
    save_task_status(run_id, task_info)
    
    # 3. Build command
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
    
    # Add data source parameters
    if data_path:
        cmd.append(f"--data_path={data_path}")
    if dataset_name:
        cmd.append(f"--dataset_name={dataset_name}")
        cmd.append(f"--dataset_split={dataset_split}")
    if num_samples:
        cmd.append(f"--num_samples={num_samples}")
    
    # 4. 使用 Celery 推送任务到后台队列
    try:
        # 推送训练任务到 Celery 队列
        celery_task = train_task.apply_async(
            kwargs={
                'run_id': run_id,
                'base_model': base_model,
                'data_path': str(data_path) if data_path else None,
                'dataset_name': dataset_name,
                'dataset_split': dataset_split,
                'num_samples': num_samples,
                'lora_r': lora_r,
                'lora_alpha': lora_alpha,
                'num_epochs': num_epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
            },
            queue='training'
        )
        
        # 更新任务信息，包含 Celery 任务 ID
        task_info["status"] = "running" if lock_acquired else "queued"
        task_info["celery_task_id"] = celery_task.id
        task_info["celery_task_queued_at"] = datetime.now().isoformat()
        save_task_status(run_id, task_info)
    except Exception as e:
        task_info["status"] = "failed"
        task_info["error"] = str(e)
        task_info["gpu_lock_status"] = "failed"
        save_task_status(run_id, task_info)
        raise HTTPException(status_code=500, detail=f"Failed to queue training task: {str(e)}")

    return {
        "status": "started",
        "run_id": run_id,
        "message": "Training started successfully",
        "gpu_status": gpu_status,
        "gpu_lock_acquired": lock_acquired,
        "queue_position": gpu_lock.get_lock_status(run_id).get('position_in_queue', -1)
    }


@app.get("/status/{run_id}")
async def get_status(run_id: str):
    """Get task status"""
    task_info = load_task_status(run_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 如果有 Celery 任务 ID，同步 Celery 任务状态
    if task_info.get("celery_task_id"):
        try:
            from celery_app import app as celery_app
            celery_result = AsyncResult(task_info["celery_task_id"], app=celery_app)
            
            # 映射 Celery 状态到我们的状态
            celery_state_map = {
                'PENDING': 'queued',
                'STARTED': 'running',
                'PROGRESS': 'running',
                'SUCCESS': 'completed',
                'FAILURE': 'failed',
                'RETRY': 'running',
                'REVOKED': 'stopped',
            }
            
            task_info['celery_status'] = celery_result.status
            
            # 如果 Celery 任务已完成但我们的状态还是 running，更新之
            if celery_result.status == 'SUCCESS' and task_info.get('status') == 'running':
                task_info['status'] = 'completed'
                task_info['progress'] = 100
                save_task_status(run_id, task_info)
            elif celery_result.status == 'FAILURE' and task_info.get('status') == 'running':
                task_info['status'] = 'failed'
                task_info['error'] = str(celery_result.info)
                save_task_status(run_id, task_info)
        except Exception as e:
            print(f"Warning: Failed to sync Celery status: {e}")
    
    # 检查进程是否还在运行（向后兼容）
    if task_info.get("pid") and task_info["status"] == "running":
        try:
            process = psutil.Process(task_info["pid"])
            if not process.is_running():
                task_info["status"] = "completed"
                save_task_status(run_id, task_info)
        except psutil.NoSuchProcess:
            task_info["status"] = "completed"
            save_task_status(run_id, task_info)
    
    # Try to get the latest metrics from MLflow
    try:
        mlruns_path = Path("./mlruns")
        if mlruns_path.exists():
            # Here you can parse MLflow's metrics files
            # Simplified example: read logs from the output directory
            output_dir = Path(f"./outputs/{run_id}")
            if output_dir.exists():
                log_file = output_dir / "trainer_state.json"
                if log_file.exists():
                    with open(log_file) as f:
                        trainer_state = json.load(f)
                        task_info["progress"] = int(
                            (trainer_state.get("global_step", 0) / 
                             trainer_state.get("max_steps", 1)) * 100
                        )
    except Exception:
        pass
    
    return task_info


@app.get("/tasks")
async def list_tasks():
    """List all tasks"""
    all_tasks = []
    
    for status_file in STATUS_DIR.glob("*.json"):
        try:
            with open(status_file) as f:
                task = json.load(f)
                all_tasks.append(task)
        except Exception:
            continue
    
    # Sort by creation time
    all_tasks.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    
    return {
        "total": len(all_tasks),
        "tasks": all_tasks
    }


@app.get("/celery/status/{celery_task_id}")
async def get_celery_task_status(celery_task_id: str):
    """Get Celery task status directly"""
    try:
        from celery_app import app as celery_app
        result = AsyncResult(celery_task_id, app=celery_app)
        
        return {
            'celery_task_id': celery_task_id,
            'status': result.status,
            'result': result.result if result.ready() else None,
            'info': result.info if not result.successful() else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Celery task status: {str(e)}")


@app.post("/stop/{run_id}")
async def stop_task(run_id: str):
    """Stop training task"""
    task_info = load_task_status(run_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # 优先使用 Celery 来终止任务
    if task_info.get("celery_task_id"):
        try:
            from celery_app import app as celery_app
            result = AsyncResult(task_info["celery_task_id"], app=celery_app)
            result.revoke(terminate=True)
            
            task_info["status"] = "stopped"
            task_info["stopped_at"] = datetime.now().isoformat()
            save_task_status(run_id, task_info)
            
            return {"status": "stopped", "message": "Task stopped successfully via Celery"}
        except Exception as e:
            print(f"Warning: Failed to revoke Celery task: {e}")
    
    # 向后兼容：如果有 PID，也尝试终止进程
    if task_info.get("pid"):
        try:
            process = psutil.Process(task_info["pid"])
            process.terminate()
            process.wait(timeout=10)
            
            task_info["status"] = "stopped"
            task_info["stopped_at"] = datetime.now().isoformat()
            save_task_status(run_id, task_info)
            
            return {"status": "stopped", "message": "Task stopped successfully"}
        except psutil.NoSuchProcess:
            task_info["status"] = "completed"
            save_task_status(run_id, task_info)
            return {"status": "completed", "message": "Task already completed"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to stop task: {str(e)}")
    
    return {"status": task_info["status"], "message": "Task not running"}


@app.delete("/task/{run_id}")
async def delete_task(run_id: str):
    """Delete task (including data and status)"""
    task_info = load_task_status(run_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Ensure the task is stopped
    if task_info["status"] == "running":
        raise HTTPException(status_code=400, detail="Cannot delete running task. Stop it first.")
    
    # Delete data file
    data_file = DATA_DIR / f"{run_id}.jsonl"
    if data_file.exists():
        data_file.unlink()
    
    # Delete status file
    status_file = STATUS_DIR / f"{run_id}.json"
    if status_file.exists():
        status_file.unlink()
    
    # Delete output directory
    output_dir = Path(f"./outputs/{run_id}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Remove from memory
    if run_id in tasks:
        del tasks[run_id]
    
    # 释放 GPU 锁
    gpu_lock.release_lock(run_id)
    
    return {"status": "deleted", "message": "Task deleted successfully"}


# ============ GPU 资源管理端点 ============

@app.get("/gpu/status")
async def get_gpu_status():
    """获取 GPU 资源状态"""
    running_tasks = gpu_lock.get_running_tasks()
    queue = gpu_lock.get_queue()
    
    return {
        "max_concurrent_tasks": gpu_lock.max_concurrent_tasks,
        "running_tasks": running_tasks,
        "running_count": len(running_tasks),
        "queued_tasks": queue,
        "queued_count": len(queue),
        "available_slots": max(0, gpu_lock.max_concurrent_tasks - len(running_tasks))
    }


@app.get("/gpu/task/{run_id}")
async def get_task_gpu_status(run_id: str):
    """获取任务的 GPU 资源锁状态"""
    lock_status = gpu_lock.get_lock_status(run_id)
    
    return {
        "run_id": run_id,
        "has_lock": lock_status['has_lock'],
        "position_in_queue": lock_status['position_in_queue'],
        "estimated_wait_time_seconds": lock_status['estimated_wait_time'],
        "estimated_wait_time_minutes": round(lock_status['estimated_wait_time'] / 60, 2)
    }


@app.post("/gpu/release/{run_id}")
async def release_gpu_lock(run_id: str):
    """手动释放 GPU 资源锁"""
    task_info = load_task_status(run_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = gpu_lock.release_lock(run_id)
    
    if result:
        task_info["gpu_lock_status"] = "released"
        task_info["gpu_lock_released_at"] = datetime.now().isoformat()
        save_task_status(run_id, task_info)
        return {"status": "released", "message": f"GPU lock for task {run_id} has been released"}
    else:
        raise HTTPException(status_code=500, detail="Failed to release GPU lock")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)