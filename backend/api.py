from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import psutil

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
    
    # 4. Start training subprocess
    import threading

    def stream_subprocess_logs(process, run_id):
        log_lines = []
        try:
            for line in process.stdout:
                log_lines.append(line)
                # Limit log length to avoid excessive size
                if len(log_lines) > 200:
                    log_lines = log_lines[-200:]
                # Write logs field in real-time
                task_info = load_task_status(run_id) or {}
                task_info.setdefault("logs", [])
                task_info["logs"] = log_lines
                save_task_status(run_id, task_info)
        except Exception as e:
            task_info = load_task_status(run_id) or {}
            task_info["status"] = "failed"
            task_info["error"] = f"Log streaming error: {str(e)}"
            save_task_status(run_id, task_info)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path.cwd()
        )
        # Start log collection thread
        threading.Thread(target=stream_subprocess_logs, args=(process, run_id), daemon=True).start()
        # Update PID
        task_info["pid"] = process.pid
        task_info["status"] = "running"
        save_task_status(run_id, task_info)
    except Exception as e:
        task_info["status"] = "failed"
        task_info["error"] = str(e)
        save_task_status(run_id, task_info)
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")

    return {
        "status": "started",
        "run_id": run_id,
        "message": "Training started successfully"
    }


@app.get("/status/{run_id}")
async def get_status(run_id: str):
    """Get task status"""
    task_info = load_task_status(run_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if the process is still running
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


@app.post("/stop/{run_id}")
async def stop_task(run_id: str):
    """Stop training task"""
    task_info = load_task_status(run_id)
    
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found")
    
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
    
    return {"status": "deleted", "message": "Task deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)