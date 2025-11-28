"""
inference - load trained LoRA models and provide chat API
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from pathlib import Path
import json

app = FastAPI(title="LLM Inference API")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store loaded models
loaded_models = {}


class ChatRequest(BaseModel):
    """Chat request"""
    model_id: str  # Model ID
    message: str   # User message
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    history: Optional[List[dict]] = None  # Chat history


class ChatResponse(BaseModel):
    """Chat response"""
    response: str
    model_id: str


class ModelInfo(BaseModel):
    """Model information"""
    model_id: str
    base_model: str
    model_path: str
    loaded: bool


def find_model_path(model_id: str) -> Optional[Path]:
    """Find model path based on model_id"""
    # Method 1: Direct full path
    path = Path(model_id)
    if path.exists() and (path / "adapter_config.json").exists():
        return path
    
    # Method 2: It's a run_id, look in outputs directory
    run_path = Path(f"./outputs/{model_id}/final_model")
    if run_path.exists():
        return run_path
    
    # Method 3: Search in outputs directory
    outputs_dir = Path("./outputs")
    if outputs_dir.exists():
        for run_dir in outputs_dir.glob("*/final_model"):
            if model_id in str(run_dir):
                return run_dir
    
    return None


def load_model(model_path: Path):
    """Load base model + LoRA weights"""
    print(f"📦 Loading model: {model_path}")
    
    # Read adapter_config.json to get base model info
    adapter_config_path = model_path / "adapter_config.json"
    if not adapter_config_path.exists():
        raise ValueError(f"adapter_config.json not found: {adapter_config_path}")
    
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError("adapter_config.json missing base_model_name_or_path")
    
    print(f"  Base model: {base_model_name}")
    
    # Load tokenizer
    print("  load tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("  load base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    print("  load LoRA weights...")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # Merge weights (optional, improves inference speed)
    #
    print("  merge weights...")
    model = model.merge_and_unload()
    
    model.eval()
    print("✅ Model loaded successfully")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "base_model_name": base_model_name,
        "model_path": str(model_path)
    }


def generate_response(model_info, message, max_length=512, temperature=0.7, top_p=0.9):
    """Generate response"""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    # Construct prompt (using training format)
    prompt = f"### Instruction:\n{message}\n\n### Response:\n"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract Response part
    if "### Response:" in full_response:
        response = full_response.split("### Response:")[-1].strip()
    else:
        response = full_response[len(prompt):].strip()
    
    return response


@app.get("/")
async def root():
    return {
        "message": "LLM Inference API",
        "loaded_models": list(loaded_models.keys()),
        "endpoints": {
            "load_model": "POST /load_model",
            "chat": "POST /chat",
            "list_models": "GET /models",
            "unload_model": "POST /unload_model/{model_id}"
        }
    }


@app.post("/load_model")
async def load_model_endpoint(model_id: str):
    """Load model"""
    # check if already loaded
    if model_id in loaded_models:
        return {
            "status": "already_loaded",
            "message": f"Model {model_id} already loaded",
            "model_info": {
                "model_id": model_id,
                "base_model": loaded_models[model_id]["base_model_name"]
            }
        }
    
    # Find model path
    model_path = find_model_path(model_id)
    if not model_path:
        raise HTTPException(
            status_code=404,
            detail=f"Model not found: {model_id}"
        )
    
    try:
        # Load model
        model_info = load_model(model_path)
        loaded_models[model_id] = model_info
        
        return {
            "status": "loaded",
            "message": f"Model {model_id} loaded successfully",
            "model_info": {
                "model_id": model_id,
                "base_model": model_info["base_model_name"],
                "model_path": model_info["model_path"]
            }
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat interface"""
    # Check if model is loaded
    if request.model_id not in loaded_models:
        # Try to auto-load the model
        try:
            await load_model_endpoint(request.model_id)
        except:
            raise HTTPException(
                status_code=404,
                detail=f"Model not loaded: {request.model_id}, please call /load_model first"
            )
    
    try:
        model_info = loaded_models[request.model_id]
        
        # Generate response
        response = generate_response(
            model_info,
            request.message,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        return ChatResponse(
            response=response,
            model_id=request.model_id
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )


@app.get("/models")
async def list_models():
    """List all available models"""
    models = []
    
    # Scan outputs directory for available models
    outputs_dir = Path("./outputs")
    if outputs_dir.exists():
        for run_dir in outputs_dir.glob("*/final_model"):
            if (run_dir / "adapter_config.json").exists():
                run_id = run_dir.parent.name
                
                # Read config
                with open(run_dir / "adapter_config.json") as f:
                    config = json.load(f)
                
                models.append({
                    "model_id": run_id,
                    "base_model": config.get("base_model_name_or_path", "unknown"),
                    "model_path": str(run_dir),
                    "loaded": run_id in loaded_models
                })
    
    return {
        "total": len(models),
        "loaded": len(loaded_models),
        "models": models
    }


@app.post("/unload_model/{model_id}")
async def unload_model(model_id: str):
    """Unload model"""
    if model_id not in loaded_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model not loaded: {model_id}"
        )
    
    # Remove model reference to allow GC to free up memory
    del loaded_models[model_id]
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "status": "unloaded",
        "message": f"Model {model_id} unloaded"
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "loaded_models": len(loaded_models),
        "cuda_available": torch.cuda.is_available(),
        "device": str(next(iter(loaded_models.values()))["model"].device) if loaded_models else "N/A"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)