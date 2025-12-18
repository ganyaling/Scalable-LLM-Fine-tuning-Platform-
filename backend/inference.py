"""
inference - Load trained LoRA models and provide chat API
with RAG enhancement
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
import requests

app = FastAPI(title="LLM Inference API")


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG API 地址
RAG_API_URL = "http://localhost:8002"

# Global variable to store loaded models
loaded_models = {}


class ChatRequest(BaseModel):
    """Chat request"""
    model_id: str  # Model identifier (run_id or model path)
    message: str   # User message
    max_length: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    history: Optional[List[dict]] = None  # Chat history    
    use_rag: Optional[bool] = False  # Whether to use RAG
    rag_top_k: Optional[int] = 3  # Number of RAG retrievals


class ChatResponse(BaseModel):
    """Chat response"""
    response: str
    model_id: str
    rag_used: Optional[bool] = False
    rag_sources: Optional[List[dict]] = None


class ModelInfo(BaseModel):
    """Model information"""
    model_id: str
    base_model: str
    model_path: str
    loaded: bool


def find_model_path(model_id: str) -> Optional[Path]:
    """Find model path based on model_id"""
    # Method 1: Directly a full path
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
    
 
    # load tokenizer
    print("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("  Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load LoRA weights
    print("  Loading LoRA weights...")
    model = PeftModel.from_pretrained(base_model, model_path)
    
    # combine base model and LoRA weights
    print("  Combining weights...")
    model = model.merge_and_unload()
    
    model.eval()
    
    print("✅ Model loaded successfully")
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "base_model_name": base_model_name,
        "model_path": str(model_path)
    }


def build_conversation_prompt(history: List[dict], current_message: str, tokenizer) -> str:
    """
    构建多轮对话上下文
    
    Args:
        history: 对话历史列表，每个元素为 {"role": "user"|"assistant", "content": "..."}
        current_message: 当前用户消息
        tokenizer: 分词器（用于检查是否有 chat_template）
    
    Returns:
        构建好的提示文本
    """
    # 优先使用 Chat Template（Transformers 官方推荐方式）
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = []
        
        # 添加历史消息
        if history:
            for msg in history:
                messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", "")
                })
        
        # 添加当前消息
        messages.append({"role": "user", "content": current_message})
        
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            print(f"Chat template 应用失败: {e}，使用备选方案")
    
    # 备选方案: 使用标准的指令格式
    conversation = []
    
    if history:
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                conversation.append(f"### Instruction:\n{content}")
            elif role == "assistant":
                conversation.append(f"### Response:\n{content}")
    
    # 添加当前消息
    conversation.append(f"### Instruction:\n{current_message}")
    conversation.append("### Response:")
    
    prompt = "\n\n".join(conversation)
    return prompt


def generate_response(model_info, message, max_length=512, temperature=0.7, top_p=0.9, history: Optional[List[dict]] = None):
    """Generate response from model with conversation history"""
    model = model_info["model"]
    tokenizer = model_info["tokenizer"]
    
    # 构建包含历史的提示
    if history:
        prompt = build_conversation_prompt(history, message, tokenizer)
    else:
        # 如果没有历史，使用单轮对话格式
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            try:
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": message}],
                    tokenize=False,
                    add_generation_prompt=True
                )
            except:
                prompt = f"### Instruction:\n{message}\n\n### Response:\n"
        else:
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
    
    # 移除可能的多余内容（下一个指令块等）
    response = response.split("### Instruction:")[0].strip()
    
    return response


def search_rag(query: str, top_k: int = 3):
    """Call RAG API to search for relevant documents"""
    try:
        response = requests.post(
            f"{RAG_API_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["results"]
        return None
    except:
        return None


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
    """Load model by model_id"""
    # Check if already loaded
    if model_id in loaded_models:
        return {
            "status": "already_loaded",
            "message": f"model {model_id} already loaded",
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
            "message": f"Model loaded successfully: {model_id}",
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
    """Chat endpoint (supports RAG)"""
    # Check if model is loaded
    if request.model_id not in loaded_models:
        # Try to auto-load
        try:
            await load_model_endpoint(request.model_id)
        except:
            raise HTTPException(
                status_code=404,
                detail=f"Model not loaded: {request.model_id}, please call /load_model first"
            )
    
    try:
        model_info = loaded_models[request.model_id]
        message = request.message
        rag_sources = None
        rag_used = False
        
        # If RAG is enabled, first search for relevant documents
        if request.use_rag:
            rag_results = search_rag(message, top_k=request.rag_top_k)
            
            if rag_results:
                rag_used = True
                rag_sources = rag_results
                
                # Construct RAG-enhanced prompt
                context = "\n\n".join([
                    f"Reference {i+1}:\n{r['content']}"
                    for i, r in enumerate(rag_results)
                ])
                
                message = f"""Please answer the question based on the following references. If the references do not contain relevant information, you may use your own knowledge to answer.

{context}

question: {request.message}

Answer:"""
        
        # Generate response with conversation history
        response = generate_response(
            model_info,
            message,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            history=request.history
        )
        
        return ChatResponse(
            response=response,
            model_id=request.model_id,
            rag_used=rag_used,
            rag_sources=rag_sources
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
    
    # Scan outputs directory
    outputs_dir = Path("./outputs")
    if outputs_dir.exists():
        for run_dir in outputs_dir.glob("*/final_model"):
            if (run_dir / "adapter_config.json").exists():
                run_id = run_dir.parent.name
                
                # read adapter_config.json
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
    """Unload a loaded model to free up resources"""
    if model_id not in loaded_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model not loaded: {model_id}"
        )
    
    # Delete model reference to allow GC to free up memory
    del loaded_models[model_id]
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "status": "unloaded",
        "message": f"Model {model_id} has been unloaded"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "loaded_models": len(loaded_models),
        "cuda_available": torch.cuda.is_available(),
        "device": str(next(iter(loaded_models.values()))["model"].device) if loaded_models else "N/A"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)