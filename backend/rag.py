"""
RAG API - Document management
Port: 8002
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import shutil
from pathlib import Path
import tempfile

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_system import RAGSystem

app = FastAPI(title="RAG API - Document Management")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system
rag_system = RAGSystem()


class SearchRequest(BaseModel):
    """Search request"""
    
    query: str
    top_k: Optional[int] = 5


class SearchResult(BaseModel):
    """Search result"""
    content: str
    metadata: dict
    score: float


@app.get("/")
async def root():
    return {
        "message": "RAG API - Document Management",
        "endpoints": {
            "upload_files": "POST /upload_files",
            "search": "POST /search",
            "stats": "GET /stats",
            "clear": "POST /clear"
        }
    }


@app.post("/upload_files")
async def upload_files(
    files: List[UploadFile] = File(...),
    chunk_size: int = Form(384)
):
    """
    Upload documents to the knowledge base
    
    Supported formats: .txt, .json, .jsonl
    """
    doc_ids = []
    temp_files = []
    
    try:
        # Save uploaded files to temporary directory
        for file in files:
            suffix = Path(file.filename).suffix
            
            if suffix not in ['.txt', '.json', '.jsonl']:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {suffix}. Only .txt, .json, .jsonl are supported."
                )
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_files.append(tmp.name)
        
        # Add to knowledge base
        doc_ids = rag_system.add_documents_from_files(temp_files)
        
        return {
            "status": "success",
            "message": f"Uploaded {len(doc_ids)} documents",
            "doc_ids": doc_ids,
            "stats": rag_system.get_stats()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary files
        for tmp_file in temp_files:
            try:
                Path(tmp_file).unlink()
            except:
                pass


@app.post("/add_text")
async def add_text(
    content: str = Form(...),
    metadata: Optional[str] = Form(None),
    chunk_size: int = Form(384)
):
    """
    Directly add text content to the knowledge base
    """
    try:
        import json
        meta = json.loads(metadata) if metadata else {}
    except:
        meta = {}
    
    try:
        doc_id = rag_system.add_document(content, metadata=meta, chunk_size=chunk_size)
        rag_system.save_knowledge_base()
        
        return {
            "status": "success",
            "doc_id": doc_id,
            "stats": rag_system.get_stats()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search(request: SearchRequest):
    """
    检索相关文档
    """
    try:
        results = rag_system.search(request.query, top_k=request.top_k)
        
        return {
            "query": request.query,
            "total_results": len(results),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    获取知识库统计信息
    """
    return rag_system.get_stats()


@app.post("/clear")
async def clear_knowledge_base():
    """
    清空知识库
    """
    try:
        rag_system.clear_knowledge_base()
        return {
            "status": "success",
            "message": "知识库已清空"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    stats = rag_system.get_stats()
    return {
        "status": "healthy",
        "knowledge_base": stats
    }


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 启动 RAG API")
    print("=" * 60)
    print(f"📂 知识库目录: {rag_system.kb_dir.resolve()}")
    print(f"📊 文档数量: {rag_system.get_stats()['total_chunks']}")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8002)