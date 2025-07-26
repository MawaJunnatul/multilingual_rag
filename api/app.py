from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline
from config.settings import settings

app = FastAPI(
    title="Multilingual RAG API",
    description="A multilingual RAG system for Bengali and English queries",
    version="1.0.0"
)

# Global RAG pipeline instance
rag_pipeline = None

class QueryRequest(BaseModel):
    question: str
    include_history: bool = True

class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    relevant_docs_count: int
    retrieved_docs: List[Dict]

class BuildKnowledgeBaseRequest(BaseModel):
    pdf_path: str

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    if settings.COHERE_API_KEY:
        rag_pipeline = RAGPipeline()

@app.post("/build-knowledge-base")
async def build_knowledge_base(request: BuildKnowledgeBaseRequest):
    """Build knowledge base from PDF"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=400, detail="RAG pipeline not initialized")
    
    if not os.path.exists(request.pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    try:
        doc_count = rag_pipeline.build_knowledge_base(request.pdf_path)
        return {"message": "Knowledge base built successfully", "document_count": doc_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error building knowledge base: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process query and return response"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=400, detail="RAG pipeline not initialized")
    
    try:
        result = rag_pipeline.query(request.question, request.include_history)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=400, detail="RAG pipeline not initialized")
    
    return rag_pipeline.get_stats()

@app.delete("/clear-history")
async def clear_history():
    """Clear conversation history"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(status_code=400, detail="RAG pipeline not initialized")
    
    rag_pipeline.clear_history()
    return {"message": "Conversation history cleared"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "pipeline_initialized": rag_pipeline is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)