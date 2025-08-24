"""
REST API for the KGE system.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from .core import KGEPipeline, KGEConfig
from .utils import DataExporter, validate_graph_data

logger = logging.getLogger(__name__)


# Pydantic models for API
class ProcessTextRequest(BaseModel):
    """Request model for text processing."""
    text: str = Field(..., description="Text to process")
    document_id: Optional[str] = Field(None, description="Optional document identifier")


class ProcessBatchRequest(BaseModel):
    """Request model for batch processing."""
    texts: List[str] = Field(..., description="List of texts to process")
    document_ids: Optional[List[str]] = Field(None, description="Optional document identifiers")


class ConfigUpdateRequest(BaseModel):
    """Request model for configuration updates."""
    chunk_size: Optional[int] = Field(None, ge=100, le=10000)
    chunk_overlap: Optional[int] = Field(None, ge=0, le=1000)
    batch_size: Optional[int] = Field(None, ge=1, le=100)
    max_tokens: Optional[int] = Field(None, ge=50, le=2048)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    entity_similarity_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class QueryRequest(BaseModel):
    """Request model for graph queries."""
    node_filters: Optional[Dict[str, Any]] = Field(None, description="Filters for node queries")
    edge_filters: Optional[Dict[str, Any]] = Field(None, description="Filters for edge queries")
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Maximum results to return")


class ExportRequest(BaseModel):
    """Request model for data export."""
    format: str = Field(..., description="Export format: json, cypher, or graphml")
    pretty: Optional[bool] = Field(True, description="Pretty format for JSON")


# Global pipeline instance
pipeline: Optional[KGEPipeline] = None


def create_app(config: Optional[KGEConfig] = None) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        config: KGE configuration
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Knowledge Graph Extraction API",
        description="API for extracting knowledge graphs from text using LLMs",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize pipeline
    global pipeline
    pipeline = KGEPipeline(config)
    
    return app


app = create_app()


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("KGE API starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("KGE API shutting down...")
    if pipeline:
        pipeline.cleanup()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Knowledge Graph Extraction API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = pipeline.get_statistics() if pipeline else {}
        return {
            "status": "healthy",
            "pipeline_initialized": pipeline is not None,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.post("/process/text")
async def process_text(request: ProcessTextRequest):
    """Process a single text document.
    
    Args:
        request: Text processing request
        
    Returns:
        Processing result
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        result = pipeline.process_text(
            text=request.text,
            document_id=request.document_id
        )
        return result
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/batch")
async def process_batch(request: ProcessBatchRequest):
    """Process a batch of text documents.
    
    Args:
        request: Batch processing request
        
    Returns:
        List of processing results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        results = pipeline.process_batch(
            texts=request.texts,
            document_ids=request.document_ids
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/file")
async def process_file(file: UploadFile = File(...)):
    """Process an uploaded text file.
    
    Args:
        file: Uploaded file
        
    Returns:
        Processing result
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(
            status_code=400, 
            detail="Only .txt and .md files are supported"
        )
    
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        result = pipeline.process_text(
            text=text,
            document_id=file.filename
        )
        return result
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """Get current configuration.
    
    Returns:
        Current configuration
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    return pipeline.config.__dict__


@app.put("/config")
async def update_config(request: ConfigUpdateRequest):
    """Update configuration.
    
    Args:
        request: Configuration update request
        
    Returns:
        Updated configuration
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # Update configuration
        config_dict = request.dict(exclude_unset=True)
        for key, value in config_dict.items():
            if hasattr(pipeline.config, key):
                setattr(pipeline.config, key, value)
        
        # Reinitialize components if needed
        if any(key in config_dict for key in ['chunk_size', 'chunk_overlap']):
            pipeline.text_processor = pipeline.text_processor.__class__(
                chunk_size=pipeline.config.chunk_size,
                chunk_overlap=pipeline.config.chunk_overlap
            )
        
        return {"message": "Configuration updated", "config": pipeline.config.__dict__}
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/nodes")
async def query_nodes(request: QueryRequest):
    """Query graph nodes.
    
    Args:
        request: Query request
        
    Returns:
        Query results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        nodes = pipeline.storage.query_nodes(request.node_filters)
        
        # Apply limit
        if request.limit and len(nodes) > request.limit:
            nodes = nodes[:request.limit]
        
        return {
            "nodes": nodes,
            "count": len(nodes),
            "filters": request.node_filters
        }
    except Exception as e:
        logger.error(f"Error querying nodes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/edges")
async def query_edges(request: QueryRequest):
    """Query graph edges.
    
    Args:
        request: Query request
        
    Returns:
        Query results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        edges = pipeline.storage.query_edges(request.edge_filters)
        
        # Apply limit
        if request.limit and len(edges) > request.limit:
            edges = edges[:request.limit]
        
        return {
            "edges": edges,
            "count": len(edges),
            "filters": request.edge_filters
        }
    except Exception as e:
        logger.error(f"Error querying edges: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/statistics")
async def get_statistics():
    """Get system statistics.
    
    Returns:
        System statistics
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/export")
async def export_data(request: ExportRequest):
    """Export graph data.
    
    Args:
        request: Export request
        
    Returns:
        Exported data
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if request.format not in ["json", "cypher", "graphml"]:
        raise HTTPException(
            status_code=400,
            detail="Format must be one of: json, cypher, graphml"
        )
    
    try:
        # Get all nodes and edges
        nodes = pipeline.storage.query_nodes()
        edges = pipeline.storage.query_edges()
        
        graph_data = {
            "nodes": nodes,
            "edges": edges,
            "statistics": pipeline.storage.get_statistics()
        }
        
        # Export in requested format
        if request.format == "json":
            exported = DataExporter.to_json(graph_data, pretty=request.pretty)
            media_type = "application/json"
        elif request.format == "cypher":
            exported = DataExporter.to_cypher(graph_data)
            media_type = "text/plain"
        elif request.format == "graphml":
            exported = DataExporter.to_graphml(graph_data)
            media_type = "application/xml"
        
        return JSONResponse(
            content={"data": exported, "format": request.format},
            media_type=media_type
        )
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/types")
async def get_supported_types():
    """Get supported entity and relation types.
    
    Returns:
        Supported types
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        types = pipeline.extractor.get_supported_types()
        return types
    except Exception as e:
        logger.error(f"Error getting types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_data():
    """Clear all stored graph data.
    
    Returns:
        Confirmation message
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # This would need to be implemented in storage backends
        # For now, just return a message
        return {"message": "Clear operation not implemented yet"}
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 8000, config: Optional[KGEConfig] = None):
    """Run the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        config: KGE configuration
    """
    global pipeline
    if config:
        pipeline = KGEPipeline(config)
    
    uvicorn.run(
        "kge.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()