import asyncio
import json
import logging
import os
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .cache import label_blob_cache
from .config import settings
from .index import SpatialIndexManager

logger = logging.getLogger(__name__)

app = FastAPI(title="Label Cache Server")
spatial_manager = SpatialIndexManager(
    max_indexes=settings.max_indexed_jobs,
    max_memory_mb=settings.max_index_memory_mb
)

# --- Middleware ---

class APIKeyMiddleware(BaseHTTPMiddleware):
    OPEN_PATHS = {"/health", "/docs", "/openapi.json"}
    
    async def dispatch(self, request: Request, call_next):
        if not settings.api_key:
            return await call_next(request)
        if request.url.path in self.OPEN_PATHS:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        if auth == f"Bearer {settings.api_key}":
            return await call_next(request)
        token = request.query_params.get("token", "")
        if token == settings.api_key:
            return await call_next(request)
        return Response(status_code=401, content="Unauthorized")

app.add_middleware(APIKeyMiddleware)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


# --- Endpoints ---

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "cached_files": label_blob_cache.file_count,
        "cached_mb": round(label_blob_cache.total_cached_mb, 1),
        **spatial_manager.stats,
    }


@app.get("/labels")
async def get_labels(blob_path: str, bbox: str | None = None):
    """Return labels within a bounding box.

    bbox format: minX,minY,maxX,maxY (image pixel coordinates)
    If bbox is omitted, returns metadata/stats only (not all labels).
    """
    job_id = blob_path.split('/')[-2] if '/' in blob_path else 'unknown'
    layer = blob_path.split('_')[-1].split('.')[0] if '_' in blob_path else 'base'
    
    # Ensure blob is cached locally
    try:
        local_path = await label_blob_cache.get(blob_path)
    except Exception as e:
        logger.error(f"Blob not found: {blob_path}: {e}")
        raise HTTPException(404, f"Annotation file not found: {blob_path}")

    # Ensure spatial index is built
    try:
        label_index = await asyncio.to_thread(
            spatial_manager.get_or_build, job_id, layer, local_path
        )
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        raise HTTPException(500, f"Failed to index annotations: {e}")

    if bbox is None:
        # Stats-only response (no labels payload)
        return {
            "job_id": job_id,
            "layer": layer,
            "total_labels": label_index.label_count,
            "memory_mb": round(label_index.memory_estimate_mb, 1),
        }

    # Parse bbox and query R-tree
    try:
        parts = [float(x) for x in bbox.split(",")]
        assert len(parts) == 4
        bbox_tuple = tuple(parts)
    except (ValueError, AssertionError):
        raise HTTPException(400, "bbox must be minX,minY,maxX,maxY")

    labels = await asyncio.to_thread(
        spatial_manager.query_bbox, job_id, layer, bbox_tuple
    )

    return Response(
        content=json.dumps(labels, separators=(',', ':')),
        media_type="application/json",
        headers={"Cache-Control": "no-cache"},  # labels can change via edits
    )


@app.get("/labels/stats")
async def label_stats(blob_path: str):
    """Quick stats without building full index."""
    job_id = blob_path.split('/')[-2] if '/' in blob_path else 'unknown'
    layer = blob_path.split('_')[-1].split('.')[0] if '_' in blob_path else 'base'
    
    try:
        local_path = await label_blob_cache.get(blob_path)
    except Exception:
        raise HTTPException(404, "Annotation file not found")

    size = os.path.getsize(local_path)
    return {
        "job_id": job_id,
        "layer": layer,
        "compressed_size_mb": round(size / (1024 * 1024), 1),
        "is_indexed": spatial_manager._cache_key(job_id, layer) in spatial_manager._indexes,
    }


@app.post("/labels/invalidate")
async def invalidate_cache(blob_path: str):
    """Called by azure-studio after a save to bust the cache."""
    job_id = blob_path.split('/')[-2] if '/' in blob_path else 'unknown'
    layer = blob_path.split('_')[-1].split('.')[0] if '_' in blob_path else 'base'
    
    key = spatial_manager._cache_key(job_id, layer)
    if key in spatial_manager._indexes:
        del spatial_manager._indexes[key]
        
    # Also remove the cached blob so next request re-downloads
    label_blob_cache.remove(blob_path)
    return {"invalidated": True}
