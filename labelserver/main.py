import asyncio
import json
import logging
import os
import jwt as pyjwt
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

class AuthMiddleware(BaseHTTPMiddleware):
    """Dual auth: JWT (browser clients) + static API key (server-to-server).

    If neither jwt_secret nor api_key is configured, all requests pass (dev mode).
    """
    OPEN_PATHS = {"/health", "/docs", "/openapi.json"}

    async def dispatch(self, request: Request, call_next):
        # Dev mode: no auth configured
        if not settings.jwt_secret and not settings.api_key:
            return await call_next(request)

        # Open paths always pass
        if request.url.path in self.OPEN_PATHS:
            return await call_next(request)

        auth = request.headers.get("authorization", "")

        if auth.startswith("Bearer "):
            token = auth[7:]

            # 1. Static API key check (cheap string compare — server-to-server)
            if settings.api_key and token == settings.api_key:
                return await call_next(request)

            # 2. JWT validation (browser clients)
            if settings.jwt_secret:
                try:
                    payload = pyjwt.decode(
                        token,
                        settings.jwt_secret,
                        algorithms=["HS256"],
                        options={"require": ["exp", "sub", "iss"]},
                    )
                    if payload.get("iss") != "azure-studio":
                        return Response(status_code=401, content="Invalid token issuer")
                    request.state.user_id = payload.get("sub")
                    request.state.user_email = payload.get("email", "")
                    return await call_next(request)
                except pyjwt.ExpiredSignatureError:
                    return Response(status_code=401, content="Token expired")
                except pyjwt.InvalidTokenError:
                    pass  # Fall through

        # 3. Query param fallback (API key only)
        token_param = request.query_params.get("token", "")
        if settings.api_key and token_param == settings.api_key:
            return await call_next(request)

        return Response(status_code=401, content="Unauthorized")

app.add_middleware(AuthMiddleware)
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

    blob_path: full Azure Blob path (used as cache key directly)
    bbox format: minX,minY,maxX,maxY (image pixel coordinates)
    If bbox is omitted, returns metadata/stats only (not all labels).
    """
    # Ensure blob is cached locally
    try:
        local_path = await label_blob_cache.get(blob_path)
    except Exception as e:
        logger.error(f"Blob not found: {blob_path}: {e}")
        raise HTTPException(404, f"Annotation file not found: {blob_path}")

    # Ensure spatial index is built (use blob_path as cache key directly)
    try:
        label_index = await asyncio.to_thread(
            spatial_manager.get_or_build, blob_path, local_path
        )
    except Exception as e:
        logger.error(f"Index build failed: {e}")
        raise HTTPException(500, f"Failed to index annotations: {e}")

    if bbox is None:
        # Stats-only response (no labels payload)
        return {
            "blob_path": blob_path,
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
        spatial_manager.query_bbox, blob_path, bbox_tuple
    )

    return Response(
        content=json.dumps(labels, separators=(',', ':')),
        media_type="application/json",
        headers={"Cache-Control": "no-cache"},  # labels can change via edits
    )


@app.get("/labels/stats")
async def label_stats(blob_path: str):
    """Quick stats without building full index."""
    try:
        local_path = await label_blob_cache.get(blob_path)
    except Exception:
        raise HTTPException(404, "Annotation file not found")

    size = os.path.getsize(local_path)
    return {
        "blob_path": blob_path,
        "compressed_size_mb": round(size / (1024 * 1024), 1),
        "is_indexed": blob_path in spatial_manager._indexes,
    }


@app.post("/labels/invalidate")
async def invalidate_cache(blob_path: str):
    """Called by azure-studio after a save to bust the cache."""
    if blob_path in spatial_manager._indexes:
        del spatial_manager._indexes[blob_path]

    # Also remove the cached blob so next request re-downloads
    label_blob_cache.remove(blob_path)
    return {"invalidated": True}
