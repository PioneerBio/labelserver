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
    OPEN_PREFIXES = ("/labels/tiles/",)  # tile images loaded by OSD <img> tags (no auth header)

    async def dispatch(self, request: Request, call_next):
        # Dev mode: no auth configured
        if not settings.jwt_secret and not settings.api_key:
            return await call_next(request)

        # Open paths always pass
        if request.url.path in self.OPEN_PATHS:
            return await call_next(request)
        if any(request.url.path.startswith(p) for p in self.OPEN_PREFIXES):
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
async def get_labels(blob_path: str, bbox: str | None = None, max_labels: int | None = None):
    """Return labels within a bounding box.

    blob_path: full Azure Blob path (used as cache key directly)
    bbox format: minX,minY,maxX,maxY (image pixel coordinates)
    max_labels: if set, returns simplified centroids when result exceeds this count (LOD)
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

    if max_labels and max_labels > 0:
        labels = await asyncio.to_thread(
            spatial_manager.query_bbox_lod, blob_path, bbox_tuple, max_labels
        )
    else:
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


@app.get("/labels/status")
async def label_status(blob_path: str):
    """Lightweight status check — no downloads or indexing triggered.

    Returns the current state of blob download and spatial index
    so the frontend can show progress indicators.
    """
    cache_status = label_blob_cache.get_status(blob_path)
    is_indexed = blob_path in spatial_manager._indexes
    index_status = spatial_manager.get_index_status(blob_path)

    result = {
        "blob_path": blob_path,
        **cache_status.to_dict(),
        "indexed": is_indexed,
    }
    if index_status:
        result["index"] = index_status
    if is_indexed:
        li = spatial_manager._indexes.get(blob_path)
        if li:
            result["total_labels"] = li.label_count
            result["memory_mb"] = round(li.memory_estimate_mb, 1)
    return result


@app.get("/labels/tiles/info")
async def tile_info(blob_path: str):
    """Return DZI-like metadata for the label tile pyramid."""
    try:
        local_path = await label_blob_cache.get(blob_path)
    except Exception as e:
        raise HTTPException(404, f"Annotation file not found: {blob_path}")

    try:
        label_index = await asyncio.to_thread(
            spatial_manager.get_or_build, blob_path, local_path
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to index: {e}")

    info = spatial_manager.get_tile_info(blob_path)
    if not info:
        raise HTTPException(404, "No tile info available")
    return info


@app.get("/labels/tiles/{level}/{col}_{row}.png")
async def get_tile(blob_path: str, level: int, col: int, row: int, request: Request):
    """JIT-render a label tile as RGBA PNG."""
    # ETag support
    etag = spatial_manager.tile_etag(blob_path, level, col, row)
    if_none_match = request.headers.get("if-none-match", "")
    if if_none_match == etag:
        return Response(status_code=304)

    # Ensure index is built
    try:
        local_path = await label_blob_cache.get(blob_path)
    except Exception:
        raise HTTPException(404, "Annotation file not found")

    try:
        await asyncio.to_thread(
            spatial_manager.get_or_build, blob_path, local_path
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to index: {e}")

    # Render tile
    png_bytes = await asyncio.to_thread(
        spatial_manager.render_tile, blob_path, level, col, row
    )

    if png_bytes is None:
        # Empty tile — transparent
        return Response(status_code=204)

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "ETag": etag,
            "Cache-Control": "public, max-age=3600",
        },
    )


@app.post("/labels/invalidate")
async def invalidate_cache(blob_path: str):
    """Called by azure-studio after a save to bust the cache."""
    if blob_path in spatial_manager._indexes:
        del spatial_manager._indexes[blob_path]

    # Also remove the cached blob so next request re-downloads
    label_blob_cache.remove(blob_path)
    return {"invalidated": True}
