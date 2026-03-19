import gzip
import hashlib
import io
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import lru_cache
from rtree import index as rtree_index

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

@dataclass
class LabelIndex:
    """In-memory spatial index for one annotation file."""
    blob_path: str
    labels: list[dict]                         # raw label dicts
    centroids: list[tuple[float, float]]       # pre-computed (cx, cy) per label
    rtree: rtree_index.Index = field(repr=False)
    label_count: int = 0
    image_width: int = 0                       # bounding box of all labels
    image_height: int = 0
    memory_estimate_mb: float = 0.0
    last_accessed: float = 0.0

class SpatialIndexManager:
    """LRU cache of per-blob spatial indexes built from .json.gz files."""

    def __init__(self, max_indexes: int = 50, max_memory_mb: float = 8192):
        self.max_indexes = max_indexes
        self.max_memory_mb = max_memory_mb
        self._indexes: OrderedDict[str, LabelIndex] = OrderedDict()
        self._index_status: dict[str, dict] = {}  # blob_path -> {"status", "progress", "error"}

    def get_index_status(self, blob_path: str) -> dict | None:
        """Get current indexing status without triggering a build."""
        if blob_path in self._indexes:
            return {"status": "ready"}
        return self._index_status.get(blob_path)

    def get_or_build(self, blob_path: str, local_path: str) -> LabelIndex:
        """Get existing index or build from local file."""
        if blob_path in self._indexes:
            self._indexes.move_to_end(blob_path)
            self._indexes[blob_path].last_accessed = time.time()
            return self._indexes[blob_path]

        # Build new index
        self._index_status[blob_path] = {"status": "indexing", "progress": 0.0}
        try:
            label_index = self._build_index(blob_path, local_path)
        except Exception as e:
            self._index_status[blob_path] = {"status": "error", "error": str(e)}
            raise
        self._indexes[blob_path] = label_index
        self._indexes.move_to_end(blob_path)
        self._index_status[blob_path] = {"status": "ready"}
        self._evict_if_needed()
        return label_index

    def _build_index(self, blob_path: str, local_path: str) -> LabelIndex:
        """Parse annotation file and build R-tree index with pre-computed centroids."""
        logger.info(f"Building spatial index: {blob_path}")

        # Handle both .json.gz and plain .json files
        if local_path.endswith('.gz'):
            with gzip.open(local_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(local_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        labels = data.get('labels', data) if isinstance(data, dict) else data

        # Read image dimensions from metadata (primary) or fall back to label extent
        meta_width = data.get('image_width') if isinstance(data, dict) else None
        meta_height = data.get('image_height') if isinstance(data, dict) else None

        idx = rtree_index.Index()
        centroids: list[tuple[float, float]] = []
        max_x, max_y = 0.0, 0.0

        for i, label in enumerate(labels):
            bbox = self._compute_bbox(label)
            if bbox:
                idx.insert(i, bbox)
                centroids.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
                max_x = max(max_x, bbox[2])
                max_y = max(max_y, bbox[3])
            else:
                centroids.append((0.0, 0.0))

        image_width = int(meta_width) if meta_width else int(math.ceil(max_x))
        image_height = int(meta_height) if meta_height else int(math.ceil(max_y))
        dims_source = "metadata" if meta_width and meta_height else "label_extent"

        # Estimate memory from file size on disk
        file_size = os.path.getsize(local_path)
        if local_path.endswith('.gz'):
            mem_mb = (file_size * 8) / (1024 * 1024)
        else:
            mem_mb = (file_size * 2) / (1024 * 1024)
        logger.info(f"Indexed {len(labels)} labels, ~{mem_mb:.0f} MB, dims={image_width}x{image_height} ({dims_source})")

        return LabelIndex(
            blob_path=blob_path,
            labels=labels,
            centroids=centroids,
            rtree=idx,
            label_count=len(labels),
            image_width=image_width,
            image_height=image_height,
            memory_estimate_mb=mem_mb,
            last_accessed=time.time()
        )

    def _compute_bbox(self, label: dict) -> tuple | None:
        """Extract bounding box (minX, minY, maxX, maxY) from label geometry."""
        # Polygons: have 'regions' field  [[{x,y}, {x,y}, ...], ...]
        regions = label.get('regions')
        if regions:
            xs, ys = [], []
            for ring in regions:
                for pt in ring:
                    xs.append(pt['x'])
                    ys.append(pt['y'])
            if not xs: return None
            return (min(xs), min(ys), max(xs), max(ys))

        # Points: have 'position' field {x, y}
        pos = label.get('position')
        if pos:
            return (pos['x'], pos['y'], pos['x'], pos['y'])

        # Boxes: have 'centre' + 'size'
        centre = label.get('centre')
        size = label.get('size')
        if centre and size:
            half_w, half_h = size['x'] / 2, size['y'] / 2
            return (centre['x'] - half_w, centre['y'] - half_h,
                    centre['x'] + half_w, centre['y'] + half_h)

        return None

    def query_bbox(self, blob_path: str, bbox: tuple) -> list[dict]:
        """Return labels intersecting the given bounding box."""
        li = self._indexes.get(blob_path)
        if not li:
            return []

        self._indexes.move_to_end(blob_path)
        li.last_accessed = time.time()

        indices = list(li.rtree.intersection(bbox))
        return [li.labels[i] for i in indices]

    def query_bbox_lod(self, blob_path: str, bbox: tuple, max_labels: int) -> list[dict]:
        """Return labels intersecting bbox, simplified to centroids if over max_labels."""
        li = self._indexes.get(blob_path)
        if not li:
            return []

        self._indexes.move_to_end(blob_path)
        li.last_accessed = time.time()

        indices = list(li.rtree.intersection(bbox))

        if len(indices) <= max_labels:
            return [li.labels[i] for i in indices]

        # Over budget: subsample evenly and return centroid-only representations
        step = max(1, len(indices) // max_labels)
        sampled = indices[::step][:max_labels]

        result = []
        for i in sampled:
            label = li.labels[i]
            cx, cy = li.centroids[i]
            simplified = {
                "_id": label.get("_id", str(i)),
                "label_class": label.get("label_class", ""),
                "label_type": label.get("label_type", "cell"),
                "source": label.get("source", ""),
                "position": {"x": cx, "y": cy},
                "_simplified": True,
            }
            result.append(simplified)
        return result

    # ── Tile rendering ──────────────────────────────────────────────────────

    TILE_SIZE = 512
    FILL_COLOR = (0, 120, 255, 80)    # semi-transparent blue fill
    STROKE_COLOR = (0, 120, 255, 200) # solid blue stroke
    POINT_RADIUS = 3

    def get_tile_info(self, blob_path: str) -> dict | None:
        """Return DZI-like metadata for building the tile pyramid."""
        li = self._indexes.get(blob_path)
        if not li:
            return None
        w, h = li.image_width, li.image_height
        if w == 0 or h == 0:
            return None
        max_dim = max(w, h)
        max_level = max(0, int(math.ceil(math.log2(max_dim / self.TILE_SIZE))))
        return {
            "width": w,
            "height": h,
            "tile_size": self.TILE_SIZE,
            "max_level": max_level,
            "total_labels": li.label_count,
        }

    def render_tile(self, blob_path: str, level: int, col: int, row: int) -> bytes | None:
        """JIT-render a label tile as RGBA PNG."""
        li = self._indexes.get(blob_path)
        if not li:
            return None

        self._indexes.move_to_end(blob_path)
        li.last_accessed = time.time()

        ts = self.TILE_SIZE
        max_dim = max(li.image_width, li.image_height)
        max_level = max(0, int(math.ceil(math.log2(max_dim / ts))))

        # Scale factor: at max_level each tile = TILE_SIZE pixels of the image
        # At level 0, one tile covers the entire image
        scale = ts * (2 ** (max_level - level)) # image pixels per tile
        bbox = (col * scale, row * scale, (col + 1) * scale, (row + 1) * scale)

        # Query R-tree
        indices = list(li.rtree.intersection(bbox))
        if not indices:
            return None  # empty tile, caller returns 204

        # Coordinate transform: image coords -> tile pixel coords
        ox, oy = bbox[0], bbox[1]
        px_per_unit = ts / scale  # tile pixels per image pixel

        img = Image.new('RGBA', (ts, ts), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        for i in indices:
            label = li.labels[i]
            regions = label.get('regions')
            if regions:
                for ring in regions:
                    if len(ring) < 3:
                        continue
                    poly = [(int((pt['x'] - ox) * px_per_unit),
                             int((pt['y'] - oy) * px_per_unit)) for pt in ring]
                    draw.polygon(poly, fill=self.FILL_COLOR, outline=self.STROKE_COLOR)
            else:
                # Point or centroid
                cx, cy = li.centroids[i]
                px = int((cx - ox) * px_per_unit)
                py = int((cy - oy) * px_per_unit)
                r = self.POINT_RADIUS
                draw.ellipse([px - r, py - r, px + r, py + r],
                             fill=self.FILL_COLOR, outline=self.STROKE_COLOR)

        buf = io.BytesIO()
        img.save(buf, format='PNG', optimize=False)
        return buf.getvalue()

    def tile_etag(self, blob_path: str, level: int, col: int, row: int) -> str:
        """Compute a stable ETag for cache validation."""
        li = self._indexes.get(blob_path)
        key = f"{blob_path}:{li.label_count if li else 0}:{level}:{col}:{row}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _evict_if_needed(self):
        total_mb = sum(li.memory_estimate_mb for li in self._indexes.values())
        while (len(self._indexes) > self.max_indexes or total_mb > self.max_memory_mb) and self._indexes:
            oldest_key, oldest = next(iter(self._indexes.items()))
            total_mb -= oldest.memory_estimate_mb
            del self._indexes[oldest_key]
            logger.info(f"Evicted index: {oldest_key}")

    @property
    def stats(self) -> dict:
        return {
            "indexed_jobs": len(self._indexes),
            "total_labels": sum(li.label_count for li in self._indexes.values()),
            "total_memory_mb": round(sum(li.memory_estimate_mb for li in self._indexes.values()), 1),
        }
