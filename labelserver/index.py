import gzip
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from rtree import index as rtree_index

logger = logging.getLogger(__name__)

@dataclass
class LabelIndex:
    """In-memory spatial index for one job+layer's annotations."""
    job_id: str
    layer: str
    labels: list[dict]                         # raw label dicts
    rtree: rtree_index.Index = field(repr=False)
    label_count: int = 0
    memory_estimate_mb: float = 0.0
    last_accessed: float = 0.0

class SpatialIndexManager:
    """LRU cache of per-job spatial indexes built from .json.gz files."""

    def __init__(self, max_indexes: int = 50, max_memory_mb: float = 8192):
        self.max_indexes = max_indexes
        self.max_memory_mb = max_memory_mb
        self._indexes: OrderedDict[str, LabelIndex] = OrderedDict()

    def _cache_key(self, job_id: str, layer: str) -> str:
        return f"{job_id}:{layer or 'base'}"

    def get_or_build(self, job_id: str, layer: str, gz_path: str) -> LabelIndex:
        """Get existing index or build from .json.gz file."""
        key = self._cache_key(job_id, layer)

        if key in self._indexes:
            self._indexes.move_to_end(key)
            self._indexes[key].last_accessed = time.time()
            return self._indexes[key]

        # Build new index
        label_index = self._build_index(job_id, layer, gz_path)
        self._indexes[key] = label_index
        self._indexes.move_to_end(key)
        self._evict_if_needed()
        return label_index

    def _build_index(self, job_id: str, layer: str, gz_path: str) -> LabelIndex:
        """Stream-parse .json.gz and build R-tree index."""
        logger.info(f"Building spatial index: {job_id}/{layer}")

        with gzip.open(gz_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        labels = data.get('labels', data) if isinstance(data, dict) else data
        idx = rtree_index.Index()

        for i, label in enumerate(labels):
            bbox = self._compute_bbox(label)
            if bbox:
                idx.insert(i, bbox)

        mem_mb = len(json.dumps(labels)) / (1024 * 1024)  # rough estimate
        logger.info(f"Indexed {len(labels)} labels, ~{mem_mb:.0f} MB")

        return LabelIndex(
            job_id=job_id,
            layer=layer,
            labels=labels,
            rtree=idx,
            label_count=len(labels),
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

    def query_bbox(self, job_id: str, layer: str, bbox: tuple) -> list[dict]:
        """Return labels intersecting the given bounding box."""
        key = self._cache_key(job_id, layer)
        li = self._indexes.get(key)
        if not li:
            return []

        self._indexes.move_to_end(key)
        li.last_accessed = time.time()

        indices = list(li.rtree.intersection(bbox))
        return [li.labels[i] for i in indices]

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
