import gzip
import json
import logging
import os
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from rtree import index as rtree_index

logger = logging.getLogger(__name__)

@dataclass
class LabelIndex:
    """In-memory spatial index for one annotation file."""
    blob_path: str
    labels: list[dict]                         # raw label dicts
    rtree: rtree_index.Index = field(repr=False)
    label_count: int = 0
    memory_estimate_mb: float = 0.0
    last_accessed: float = 0.0

class SpatialIndexManager:
    """LRU cache of per-blob spatial indexes built from .json.gz files."""

    def __init__(self, max_indexes: int = 50, max_memory_mb: float = 8192):
        self.max_indexes = max_indexes
        self.max_memory_mb = max_memory_mb
        self._indexes: OrderedDict[str, LabelIndex] = OrderedDict()

    def get_or_build(self, blob_path: str, local_path: str) -> LabelIndex:
        """Get existing index or build from local file."""
        if blob_path in self._indexes:
            self._indexes.move_to_end(blob_path)
            self._indexes[blob_path].last_accessed = time.time()
            return self._indexes[blob_path]

        # Build new index
        label_index = self._build_index(blob_path, local_path)
        self._indexes[blob_path] = label_index
        self._indexes.move_to_end(blob_path)
        self._evict_if_needed()
        return label_index

    def _build_index(self, blob_path: str, local_path: str) -> LabelIndex:
        """Parse annotation file and build R-tree index."""
        logger.info(f"Building spatial index: {blob_path}")

        # Handle both .json.gz and plain .json files
        if local_path.endswith('.gz'):
            with gzip.open(local_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(local_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        labels = data.get('labels', data) if isinstance(data, dict) else data
        idx = rtree_index.Index()

        for i, label in enumerate(labels):
            bbox = self._compute_bbox(label)
            if bbox:
                idx.insert(i, bbox)

        # Estimate memory from file size on disk (avoids re-serializing all labels)
        file_size = os.path.getsize(local_path)
        # Gzipped files are ~5-10x smaller than in-memory; plain JSON ~2x (overhead of Python dicts)
        if local_path.endswith('.gz'):
            mem_mb = (file_size * 8) / (1024 * 1024)
        else:
            mem_mb = (file_size * 2) / (1024 * 1024)
        logger.info(f"Indexed {len(labels)} labels, ~{mem_mb:.0f} MB (estimated)")

        return LabelIndex(
            blob_path=blob_path,
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

    def query_bbox(self, blob_path: str, bbox: tuple) -> list[dict]:
        """Return labels intersecting the given bounding box."""
        li = self._indexes.get(blob_path)
        if not li:
            return []

        self._indexes.move_to_end(blob_path)
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
