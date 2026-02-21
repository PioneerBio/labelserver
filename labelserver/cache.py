import asyncio
import logging
import os
import shutil
from collections import OrderedDict

from azure.storage.blob import BlobServiceClient

from .config import settings

logger = logging.getLogger(__name__)


class LabelBlobCache:
    """LRU disk cache for .json.gz annotation files from Azure Blob."""
    
    def __init__(self):
        self.cache_dir = settings.cache_dir
        self.max_bytes = int(settings.cache_max_size_gb * 1024 * 1024 * 1024)
        self._locks: dict[str, asyncio.Lock] = {}
        self._lock_guard = asyncio.Lock()
        self._files: OrderedDict[str, int] = OrderedDict()  # path -> size
        self._blob_client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        self._container = self._blob_client.get_container_client(
            settings.azure_container_name
        )
        os.makedirs(self.cache_dir, exist_ok=True)
        self._scan_existing()

    def _scan_existing(self):
        """Rebuild LRU tracking from existing cached files."""
        entries = []
        for root, _, files in os.walk(self.cache_dir):
            for f in files:
                full = os.path.join(root, f)
                try:
                    stat = os.stat(full)
                    entries.append((full, stat.st_size, stat.st_atime))
                except OSError:
                    continue
        # Sort by access time (oldest first = evict first)
        entries.sort(key=lambda e: e[2])
        for path, size, _ in entries:
            rel = os.path.relpath(path, self.cache_dir)
            self._files[rel] = size
        total_mb = sum(self._files.values()) / (1024 * 1024)
        logger.info(f"Cache scan: {len(self._files)} files, {total_mb:.0f} MB")

    async def _get_lock(self, blob_path: str) -> asyncio.Lock:
        async with self._lock_guard:
            if blob_path not in self._locks:
                self._locks[blob_path] = asyncio.Lock()
            return self._locks[blob_path]

    def _local_path(self, blob_path: str) -> str:
        return os.path.join(self.cache_dir, blob_path)

    async def get(self, blob_path: str) -> str:
        """Get local path to cached .json.gz, downloading if needed."""
        local = self._local_path(blob_path)

        if os.path.exists(local):
            # Move to end of LRU
            if blob_path in self._files:
                self._files.move_to_end(blob_path)
            return local

        lock = await self._get_lock(blob_path)
        async with lock:
            # Double-check after acquiring lock
            if os.path.exists(local):
                if blob_path in self._files:
                    self._files.move_to_end(blob_path)
                return local

            logger.info(f"Downloading: {blob_path}")
            await asyncio.to_thread(self._download, blob_path, local)
            size = os.path.getsize(local)
            self._files[blob_path] = size
            self._files.move_to_end(blob_path)
            await asyncio.to_thread(self._evict_if_needed)
            logger.info(f"Cached: {blob_path} ({size / 1024 / 1024:.0f} MB)")
            return local

    def _download(self, blob_path: str, local_path: str):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tmp = local_path + ".tmp"
        blob = self._container.get_blob_client(blob_path)
        with open(tmp, "wb") as f:
            stream = blob.download_blob()
            stream.readinto(f)
        os.rename(tmp, local_path)

    def _evict_if_needed(self):
        total = sum(self._files.values())
        while total > self.max_bytes and self._files:
            oldest_key, oldest_size = next(iter(self._files.items()))
            local = self._local_path(oldest_key)
            try:
                os.remove(local)
                # Clean empty parent directories
                parent = os.path.dirname(local)
                while parent != self.cache_dir:
                    if not os.listdir(parent):
                        os.rmdir(parent)
                        parent = os.path.dirname(parent)
                    else:
                        break
            except OSError:
                pass
            del self._files[oldest_key]
            total -= oldest_size
            logger.info(f"Evicted: {oldest_key}")

    def remove(self, blob_path: str):
        local = self._local_path(blob_path)
        try:
            os.remove(local)
            if blob_path in self._files:
                del self._files[blob_path]
        except OSError:
            pass

    @property
    def total_cached_mb(self) -> float:
        return sum(self._files.values()) / (1024 * 1024)

    @property
    def file_count(self) -> int:
        return len(self._files)


label_blob_cache = LabelBlobCache()
