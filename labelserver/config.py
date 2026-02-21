from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    azure_storage_connection_string: str
    azure_container_name: str = "media"
    cache_dir: str = "/data/label-cache"     # disk cache for .json.gz files
    cache_max_size_gb: float = 50.0          # smaller than tile cache
    max_indexed_jobs: int = 50               # max R-tree indexes in memory
    max_index_memory_mb: float = 8192        # 8GB ceiling for all indexes
    api_key: str = ""                        # static key for server-to-server calls
    jwt_secret: str = ""                     # shared secret for JWT validation (browser clients)
    django_api_url: str = ""                 # azure-studio URL for writeback
    chunk_size: int = 4096                   # spatial chunk size in image pixels

    class Config:
        env_file = ".env"

settings = Settings()
