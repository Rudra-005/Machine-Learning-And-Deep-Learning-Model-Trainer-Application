"""
Cache manager for in-memory caching
"""
from functools import lru_cache
from typing import Any
from app.utils.logger import logger

class CacheManager:
    """Simple in-memory cache manager"""
    
    _cache = {}
    
    @staticmethod
    def set(key: str, value: Any, ttl: int = None) -> None:
        """
        Store value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (not implemented in basic version)
        """
        CacheManager._cache[key] = value
        logger.debug(f"Cached: {key}")
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """
        Retrieve value from cache
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        value = CacheManager._cache.get(key, default)
        logger.debug(f"Retrieved from cache: {key}")
        return value
    
    @staticmethod
    def delete(key: str) -> None:
        """
        Delete value from cache
        
        Args:
            key: Cache key
        """
        if key in CacheManager._cache:
            del CacheManager._cache[key]
            logger.debug(f"Deleted from cache: {key}")
    
    @staticmethod
    def clear() -> None:
        """Clear entire cache"""
        CacheManager._cache.clear()
        logger.info("Cache cleared")
