"""
Semantic caching for LLM planner using sentence-transformers and FAISS.

Provides fast similarity-based cache lookups to reduce LLM API calls
and improve response latency for similar prompts.
"""

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from challenge.models.plan import Plan

if TYPE_CHECKING:
    from challenge.planner.protocol import Planner

logger = logging.getLogger(__name__)


class CacheEntry(BaseModel):
    """
    Single cache entry with prompt, plan, and metadata.

    Attributes:
        prompt: Original user prompt
        plan: Cached execution plan
        embedding: Embedding vector for similarity search
        hits: Number of times this entry was retrieved from cache

    """

    prompt: str = Field(..., description="Original prompt")
    plan: Plan = Field(..., description="Cached plan")
    embedding: list[float] = Field(..., description="Prompt embedding vector")
    hits: int = Field(default=0, ge=0, description="Cache hit count")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class CacheMetrics(BaseModel):
    """
    Cache performance metrics.

    Attributes:
        total_requests: Total number of cache lookup requests
        hits: Number of successful cache hits
        misses: Number of cache misses
        hit_rate: Cache hit rate (0.0 to 1.0)
        total_entries: Current number of cached entries

    """

    total_requests: int = Field(default=0, ge=0, description="Total requests")
    hits: int = Field(default=0, ge=0, description="Cache hits")
    misses: int = Field(default=0, ge=0, description="Cache misses")
    hit_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Hit rate")
    total_entries: int = Field(default=0, ge=0, description="Total entries")

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=False,
        strict=True,
        extra="forbid",
    )


class SemanticCache:
    """
    Semantic similarity-based cache for LLM planner results.

    Uses sentence-transformers (all-MiniLM-L6-v2) for embedding generation
    and FAISS for fast similarity search with cosine similarity.

    Features:
    - Fast similarity search with FAISS index
    - Configurable similarity threshold (default: 0.85)
    - Cache hit/miss tracking for metrics
    - Automatic index updates on cache additions

    Example:
        >>> cache = SemanticCache(similarity_threshold=0.85)
        >>> cache.add("calculate 2 + 2", plan)
        >>> cached_plan = cache.get("compute 2 plus 2")  # Similar prompt
        >>> metrics = cache.get_metrics()

    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
    ):
        """
        Initialize semantic cache with embedding model and FAISS index.

        Args:
            similarity_threshold: Minimum cosine similarity for cache hit (0.0-1.0)
            model_name: HuggingFace model name for embeddings
            cache_dir: Optional directory for model caching

        """
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name

        # Initialize embedding model
        logger.info(f"Loading embedding model: {model_name}")
        cache_folder_str = str(cache_dir) if cache_dir is not None else None
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder_str)

        # Get embedding dimension from model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize FAISS index for cosine similarity
        # Using IndexFlatIP (Inner Product) which with normalized vectors equals cosine similarity
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # type: ignore[possibly-unbound-attribute]

        # Storage for cache entries
        self.entries: list[CacheEntry] = []

        # Metrics tracking
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0

        logger.info(f"Semantic cache initialized: dim={self.embedding_dim}, threshold={similarity_threshold}")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding vector for text using sentence-transformers.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector

        """
        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Normalize for cosine similarity (required for IndexFlatIP)
        faiss.normalize_L2(embedding.reshape(1, -1))

        return embedding

    def add(self, prompt: str, plan: Plan) -> None:
        """
        Add prompt-plan pair to cache with similarity indexing.

        Args:
            prompt: User prompt
            plan: Execution plan to cache

        """
        # Generate embedding
        embedding = self._generate_embedding(prompt)

        # Create cache entry
        entry = CacheEntry(
            prompt=prompt,
            plan=plan,
            embedding=embedding.tolist(),
        )

        # Add to storage
        self.entries.append(entry)

        # Add to FAISS index
        self.index.add(embedding.reshape(1, -1))

        logger.debug(f"Added to cache: '{prompt[:50]}...' (total entries: {len(self.entries)})")

    def get(self, prompt: str) -> Plan | None:
        """
        Retrieve cached plan for similar prompt using semantic search.

        Args:
            prompt: User prompt to search for

        Returns:
            Cached plan if similarity above threshold, None otherwise

        """
        self.total_requests += 1

        # Return None if cache is empty
        if len(self.entries) == 0:
            self.cache_misses += 1
            logger.debug("Cache miss: empty cache")
            return None

        # Generate embedding for query
        query_embedding = self._generate_embedding(prompt)

        # Search FAISS index for most similar entry
        # k=1 returns single nearest neighbor
        similarities, indices = self.index.search(query_embedding.reshape(1, -1), k=1)

        # Extract top result
        similarity = float(similarities[0][0])
        index = int(indices[0][0])

        # Check if similarity meets threshold
        if similarity >= self.similarity_threshold:
            # Cache hit
            self.cache_hits += 1
            entry = self.entries[index]
            entry.hits += 1

            logger.info(f"Cache HIT: '{prompt[:50]}...' → '{entry.prompt[:50]}...' (similarity: {similarity:.3f})")

            return entry.plan

        # Cache miss
        self.cache_misses += 1
        logger.debug(
            f"Cache MISS: '{prompt[:50]}...' (best similarity: {similarity:.3f} < {self.similarity_threshold})"
        )
        return None

    def get_metrics(self) -> CacheMetrics:
        """
        Get current cache performance metrics.

        Returns:
            CacheMetrics with hit rate and entry statistics

        """
        hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0

        return CacheMetrics(
            total_requests=self.total_requests,
            hits=self.cache_hits,
            misses=self.cache_misses,
            hit_rate=hit_rate,
            total_entries=len(self.entries),
        )

    def clear(self) -> None:
        """Clear all cached entries and reset metrics."""
        self.entries.clear()
        self.index.reset()
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache cleared")

    def size(self) -> int:
        """Get current number of cached entries."""
        return len(self.entries)


class CachingPlanner:
    """
    Wrapper that adds semantic caching to any planner implementation.

    Implements the Planner protocol while delegating to wrapped planner
    for cache misses. Automatically caches generated plans for future reuse.

    Example:
        >>> base_planner = LLMPlanner(api_key="...")
        >>> cached_planner = CachingPlanner(base_planner)
        >>> orchestrator = Orchestrator(planner=cached_planner)

    """

    def __init__(
        self,
        planner: "Planner",
        cache: SemanticCache | None = None,
        similarity_threshold: float = 0.85,
    ):
        """
        Initialize caching planner wrapper.

        Args:
            planner: Base planner to wrap with caching
            cache: Optional existing cache instance (creates new if None)
            similarity_threshold: Similarity threshold for cache hits

        """
        self.planner = planner
        self.cache = cache if cache is not None else SemanticCache(similarity_threshold=similarity_threshold)
        logger.info(
            f"CachingPlanner initialized with {type(planner).__name__} (threshold: {self.cache.similarity_threshold})"
        )

    async def create_plan(self, prompt: str) -> Plan:
        """
        Create plan with caching - check cache first, generate on miss.

        Args:
            prompt: Natural language task description

        Returns:
            Plan from cache (if hit) or freshly generated (if miss)

        Raises:
            ValueError: If prompt is invalid or cannot be parsed

        """
        # Check cache first
        cached_plan = self.cache.get(prompt)
        if cached_plan is not None:
            logger.info(f"Cache HIT for prompt: '{prompt[:50]}...'")
            return cached_plan

        # Cache miss - delegate to base planner
        logger.info(f"Cache MISS for prompt: '{prompt[:50]}...', generating plan")

        # Handle both sync and async planners
        if inspect.iscoroutinefunction(self.planner.create_plan):
            plan = await self.planner.create_plan(prompt)
        else:
            plan = self.planner.create_plan(prompt)

        # Add to cache for future use
        self.cache.add(prompt, plan)

        return plan

    def get_metrics(self) -> CacheMetrics:
        """Get cache performance metrics."""
        return self.cache.get_metrics()

    def clear_cache(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()
