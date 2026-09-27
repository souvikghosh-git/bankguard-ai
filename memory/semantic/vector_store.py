"""
BankGuard AI — Semantic Memory (pgvector).

Stores and retrieves banking policies, SOPs and resolution patterns
as dense vector embeddings for similarity search.

Embedding model: BAAI/bge-small-en-v1.5 (384-dim, CPU-friendly)
Storage:         PostgreSQL memory.semantic_store with pgvector HNSW index

Workflow:
  1. Ingest:  chunk policy → embed → upsert into semantic_store
  2. Retrieve: embed query → ANN search → return top-k chunks
"""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import Any

import structlog

from config import settings

log = structlog.get_logger(__name__)

# ── Embedding model (lazy load) ───────────────────────────────────────────────


@lru_cache(maxsize=1)
def _load_model() -> Any:
    """Load the sentence-transformer model (once per process)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model, device=settings.embedding_device)
    log.info("embedding_model_loaded", model=settings.embedding_model)
    return model


async def get_embedding(text: str) -> list[float]:
    """
    Compute embedding for text.
    Runs in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()
    model = await loop.run_in_executor(None, _load_model)
    embedding = await loop.run_in_executor(
        None,
        lambda: model.encode(text, normalize_embeddings=True).tolist(),
    )
    return embedding


# ── Semantic store ────────────────────────────────────────────────────────────


class SemanticMemory:
    """
    Semantic memory backed by pgvector.

    Usage:
        sm = SemanticMemory(db_pool)
        await sm.ingest_policy(policy_ref="PAY-REC-101", title=..., content=..., category=...)
        results = await sm.search("beneficiary bank timeout IMPS", top_k=3)
    """

    def __init__(self, db: Any) -> None:
        self.db = db

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def ingest_policy(
        self,
        policy_ref: str,
        title: str,
        content: str,
        category: str,
        chunk_size: int = 800,
    ) -> int:
        """
        Chunk a policy document, embed each chunk, store in semantic_store.
        Returns the number of chunks stored.
        """
        chunks = self._chunk_text(content, chunk_size)
        count = 0
        for i, chunk in enumerate(chunks):
            chunk_text = f"{title}\n\n{chunk}"
            embedding = await get_embedding(chunk_text)
            metadata = {
                "policy_ref": policy_ref,
                "title": title,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            await self._upsert(
                source_type="POLICY",
                source_ref=f"{policy_ref}-{i}",
                content=chunk_text,
                embedding=embedding,
                metadata=metadata,
            )
            count += 1
        log.info("policy_ingested", policy_ref=policy_ref, chunks=count)
        return count

    async def ingest_case_pattern(
        self,
        case_ref: str,
        root_cause: str,
        resolution: str,
    ) -> None:
        """Store a resolved case pattern for future similarity retrieval."""
        text = f"Root cause: {root_cause}. Resolution: {resolution}."
        embedding = await get_embedding(text)
        await self._upsert(
            source_type="CASE_PATTERN",
            source_ref=case_ref,
            content=text,
            embedding=embedding,
            metadata={"case_ref": case_ref, "root_cause": root_cause},
        )

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 3,
        source_type: str | None = None,
        min_score: float = 0.3,
    ) -> list[dict[str, Any]]:
        """
        ANN cosine similarity search.
        Returns list of {source_ref, content, score, metadata}.
        """
        query_embedding = await get_embedding(query)
        vec_str = f"[{','.join(str(x) for x in query_embedding)}]"

        try:
            async with self.db.acquire() as conn:
                type_filter = "AND source_type = $3" if source_type else ""
                params: list[Any] = [vec_str, top_k]
                if source_type:
                    params.append(source_type)

                rows = await conn.fetch(
                    f"""
                    SELECT source_ref, source_type, content, metadata,
                           1 - (embedding <=> $1::vector) AS score
                    FROM memory.semantic_store
                    WHERE 1 - (embedding <=> $1::vector) >= {min_score}
                    {type_filter}
                    ORDER BY embedding <=> $1::vector
                    LIMIT $2
                    """,
                    *params,
                )
                return [
                    {
                        "source_ref": r["source_ref"],
                        "source_type": r["source_type"],
                        "content": r["content"],
                        "score": float(r["score"]),
                        "metadata": json.loads(r["metadata"] or "{}"),
                    }
                    for r in rows
                ]
        except Exception as exc:
            log.warning("semantic_search_error", error=str(exc))
            return []

    # ── Bulk ingestion helper ─────────────────────────────────────────────────

    async def ingest_all_policies(self) -> int:
        """Ingest all active policies from banking.policies into semantic store."""
        total = 0
        try:
            async with self.db.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT policy_ref, title, content, category FROM banking.policies WHERE active"
                )
                for row in rows:
                    n = await self.ingest_policy(
                        policy_ref=row["policy_ref"],
                        title=row["title"],
                        content=row["content"],
                        category=row["category"],
                    )
                    total += n
            log.info("all_policies_ingested", total_chunks=total)
        except Exception as exc:
            log.error("ingest_all_policies_error", error=str(exc))
        return total

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _upsert(
        self,
        source_type: str,
        source_ref: str,
        content: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        vec_str = f"[{','.join(str(x) for x in embedding)}]"
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO memory.semantic_store
                    (source_type, source_ref, content, embedding, metadata)
                VALUES ($1, $2, $3, $4::vector, $5::jsonb)
                ON CONFLICT DO NOTHING
                """,
                source_type,
                source_ref,
                content,
                vec_str,
                json.dumps(metadata),
            )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 800) -> list[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks: list[str] = []
        overlap = 50
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + chunk_size])
            chunks.append(chunk)
            i += chunk_size - overlap
        return chunks or [text]
