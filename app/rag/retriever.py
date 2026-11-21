import asyncio
import re

from app.rag.ingest import create_embeddings, get_or_create_collection, load_catalog
from app.rag.bm25_index import BM25Index
from app.config import Settings

settings = Settings()

def rrf_fuse(vector_hits: list[dict], bm25_hits: list[dict], k: int = 60) -> list[dict]:
    hits_by_id: dict[str, dict] = {}
    scores_by_id: dict[str, float] = {}

    for hits in (vector_hits, bm25_hits):
        seen: set[str] = set()

        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            hits_by_id.setdefault(chunk_id, hit)
            scores_by_id[chunk_id] = (scores_by_id.get(chunk_id, 0.0) + 1 / (k + rank))

    ranked_ids = sorted(
        scores_by_id,
        key=scores_by_id.__getitem__,
        reverse=True,
    )

    return [{**hits_by_id[chunk_id], "rrf_score": scores_by_id[chunk_id]} for chunk_id in ranked_ids]


async def search_knowledge_base(
    query: str,
    mode: str,
    top_k: int = 5,
    strategy: str = "hybrid",
    manual_family: str | None = None,
):
    top_k = min(top_k, 5)
    candidate_k = top_k*2 if strategy == "hybrid" else top_k
    filters = {}
    if mode == "current":
        filters["citation_eligible"] = True
    if manual_family is not None:
        filters["manual_family"] = manual_family

    if mode == "history":
        catalog = await asyncio.to_thread(load_catalog)
        known_manuals = {
            document_id
            for versions in catalog["manual_families"].values()
            for document_id in versions.values()
            if isinstance(document_id, str)
        }
        explicit_manuals = [
            document_id for document_id in known_manuals
            if re.search(r"(?<![A-Za-z0-9-])" + re.escape(document_id) + r"(?![A-Za-z0-9-])",
                         query, flags=re.IGNORECASE)
        ]
        # A named single manual defines the historical lookup's scope. Do not
        # collapse cross-version comparisons to one document.
        if len(explicit_manuals) == 1:
            filters["document_id"] = explicit_manuals[0]

    bm25_where = filters or None
    chroma_where = bm25_where
    if len(filters) > 1:
        chroma_where = {"$and": [{key: value} for key, value in filters.items()]}

    if strategy in ("vector", "hybrid"):
        query_embeddings = await asyncio.to_thread(
            create_embeddings,
            [query]
        )
        query_embedding = query_embeddings[0]

        collection = await asyncio.to_thread(
            get_or_create_collection
        )

        result = await asyncio.to_thread(
            collection.query,
            query_embeddings = [query_embedding],
            n_results = candidate_k,
            where = chroma_where,
            include = ["documents", "metadatas", "distances"]
        )

        vector_hits  = []
        for chunk_id, text, metadata, distances in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            vector_hits .append({
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "distances": distances,
                "source_path": metadata["source_path"],
                "chapter_id": metadata["chapter_id"],
                "section_id": metadata["section_id"],
                "version_status": metadata["version_status"],
                "citation_eligible": metadata["citation_eligible"],
            })

    if strategy in ("bm25", "hybrid"):
        index_path = settings.DATA_DIR / "bm25_index.pkl"
        index = await asyncio.to_thread(BM25Index.load, index_path)

        bm25_hits = await asyncio.to_thread(
            index.search, query, top_k=candidate_k, where=bm25_where
        )

    if strategy == "vector":
        top_k_chunks = vector_hits
    elif strategy == "bm25":
        top_k_chunks = bm25_hits
    elif strategy == "hybrid":
        top_k_chunks = rrf_fuse(vector_hits, bm25_hits)[:top_k]
    else:
        raise ValueError(f"不支持的检索策略：{strategy}")


    return {
        "success": True,
        "mode": mode,
        "top_k_chunks": top_k_chunks,
    }
