import asyncio

from app.rag.ingest import create_embeddings, get_or_create_collection


async def search_knowledge_base(query: str, mode: str, top_k: int = 5):
    top_k = min(top_k, 5)

    query_embeddings = await asyncio.to_thread(
        create_embeddings,
        [query]
    )
    query_embedding = query_embeddings[0]

    if mode == "current":
        where = {
            "citation_eligible": True,
        }
    else:
        where = None

    collection = await asyncio.to_thread(
        get_or_create_collection
    )

    result = await asyncio.to_thread(
        collection.query,
        query_embeddings = [query_embedding],
        n_results = top_k,
        where = where,
        include = ["documents", "metadatas", "distances"]
    )

    top_k_chunks = []
    for chunk_id, text, metadata, distances in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        top_k_chunks.append({
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

    return {
        "success": True,
        "mode": mode,
        "top_k_chunks": top_k_chunks,
    }