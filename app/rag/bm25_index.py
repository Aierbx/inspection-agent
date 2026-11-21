import jieba
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.chunks: list[dict] = []
        self.model: BM25Okapi | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in jieba.lcut(text.lower()) if token.strip()]

    def build(self, chunks: list[dict]) -> None:
        corpus = [self._tokenize(chunk["text"]) for chunk in chunks]
        model = BM25Okapi(corpus) if any(corpus) else None

        self.chunks = list(chunks)
        self.model = model

    def search(self, query: str, top_k: int = 5, where: dict | None = None) -> list[dict]:
        if self.model is None or top_k < 1:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.model.get_scores(query_tokens)
        hits = []

        for chunk, score, term_counts in zip(self.chunks, scores, self.model.doc_freqs):
            metadata = chunk["metadata"]

            if where and any(key not in metadata or metadata[key] != value for key, value in where.items()):
                continue

            if not any(token in term_counts for token in query_tokens):
                continue

            hits.append({
                **chunk,
                "source_path": metadata.get("source_path"),
                "chapter_id": metadata.get("chapter_id"),
                "section_id": metadata.get("section_id"),
                "version_status": metadata.get("version_status"),
                "citation_eligible": metadata.get("citation_eligible"),
                "score": float(score),
            })

        hits.sort(key=lambda hit: hit["score"], reverse=True)
        return hits[:top_k]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as file:
            pickle.dump({"chunks": self.chunks, "model": self.model}, file)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        with Path(path).open("rb") as file:
            state = pickle.load(file)

        index = cls()
        index.chunks = state["chunks"]
        index.model = state["model"]
        return index