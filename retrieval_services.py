
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

from services.vector_store_service import vector_store


RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

_reranker = None


def tokenize(text: str):
    return re.findall(r"\w+", text.lower())


def get_reranker():
    global _reranker

    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)

    return _reranker


def document_key(document: Document):
    metadata = document.metadata

    return (
        metadata.get("source", ""),
        metadata.get("page", 0),
        metadata.get("chunk_id", 0),
        document.page_content
    )


def hybrid_search(question: str, candidate_count: int = 20):
    stored = vector_store._collection.get(
        include=["documents", "metadatas"]
    )

    texts = stored["documents"]
    metadatas = stored["metadatas"]

    if not texts:
        return []

    documents = [
        Document(page_content=text, metadata=metadata or {})
        for text, metadata in zip(texts, metadatas)
    ]

    tokenized_corpus = [tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize(question)
    bm25_scores = bm25.get_scores(query_tokens)

    bm25_indices = sorted(
        range(len(documents)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:candidate_count]

    bm25_results = [documents[i] for i in bm25_indices]

    vector_results = vector_store.similarity_search(
        question,
        k=min(candidate_count, len(documents))
    )

    combined = {}
    rrf_k = 60

    for results in [bm25_results, vector_results]:
        for rank, document in enumerate(results, start=1):
            key = document_key(document)

            if key not in combined:
                combined[key] = {
                    "document": document,
                    "rrf_score": 0.0
                }

            combined[key]["rrf_score"] += 1 / (rrf_k + rank)

    candidates = sorted(
        combined.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    candidate_documents = [
        item["document"] for item in candidates
    ]

    reranker = get_reranker()

    pairs = [
        (question, document.page_content)
        for document in candidate_documents
    ]

    scores = reranker.predict(pairs)

    reranked = sorted(
        zip(candidates, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    results = []

    for item, score in reranked:
        document = item["document"]

        results.append({
            "text": document.page_content,
            "metadata": document.metadata,
            "rrf_score": item["rrf_score"],
            "rerank_score": float(score)
        })

    return results
