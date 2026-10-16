from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    collection_name="ai_document_assistant",
    embedding_function=embeddings,
    persist_directory=str(CHROMA_DIR)
)


def store_chunks(chunks, source_filename: str):
    documents = []
    ids = []

    for chunk in chunks:
        document = Document(
            page_content=chunk["text"],
            metadata={
                "source": source_filename,
                "page": int(chunk["page"]),
                "chunk_id": int(chunk["chunk_id"])
            }
        )

        documents.append(document)

        ids.append(
            f"{source_filename}:{chunk['page']}:{chunk['chunk_id']}"
        )

    if not documents:
        return 0

    vector_store.delete(
        where={"source": source_filename}
    )

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    return len(documents)


def retrieve_documents(question: str, k: int = 3):
    results = vector_store.similarity_search_with_score(
        question,
        k=k
    )

    documents = []

    for document, distance in results:
        documents.append({
            "text": document.page_content,
            "distance": float(distance),
            "metadata": document.metadata
        })

    return documents
