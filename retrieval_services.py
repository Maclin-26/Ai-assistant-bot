from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(collection_name="ai_document_assitant",embedding_function=embeddings,persist_directory="./chroma_db")

def retrieve_data(question : str,k : int = 3 ):
    results = vector_store.similarity_search_with_score(question,k=k)

    documents = []

    for document,distance in results:
        documents.append(
            {
                "document":document.page_content,
                "distance":float(distance),
                "metadata":document.metadata
            }
        )
    return documents