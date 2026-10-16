
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    return splitter.split_text(text)


def chunk_pages(pages, chunk_size: int = 1000, overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )

    chunks = []
    chunk_id = 1

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "page": page["page"],
                "text": chunk
            })

            chunk_id += 1

    return chunks
    