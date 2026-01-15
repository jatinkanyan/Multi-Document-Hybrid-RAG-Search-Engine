import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from schema.models import DocumentChunk

INDEX_PATH = "data/faiss_index"


def _to_langchain_docs(chunks: List[DocumentChunk]) -> List[Document]:
    return [
        Document(
            page_content=chunk.content,
            metadata={
                "source_id": chunk.source_id,
                "source_type": chunk.source_type,
                "title": chunk.title,
                "chunk_index": chunk.chunk_index
            }
        )
        for chunk in chunks
    ]


def index_documents(chunks: List[DocumentChunk]):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docs = _to_langchain_docs(chunks)
    vectorstore = FAISS.from_documents(docs, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(INDEX_PATH)


def load_faiss_index():
    if not os.path.exists(INDEX_PATH):
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
