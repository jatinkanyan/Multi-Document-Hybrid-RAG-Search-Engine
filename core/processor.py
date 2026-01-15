# core/document_processor.py
from pathlib import Path
from typing import List
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """
    Handles document ingestion and chunking.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def ingest_document(self, file_path: str) -> List[Document]:
        """
        Load a text or PDF document, split into chunks, and return as Document objects.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist")

        # Simple PDF/Text loader
        if file_path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(file_path))
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            text = file_path.read_text(encoding="utf-8")

        # Clean text
        text = re.sub(r"\s+", " ", text).strip()

        # Split into chunks
        chunks = self.text_splitter.split_text(text)

        # Convert to Document objects
        documents = [
            Document(
                page_content=chunk,
                metadata={"source": file_path.name, "chunk_index": i}
            )
            for i, chunk in enumerate(chunks)
        ]
        return documents
