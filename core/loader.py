import os
import tempfile
import uuid
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from schema.models import UnifiedDocument


def load_pdfs(uploaded_files) -> List[UnifiedDocument]:
    documents = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        full_text = "\n".join(p.page_content for p in pages)

        documents.append(
            UnifiedDocument(
                source_id=str(uuid.uuid4()),
                source_type="pdf",
                title=file.name,
                content=full_text,
                metadata={"filename": file.name}
            )
        )

        os.remove(tmp_path)

    return documents
