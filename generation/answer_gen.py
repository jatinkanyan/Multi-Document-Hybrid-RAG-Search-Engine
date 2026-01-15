from typing import List, Tuple
from schema.models import AnswerSource
from langchain_groq import ChatGroq
from langchain_core.documents import Document

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)


def generate_answer(
    query: str,
    document_chunks: List[Document],
    web_docs: List[Document]
) -> Tuple[str, List[AnswerSource]]:


    context = []
    sources = []

    for doc in document_chunks:
        context.append(doc.page_content)
        sources.append(
            AnswerSource(
                source_type="Doc",
                reference=f"{doc.metadata['title']} – Chunk {doc.metadata['chunk_index']}"
            )
        )

    for doc in web_docs:
        context.append(doc.page_content)
        sources.append(
            AnswerSource(
                source_type="Web",
                reference=doc.metadata.get("url", "Tavily")
            )
        )

    if not context:
        return "No relevant information found.", []

    prompt = f"""
Answer the question strictly using the context below.

Context:
{chr(10).join(context)}

Question:
{query}
"""

    response = llm.invoke(prompt)
    return response.content, sources
