from collections import defaultdict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are an AI assistant summarizing a document for evidence display.

RULES:
1. Use ONLY the provided content
2. Do NOT add external knowledge
3. Produce a concise, factual summary
4. Do NOT hallucinate

--------------------
DOCUMENT TITLE:
{title}

--------------------
CONTENT:
{content}

--------------------
SUMMARY:
""")

def summarize_top_documents(document_chunks, top_n=3):
    """
    Groups chunks by document title and summarizes top-N documents
    """

    # -----------------------------
    # Group chunks by document title
    # -----------------------------
    grouped_docs = defaultdict(list)

    for doc in document_chunks:
        title = doc.metadata.get("title", "Unknown Document")
        grouped_docs[title].append(doc.page_content)

    # -----------------------------
    # Rank documents by chunk count
    # -----------------------------
    ranked_docs = sorted(
        grouped_docs.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:top_n]

    summaries = []

    # -----------------------------
    # Summarize each document
    # -----------------------------
    for title, chunks in ranked_docs:
        combined_text = "\n\n".join(chunks)

        prompt = SUMMARY_PROMPT.format(
            title=title,
            content=combined_text
        )

        summary = llm.invoke(prompt).content

        summaries.append({
            "title": title,
            "summary": summary,
            "chunks_used": len(chunks)
        })

    return summaries
