import os
import streamlit as st
from dotenv import load_dotenv

# -------------------------
# Environment & Config
# -------------------------
load_dotenv()

st.set_page_config(
    page_title="Multi-Document RAG Search Engine",
    page_icon="🔍",
    layout="wide"
)

# -------------------------
# Modular Imports
# -------------------------
from core.loader import load_pdfs
from core.processor import DocumentProcessor
from retrieval.faiss_index import load_faiss_index, index_documents
from retrieval.tavily_search import TavilyRetriever
from retrieval.query_router import classify_query, QueryType
from generation.answer_gen import generate_answer
from generation.summarizer import summarize_top_documents
from schema.models import DocumentChunk

# -------------------------
# UI Header
# -------------------------
st.title("🔍 Multi-Document RAG Search Engine")
st.caption("Hybrid RAG: FAISS (Local) + Tavily (Web) + Groq (LLM)")

# ======================================================
# Sidebar – Document Management
# ======================================================
with st.sidebar:
    st.header("📂 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("📥 Index Documents"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing & Indexing..."):
                # 1. Load PDFs into UnifiedDocument objects
                raw_docs = load_pdfs(uploaded_files)
                
                # 2. Initialize Processor for Chunking
                processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
                all_chunks = []

                for doc in raw_docs:
                    # Split the content of each UnifiedDocument
                    split_texts = processor.text_splitter.split_text(doc.content)
                    
                    for i, text in enumerate(split_texts):
                        all_chunks.append(
                            DocumentChunk(
                                chunk_id=f"{doc.source_id}_{i}",
                                source_id=doc.source_id,
                                source_type=doc.source_type,
                                title=doc.title,
                                content=text,
                                chunk_index=i,
                                metadata=doc.metadata
                            )
                        )

                # 3. Save to FAISS
                index_documents(all_chunks)
                
                # Update Session State
                st.session_state["faiss_index"] = load_faiss_index()
                st.success(f"Successfully indexed {len(all_chunks)} chunks from {len(raw_docs)} files.")

    use_web = st.toggle("🌐 Enable Tavily Web Search", value=True)
    
    st.markdown("---")
    st.subheader("📑 System Status")
    if os.path.exists("data/faiss_index"):
        st.success("FAISS Index: Ready")
    else:
        st.warning("FAISS Index: Not Found")

# ======================================================
# Initialization
# ======================================================
if "faiss_index" not in st.session_state:
    st.session_state["faiss_index"] = load_faiss_index()

faiss_index = st.session_state["faiss_index"]
retriever = faiss_index.as_retriever(search_kwargs={"k": 5}) if faiss_index else None
tavily = TavilyRetriever(max_results=3)

# ======================================================
# Main Chat Logic
# ======================================================
query = st.chat_input("Ask about your documents or the web...")

if query:
    # 1. Query Routing (Classification)
    q_type = classify_query(query)
    
    # UI Visual Indicator for Routing
    route_icons = {QueryType.DOCUMENT: "📄", QueryType.WEB: "🌐", QueryType.HYBRID: "🔀"}
    st.info(f"{route_icons.get(q_type, '🔍')} Routing as: {q_type.value.upper()}")

    with st.spinner("Retrieving and Generating..."):
        # 2. Retrieval Phase
        doc_chunks = []
        if q_type in [QueryType.DOCUMENT, QueryType.HYBRID] and retriever:
            doc_chunks = retriever.invoke(query)

        web_results = []
        # FIX: Force web search if the toggle is ON, ignoring the router's classification
        if use_web:
            web_results = tavily.search(query)
    

        # 3. Answer Generation Phase
        if not doc_chunks and not web_results:
            answer = "I couldn't find any relevant information in the documents or the web."
            sources = []
        else:
            # Note: generate_answer returns (text, List[AnswerSource])
            answer, sources = generate_answer(query, doc_chunks, web_results)

        # 4. Summarization Phase (Requirement GA02)
        summaries = summarize_top_documents(doc_chunks) if doc_chunks else []

    # ======================================================
    # Results Display
    # ======================================================
    tab_ans, tab_evid, tab_sum, tab_web = st.tabs([
        "💬 Answer", "📄 Doc Evidence", "📑 Top Summaries", "🌐 Web Sources"
    ])

    with tab_ans:
        st.markdown(answer)
        if sources:
            with st.expander("View Citations"):
                for src in sources:
                    st.write(f"- **{src.source_type}**: {src.reference}")

    with tab_evid:
        if doc_chunks:
            for d in doc_chunks:
                st.markdown(f"**Source:** {d.metadata.get('title', 'Unknown')}")
                st.caption(d.page_content)
                st.divider()
        else:
            st.info("No local document evidence found for this query.")

    with tab_sum:
        if summaries:
            for s in summaries:
                st.subheader(f"📘 {s['title']}")
                st.write(s['summary'])
                st.caption(f"Based on {s['chunks_used']} semantic chunks.")
        else:
            st.info("Summaries are generated from retrieved local documents.")

    with tab_web:
        if web_results:
            for w in web_results:
                st.markdown(f"**[{w.metadata.get('title')}]({w.metadata.get('url')})**")
                st.write(w.page_content)
                st.divider()
        else:
            st.info("No web results retrieved for this query.")