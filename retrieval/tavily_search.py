import os
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()


class TavilyRetriever:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        # Use TavilySearchResults class
        self.client = TavilySearchResults(max_results=max_results)

    def search(self, query: str) -> List[Document]:
        try:
            # invoke() returns a list of dictionaries
            raw_results = self.client.invoke({"query": query})
            
            documents = []
            for item in raw_results:
                documents.append(
                    Document(
                        page_content=item.get("content", ""),
                        metadata={
                            "title": item.get("title", "Web Result"),
                            "url": item.get("url"),
                            "source": "tavily_web",
                        },
                    )
                )
            return documents
        except Exception as e:
            st.error(f"Tavily Search Error: {e}")
            return []
