import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()


class GroqLLM:
    """
    Wrapper around Groq Chat LLM for RAG-based question answering
    """

    def __init__(self):
        self.model_name = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", 0))

        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name=self.model_name,
            temperature=self.temperature
        )

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using retrieved context

        Args:
            question (str): User query
            context (str): Retrieved context (FAISS + Web)

        Returns:
            str: LLM response
        """

        system_prompt = (
            "You are an intelligent AI assistant. "
            "Answer the user's question ONLY using the provided context. "
            "If the answer is not present in the context, say "
            "'I could not find this information in the provided documents.'"
        )

        user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = self.llm(messages)
        return response.content
