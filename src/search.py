import os
from collections import OrderedDict
from typing import Dict, List

from dotenv import load_dotenv

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:
    def __init__(
        self,
        data_dir: str = "data",
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.1-8b-instant",
    ):
        from langchain_groq import ChatGroq

        self.data_dir = data_dir
        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
        )

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Put it in your .env file or environment variables."
            )

        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=llm_model,
        )
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def refresh_index(self) -> None:
        docs = load_all_documents(self.data_dir)
        if not docs:
            raise ValueError("No supported documents found in the data folder.")
        self.vectorstore.build_from_documents(docs)

    def ensure_index(self) -> None:
        if self.vectorstore.exists():
            self.vectorstore.load()
        else:
            self.refresh_index()

    @staticmethod
    def _format_context(results: List[Dict]) -> str:
        context_blocks = []

        for i, result in enumerate(results, start=1):
            meta = result.get("metadata") or {}
            file_name = meta.get("file_name", "Unknown")
            page = meta.get("page")
            page_label = f", page {page + 1}" if isinstance(page, int) else ""
            text = meta.get("text", "")

            context_blocks.append(f"[{i}] Source: {file_name}{page_label}\n{text}")

        return "\n\n".join(context_blocks)

    @staticmethod
    def _build_citations(results: List[Dict]) -> List[Dict]:
        citations = []

        for i, result in enumerate(results, start=1):
            meta = result.get("metadata") or {}
            citations.append(
                {
                    "id": i,
                    "file_name": meta.get("file_name", "Unknown"),
                    "source": meta.get("source", "Unknown"),
                    "page": (meta.get("page") + 1) if isinstance(meta.get("page"), int) else None,
                    "chunk_id": meta.get("chunk_id"),
                    "excerpt": meta.get("text", "")[:300].strip(),
                }
            )

        return citations

    @staticmethod
    def _unique_documents(results: List[Dict]) -> List[str]:
        docs = OrderedDict()

        for result in results:
            meta = result.get("metadata") or {}
            docs[meta.get("file_name", "Unknown")] = True

        return list(docs.keys())

    def answer_query(self, query: str, top_k: int = 5) -> Dict:
        self.ensure_index()
        results = self.vectorstore.query(query, top_k=top_k)

        if not results:
            return {
                "query": query,
                "summary": "No relevant documents found.",
                "citations": [],
                "documents": [],
                "raw_results": [],
            }

        context = self._format_context(results)

        prompt = f"""
You are a retrieval assistant.
Answer the user query using only the provided context.
Return:
1. A short summary
2. Key points in bullet form
3. Inline citations like [1], [2]
Do not invent facts outside the context.

User query: {query}

Context:
{context}
""".strip()

        response = self.llm.invoke(prompt)

        return {
            "query": query,
            "summary": response.content,
            "citations": self._build_citations(results),
            "documents": self._unique_documents(results),
            "raw_results": results,
        }


if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is LangServe for API Deployment?"
    result = rag_search.answer_query(query, top_k=3)
    print("Summary:\n", result["summary"])
    print("\nDocuments used:", result["documents"])
    print("\nCitations:")
    for citation in result["citations"]:
        print(citation)