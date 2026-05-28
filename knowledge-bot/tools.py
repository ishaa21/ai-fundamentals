from crewai.tools import BaseTool
from pydantic import Field

class RAGSearchTool(BaseTool):
    name: str = "search_knowledge_base"
    description: str = "Search the AI knowledge base to find relevant information about AI concepts, RAG, LLMs, embeddings, and multi-agent systems. Input should be a search query string."
    vectorstore: object = Field(default=None)

    def _run(self, query: str) -> str:
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        docs = retriever.invoke(query)
        results = []
        for i, doc in enumerate(docs):
            results.append(f"Result {i+1}: {doc.page_content}")
        return "\n\n".join(results)