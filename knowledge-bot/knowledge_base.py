from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os

def build_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # If already exists, just load it
    if os.path.exists("./chroma_db"):
        vectorstore = Chroma(
            collection_name="ai_knowledge_base",
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        print(f"Knowledge base loaded: {vectorstore._collection.count()} chunks")
        return vectorstore, embeddings
    
    # Otherwise build fresh
    documents = [
        Document(page_content="Large Language Models (LLMs) are neural networks trained on massive text datasets. They predict the next token based on context. Popular LLMs include GPT-4, Claude, Gemini, and Llama.", metadata={"source": "ai_concepts"}),
        Document(page_content="RAG stands for Retrieval Augmented Generation. It combines a retrieval system with an LLM. The retrieval system fetches relevant documents, which are added to the LLM prompt as context to generate grounded answers.", metadata={"source": "ai_concepts"}),
        Document(page_content="Vector databases store text as numerical vectors called embeddings. They enable semantic search — finding documents by meaning rather than keywords. Popular vector databases include ChromaDB, Pinecone, and FAISS.", metadata={"source": "ai_concepts"}),
        Document(page_content="Prompt engineering is the practice of designing inputs to LLMs to get desired outputs. Key techniques include zero-shot, few-shot, chain-of-thought, and role prompting.", metadata={"source": "ai_concepts"}),
        Document(page_content="Multi-agent systems use multiple AI agents working together. Each agent has a specific role, goal, and set of tools. Frameworks like CrewAI, AutoGen, and Agno enable building multi-agent systems.", metadata={"source": "ai_concepts"}),
        Document(page_content="Hallucination in LLMs refers to generating confident but factually incorrect information. RAG reduces hallucination by grounding responses in retrieved factual documents.", metadata={"source": "ai_concepts"}),
        Document(page_content="Fine-tuning is the process of training a pre-trained LLM on a specific dataset to adapt it for a particular task or domain. LoRA and QLoRA are popular efficient fine-tuning techniques.", metadata={"source": "ai_concepts"}),
        Document(page_content="LangChain is a framework for building LLM applications. It provides abstractions for chains, retrievers, memory, and agents. It simplifies building RAG pipelines and agentic workflows.", metadata={"source": "ai_concepts"}),
        Document(page_content="Embeddings are numerical vector representations of text. Similar texts have similar embeddings. The all-MiniLM-L6-v2 model produces 384-dimensional embeddings for semantic search.", metadata={"source": "ai_concepts"}),
        Document(page_content="CrewAI is a multi-agent orchestration framework. It allows defining agents with roles, goals and tools. Agents collaborate through tasks in a crew to complete complex workflows.", metadata={"source": "ai_concepts"}),
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="ai_knowledge_base",
        persist_directory="./chroma_db"
    )

    print(f"Knowledge base built: {vectorstore._collection.count()} chunks")
    return vectorstore, embeddings