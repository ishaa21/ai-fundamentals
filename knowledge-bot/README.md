# AI Knowledge Bot — Multi-Agent RAG System

> A multi-agent AI system that answers questions by retrieving 
> relevant information from a knowledge base — powered by 
> CrewAI, RAG, and Groq LLM. Zero hallucination through 
> grounded, evidence-based responses.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-orange?style=flat-square)](https://crewai.com)
[![Groq](https://img.shields.io/badge/Groq-LLM-green?style=flat-square)](https://groq.com)

---

## What is this?

Most AI chatbots answer from training data alone — which causes 
hallucination. This bot uses **RAG (Retrieval Augmented Generation)** 
combined with a **multi-agent architecture** to answer questions 
grounded in a verified knowledge base.

One question → two agents work together:
- **Researcher Agent** searches the knowledge base using RAG
- **Summarizer Agent** produces a clean, concise answer from findings

---

## Architecture
User Question
│
▼
Researcher Agent (Groq LLM)
│
▼ uses tool
RAG Search Tool → ChromaDB (vector search)
│
▼ retrieves chunks
Researcher compiles findings
│
▼ passes via context
Summarizer Agent (Groq LLM)
│
▼
Final Grounded Answer

---

## Example

**Question:** What is RAG and how does it reduce hallucination?

**Researcher retrieves:**
- "RAG stands for Retrieval Augmented Generation..."
- "Hallucination in LLMs refers to generating confident 
   but factually incorrect information. RAG reduces 
   hallucination by grounding responses in retrieved 
   factual documents."

**Summarizer final answer:**
> RAG combines a retrieval system with an LLM, where the 
> retrieval system fetches relevant documents added to the 
> LLM prompt as context. This reduces hallucination by 
> grounding responses in retrieved factual documents rather 
> than the model's training data alone.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Multi-Agent Framework | CrewAI |
| LLM | Groq (Llama 3.3 70B) |
| RAG Pipeline | LangChain + ChromaDB |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (persistent) |

---

## Project Structure
knowledge-bot/
├── main.py              # Entry point
├── knowledge_base.py    # Builds and loads ChromaDB
├── agents.py            # Researcher + Summarizer agents
├── tasks.py             # Task definitions
├── tools.py             # RAG search tool
├── .env                 # API keys (not committed)
└── chroma_db/           # Persistent vector store

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/ishaa21/ai-fundamentals.git
cd ai-fundamentals/knowledge-bot

# 2. Install dependencies
pip install crewai langchain-huggingface langchain-chroma \
    langchain-text-splitters groq python-dotenv litellm pyvis

# 3. Set your Groq API key
# Get free key at: https://console.groq.com
echo "GROQ_API_KEY=your_key_here" > .env

# 4. Run the bot
python main.py
```

---

## Key Features

- **Zero hallucination** — answers grounded in knowledge base only
- **Multi-agent collaboration** — Researcher and Summarizer 
  work sequentially
- **Persistent vector store** — ChromaDB saves embeddings 
  to disk, no re-embedding on every run
- **Extensible** — add your own documents to `knowledge_base.py`
- **Production-ready structure** — modular files, not a notebook

---

## What I Learned Building This

- RAG from scratch vs LangChain abstractions
- How agents use tools autonomously in CrewAI
- How `context=[task]` passes output between agents
- ChromaDB persistence to avoid re-embedding
- Prompt engineering for agent backstories and task descriptions

---

## Author

Built by [Isha Zalavadia](https://github.com/ishaa21) as part of 
an AI/ML learning journey — from raw API calls to multi-agent systems.