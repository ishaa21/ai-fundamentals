import os
from dotenv import load_dotenv
from crewai import Crew, Process, LLM
from knowledge_base import build_knowledge_base
from agents import create_agents
from tasks import create_tasks

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LITELLM_CACHE"] = "false"
os.environ["CREWAI_DISABLE_CACHE"] = "true"

def run_knowledge_bot(question: str, vectorstore, llm) -> str:
    print(f"\nQuestion: {question}")
    print("="*60)

    researcher, summarizer = create_agents(vectorstore, llm)
    research_task, summarize_task = create_tasks(question, researcher, summarizer)

    crew = Crew(
        agents=[researcher, summarizer],
        tasks=[research_task, summarize_task],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    print("AI Knowledge Bot — Powered by CrewAI + RAG")
    print("="*60)

    # Build once, reuse for all questions
    vectorstore, embeddings = build_knowledge_base()
    llm = LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.3
    )

    questions = [
        "What is RAG and how does it reduce hallucination?",
        "How do multi-agent systems work?",
        "What is the difference between LangChain and CrewAI?"
    ]

    for q in questions:
        answer = run_knowledge_bot(q, vectorstore, llm)
        print(f"\nFINAL ANSWER: {answer}")
        print("="*60)