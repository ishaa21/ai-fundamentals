from crewai import Agent
from tools import RAGSearchTool

def create_agents(vectorstore, llm):
    rag_tool = RAGSearchTool(vectorstore=vectorstore)

    researcher = Agent(
        role="AI Knowledge Researcher",
        goal="Search the knowledge base to find accurate information that answers the user's question.",
        backstory="""You are an expert AI researcher. You always search 
        the knowledge base before answering and never make up information. 
        Be concise and factual.""",
        tools=[rag_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3
    )

    summarizer = Agent(
        role="AI Knowledge Summarizer",
        goal="Produce a clear, concise answer from the researcher's findings.",
        backstory="""You are a technical writer who explains AI concepts 
        clearly. Use ONLY information from research findings. 
        Keep answers under 5 sentences.""",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2
    )

    return researcher, summarizer