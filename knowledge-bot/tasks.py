from crewai import Task

def create_tasks(question, researcher, summarizer):
    research_task = Task(
        description=f"""Search the knowledge base to find comprehensive 
        information about: {question}
        Use the search_knowledge_base tool.
        Present findings clearly with retrieved evidence.
        Be concise — do not repeat yourself.""",
        expected_output="A concise research report with relevant facts from the knowledge base.",
        agent=researcher
    )

    summarize_task = Task(
        description=f"""Create a clear, concise answer to: {question}
        Use ONLY the research findings provided.
        Maximum 5 sentences. No repetition.""",
        expected_output="A clear 3-5 sentence answer grounded in the research findings.",
        agent=summarizer,
        context=[research_task]
    )

    return research_task, summarize_task