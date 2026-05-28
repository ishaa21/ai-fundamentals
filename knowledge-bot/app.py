from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from crewai import LLM
from knowledge_base import build_knowledge_base
from agents import create_agents
from tasks import create_tasks
from crewai import Crew, Process

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LITELLM_CACHE"] = "false"
os.environ["CREWAI_DISABLE_CACHE"] = "true"

app = FastAPI(title="AI Knowledge Bot", description="Multi-Agent RAG System")

# Load once at startup
vectorstore, embeddings = build_knowledge_base()
llm = LLM(model="groq/llama-3.3-70b-versatile", temperature=0.3)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Knowledge Bot</title>
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { 
                font-family: -apple-system, sans-serif; 
                background: #0f0f0f; 
                color: #f0f0f0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .container {
                width: 100%;
                max-width: 700px;
                padding: 2rem;
            }
            h1 { 
                font-size: 1.8rem; 
                margin-bottom: 0.5rem;
                color: #fff;
            }
            .subtitle {
                color: #888;
                font-size: 0.9rem;
                margin-bottom: 2rem;
            }
            .badge {
                display: inline-block;
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 20px;
                padding: 4px 12px;
                font-size: 0.75rem;
                color: #aaa;
                margin-right: 6px;
                margin-bottom: 1.5rem;
            }
            textarea {
                width: 100%;
                padding: 1rem;
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
                color: #f0f0f0;
                font-size: 1rem;
                resize: vertical;
                min-height: 80px;
                outline: none;
            }
            textarea:focus { border-color: #555; }
            button {
                margin-top: 1rem;
                width: 100%;
                padding: 0.9rem;
                background: #fff;
                color: #000;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 500;
                cursor: pointer;
            }
            button:hover { background: #e0e0e0; }
            button:disabled { background: #333; color: #666; cursor: not-allowed; }
            .answer-box {
                margin-top: 1.5rem;
                padding: 1.2rem;
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 8px;
                line-height: 1.7;
                display: none;
            }
            .answer-label {
                font-size: 0.75rem;
                color: #888;
                margin-bottom: 0.5rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .examples {
                margin-top: 1.5rem;
            }
            .examples p {
                font-size: 0.8rem;
                color: #666;
                margin-bottom: 0.5rem;
            }
            .example-btn {
                display: inline-block;
                margin: 4px 4px 4px 0;
                padding: 6px 12px;
                background: #1a1a1a;
                border: 1px solid #333;
                border-radius: 20px;
                font-size: 0.8rem;
                color: #aaa;
                cursor: pointer;
                width: auto;
                font-weight: normal;
            }
            .example-btn:hover { 
                background: #222; 
                color: #fff;
            }
            .loading { color: #888; font-style: italic; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Knowledge Bot</h1>
            <p class="subtitle">Multi-Agent RAG System — answers grounded in verified knowledge</p>
            <div>
                <span class="badge">CrewAI</span>
                <span class="badge">RAG</span>
                <span class="badge">Groq LLM</span>
                <span class="badge">ChromaDB</span>
            </div>

            <textarea id="question" placeholder="Ask anything about AI concepts..."></textarea>
            <button id="askBtn" onclick="askQuestion()">Ask →</button>

            <div class="answer-box" id="answerBox">
                <div class="answer-label">Answer</div>
                <div id="answerText"></div>
            </div>

            <div class="examples">
                <p>Try these:</p>
                <button class="example-btn" onclick="setQuestion('What is RAG and how does it reduce hallucination?')">What is RAG?</button>
                <button class="example-btn" onclick="setQuestion('How do multi-agent systems work?')">Multi-agent systems?</button>
                <button class="example-btn" onclick="setQuestion('What is the difference between LangChain and CrewAI?')">LangChain vs CrewAI?</button>
                <button class="example-btn" onclick="setQuestion('What are embeddings and how do vector databases work?')">Embeddings?</button>
                <button class="example-btn" onclick="setQuestion('What is prompt engineering?')">Prompt engineering?</button>
            </div>
        </div>

        <script>
            function setQuestion(q) {
                document.getElementById('question').value = q;
            }

            async function askQuestion() {
                const question = document.getElementById('question').value.trim();
                if (!question) return;

                const btn = document.getElementById('askBtn');
                const answerBox = document.getElementById('answerBox');
                const answerText = document.getElementById('answerText');

                btn.disabled = true;
                btn.textContent = 'Agents working...';
                answerBox.style.display = 'block';
                answerText.innerHTML = '<span class="loading">Researcher agent searching knowledge base...</span>';

                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });
                    const data = await response.json();
                    answerText.textContent = data.answer;
                } catch (error) {
                    answerText.textContent = 'Error getting answer. Please try again.';
                } finally {
                    btn.disabled = false;
                    btn.textContent = 'Ask →';
                }
            }

            document.getElementById('question').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    researcher, summarizer = create_agents(vectorstore, llm)
    research_task, summarize_task = create_tasks(
        request.question, researcher, summarizer
    )
    crew = Crew(
        agents=[researcher, summarizer],
        tasks=[research_task, summarize_task],
        process=Process.sequential,
        verbose=False
    )
    result = crew.kickoff()
    return AnswerResponse(
        question=request.question,
        answer=str(result)
    )

@app.get("/health")
def health():
    return {"status": "ok", "model": "groq/llama-3.3-70b-versatile"}