from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.comprehensive_agent_fixed import build_correct_graph

app = FastAPI()

# ✅ Mount UI immediately after app creation
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")


@app.get("/")
def home():
    return {
        "message": "AutoStream AI Agent API is running",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "ui": "/ui",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    message: str


sessions = {}


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    if req.session_id not in sessions:
        sessions[req.session_id] = {
            "user_message": "",
            "intent": "",
            "retrieved_context": "",
            "name": None,
            "email": None,
            "platform": None,
            "response": ""
        }

    state = sessions[req.session_id]
    state["user_message"] = req.message

    graph = build_correct_graph()
    result = graph.invoke(state)

    sessions[req.session_id] = result
    return {"state": result}

