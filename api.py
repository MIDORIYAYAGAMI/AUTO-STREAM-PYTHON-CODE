from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.comprehensive_agent_fixed import build_correct_graph

app = FastAPI()

# ---------------- UI ----------------
app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

# ---------------- Agent ----------------
graph = build_correct_graph()
sessions = {}


# ---------------- Models ----------------
class ChatRequest(BaseModel):
    session_id: str
    message: str


# ---------------- Routes ----------------
@app.get("/api")
def home():
    return {
        "message": "AutoStream AI Agent API is running",
        "endpoints": {
            "health": "/api/health",
            "chat": "/api/chat",
            "ui": "/ui",
            "docs": "/docs"
        }
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
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

    result = graph.invoke(state)
    sessions[req.session_id] = result

    return {"response": result["response"]}
