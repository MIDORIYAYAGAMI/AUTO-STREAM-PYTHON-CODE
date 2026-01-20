from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings


# -------------------- STATE --------------------
class AgentState(TypedDict):
    user_message: str
    intent: str
    retrieved_context: str
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    response: str


# -------------------- KNOWLEDGE --------------------
RAW_KNOWLEDGE = [
    "Starter Plan costs $9 per month and includes 100 hours of video processing, basic analytics, email support, and HD export.",
    "Pro Plan costs $29 per month and includes 500 hours of processing, advanced analytics, priority support, custom branding, and 4K export.",
    "Enterprise Plan offers unlimited processing, white-label features, API access, and dedicated support with custom pricing.",
    "AutoStream offers a 14-day free trial with full access to Pro features and no credit card required.",
    "AutoStream supports YouTube, Instagram, TikTok, and LinkedIn.",
    "AutoStream is an AI-powered video editing and optimization platform for creators and businesses."
]


# -------------------- EMBEDDINGS + VECTORSTORE --------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

documents = [Document(page_content=text) for text in RAW_KNOWLEDGE]
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# -------------------- LLM (ONLY ONE) --------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)


# -------------------- INTENT DETECTION --------------------
def detect_intent(message: str) -> str:
    msg = message.lower()

    if any(w in msg for w in ["hi", "hello", "hey"]):
        return "greeting"
    if any(w in msg for w in ["price", "pricing", "cost", "plan"]):
        return "pricing"
    if any(w in msg for w in ["feature", "support", "platform"]):
        return "features"
    if any(w in msg for w in ["trial", "free"]):
        return "trial"
    if any(w in msg for w in ["sign up", "signup", "subscribe", "start", "ready", "want"]):
        return "high_intent"

    return "general"


# -------------------- GRAPH NODES --------------------
def intent_node(state: AgentState):
    state["intent"] = detect_intent(state["user_message"])
    return state


def rag_node(state: AgentState):
    docs = retriever.invoke(state["user_message"])
    state["retrieved_context"] = "\n".join(d.page_content for d in docs)
    return state


def response_node(state: AgentState):
    prompt = f"""
You are AutoStream AI assistant.

User intent: {state["intent"]}
User message: {state["user_message"]}

Relevant information:
{state.get("retrieved_context", "")}

Respond clearly and helpfully.
"""
    state["response"] = llm.invoke(prompt).content
    return state


# -------------------- BUILD GRAPH --------------------
def build_correct_graph():
    graph = StateGraph(AgentState)

    graph.add_node("intent", intent_node)
    graph.add_node("rag", rag_node)
    graph.add_node("response", response_node)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "rag")
    graph.add_edge("rag", "response")
    graph.add_edge("response", END)

    return graph.compile()
