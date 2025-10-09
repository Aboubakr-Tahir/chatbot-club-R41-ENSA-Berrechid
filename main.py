import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware # Import the middleware
from pydantic import BaseModel
from typing import List

# Import your chatbot logic
from src.r41_bot.chains import build_rag_chain, build_router_chain, build_query_rewriter_chain
from src.r41_bot.retriever import get_retriever

# --- 1. Initialize Chatbot Components ---
# This is done once when the server starts
retriever = get_retriever(k=6)
rag_chain = build_rag_chain(retriever)
router_chain = build_router_chain()
rewriter_chain = build_query_rewriter_chain()

# --- 2. Define API Data Models ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    chat_history: List[ChatMessage]

# --- 3. Create FastAPI App ---
app = FastAPI()

# --- ADD THIS SECTION FOR CORS ---
# This allows your React frontend (running on port 5173) to make requests to this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # The address of our React app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------

# --- 4. Define the Streaming Chat Endpoint ---
async def chat_stream_generator(request: ChatRequest):
    """
    This generator function handles the core logic of rewriting, routing,
    and invoking the RAG chain, yielding tokens as they are generated.
    """
    q = request.question
    # Pydantic v2+ requires .model_dump() instead of .dict()
    chat_history = [msg.model_dump() for msg in request.chat_history]

    # --- THIS IS THE UPDATED LOGIC ---
    # Rewrite the question, now WITH history
    rewritten_q = rewriter_chain.invoke({"question": q, "chat_history": chat_history})
    
    # Route the question, also WITH history
    route_decision = router_chain.invoke({"question": rewritten_q, "chat_history": chat_history})
    route = route_decision.get("route")
    # ---------------------------------

    if route == "irrelevant":
        yield "I can only answer questions about the R41 club. How can I help you with that?"
        return

    if route == "vector_search":
        # Use .astream() for asynchronous streaming
        # The RAG chain already correctly receives the history
        async for chunk in rag_chain.astream({"question": rewritten_q, "chat_history": chat_history}):
            yield chunk

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(chat_stream_generator(request), media_type="text/event-stream")

# The old static file serving logic has been removed, as the Vite server now handles the frontend.