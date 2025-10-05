import sys
from langchain.memory import ConversationBufferMemory
from retriever import get_retriever
from chains import build_rag_chain, build_router_chain, build_query_rewriter_chain

def main():
    retriever = get_retriever(k=6)
    rag_chain = build_rag_chain(retriever)
    router_chain = build_router_chain()
    rewriter_chain = build_query_rewriter_chain()

    # 1. Initialize conversation memory
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

    def answer(q, chat_history):
        # First, rewrite the question to add temporal context
        rewritten_q = rewriter_chain.invoke({"question": q})
        print(f"[Rewritten Question: {rewritten_q}]") # For debugging

        # Then, use the router on the rewritten question
        route_decision = router_chain.invoke({"question": rewritten_q})
        route = route_decision.get("route")

        # Path 1: The question is irrelevant
        if route == "irrelevant":
            return "Hi!, I can only answer questions about the R41 club. How can I help you with that?"

        # Path 2: The question is relevant and requires vector search
        if route == "vector_search":
            print("[Routing to: Vector Search (RAG)]")
            # Invoke the RAG chain with the REWRITTEN question AND the history
            return rag_chain.invoke({"question": rewritten_q, "chat_history": chat_history})
        
        # Default fallback if router fails for some reason
        return "I'm not sure how to handle that question. Please try rephrasing."


    # This part for single-shot questions will remain memoryless
    if len(sys.argv) > 1:
        # Pass an empty history for single-shot questions
        print(answer(" ".join(sys.argv[1:]), []))
        return

    print("R41 ENSAB Chatbot (CLI). Type your question, or 'exit' to quit.")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        
        # 2. Load the history from the memory object
        chat_history = memory.load_memory_variables({}).get("chat_history", [])

        # 3. Get the answer from the chain
        ans = answer(q, chat_history)

        # 4. Save the original question and answer to memory for the next turn
        memory.save_context({"question": q}, {"answer": ans})
        
        print(ans)


if __name__ == "__main__":
    main()