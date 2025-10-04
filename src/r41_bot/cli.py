import sys
from retriever import get_retriever
from chains import build_rag_chain, build_router_chain

def main():
    retriever = get_retriever(k=6)
    rag_chain = build_rag_chain(retriever)
    router_chain = build_router_chain()

    def answer(q):
        # First, use the router to classify the question
        route_decision = router_chain.invoke({"question": q})
        route = route_decision.get("route")

        # Path 1: The question is irrelevant
        if route == "irrelevant":
            return "I can only answer questions about the R41 club. How can I help you with that?"

        # Path 2: The question is relevant and requires vector search
        if route == "vector_search":
            print("[Routing to: Vector Search (RAG)]")
            return rag_chain.invoke(q)
        
        # Default fallback if router fails for some reason
        return "I'm not sure how to handle that question. Please try rephrasing."


    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:])))
        return

    print("R41 ENSAB Chatbot (CLI). Type your question, or 'exit' to quit.")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        print(answer(q))


if __name__ == "__main__":
    main()