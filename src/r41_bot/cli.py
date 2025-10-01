# src/r41_bot/cli.py
import sys
from retriever import get_retriever
from chains import build_rag_chain
from faq_fastpath import try_fastpath

def main():
    retriever = get_retriever(k=6)
    chain = build_rag_chain(retriever)

    def answer(q):
        # 1) fuzzy direct hit (no LLM cost)
        if len(q.split()) < 10:
            a = try_fastpath(q, threshold=85)
            if a:
                return a
        # 2) RAG; if the retriever returned no context, chain will see empty context.
        ans = chain.invoke(q)
        # small post-guard: if LLM says it doesn’t know but we *do* have a close FAQ, relax threshold
        if "don’t have this info" in ans.lower():
            a2 = try_fastpath(q, threshold=70)
            if a2: return a2
        return ans

    if len(sys.argv) > 1:
        print(answer(" ".join(sys.argv[1:])))
        return

    print("R41 ENSAB Chatbot (CLI). Type your question, or 'exit' to quit.")
    while True:
        q = input("> ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue
        try:
            print(answer(q))
        except Exception as e:
            print(f"[error] {e}")

if __name__ == "__main__":
    main()
