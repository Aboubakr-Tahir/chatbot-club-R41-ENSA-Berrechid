import os, csv
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from config import VECTOR_DIR

def _load_csv_docs():
    # repo root -> data/faq.csv
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(root, "data", "faq.csv")
    docs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q, a = row["question"].strip(), row["answer"].strip()
            docs.append(Document(page_content=f"Q: {q}\nA: {a}", metadata={"source": "faq", "q": q}))
    return docs

def get_vectorstore():
    emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")  # or multilingual-e5-small later
    return Chroma(persist_directory=VECTOR_DIR, embedding_function=emb)

def get_retriever(k: int = 8):
    # Vector retriever (MMR for diversity)
    vec = get_vectorstore().as_retriever(
    search_type="mmr",
    search_kwargs={"k": k, "fetch_k": 24, "lambda_mult": 0.5},
)
    # Keyword retriever (no downloads, robust to typos/short queries)
    bm25 = BM25Retriever.from_documents(_load_csv_docs()); bm25.k = k
    # Ensemble (tune weights if needed)
    return EnsembleRetriever(retrievers=[vec, bm25], weights=[0.6, 0.4])
