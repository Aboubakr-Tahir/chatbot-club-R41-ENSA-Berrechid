import csv, os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.docstore.document import Document

load_dotenv()

VECTOR_DIR = os.getenv("VECTOR_DIR", ".chroma")
DATA_FILE = os.path.join("data", "faq.csv")

def load_docs():
    docs = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"].strip()
            a = row["answer"].strip()
            docs.append(Document(
                page_content=f"Q: {q}\nA: {a}",
                metadata={"source": "faq", "question": q}
            ))
    return docs

def main():
    os.makedirs(VECTOR_DIR, exist_ok=True)
    emb = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    docs = load_docs()
    Chroma.from_documents(docs, emb, persist_directory=VECTOR_DIR)
    print(f"[index_faq] Indexed {len(docs)} FAQs into {VECTOR_DIR}")

if __name__ == "__main__":
    main()
