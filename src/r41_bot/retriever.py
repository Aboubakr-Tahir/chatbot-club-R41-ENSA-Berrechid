import csv
from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.document_loaders import DirectoryLoader

from .config import VECTOR_DIR, EMBEDDING_MODEL

KNOWLEDGE_BASE_DIR = "data/knowledge_base"

def _load_markdown_docs():
    """Loads all markdown documents from the knowledge base directory."""
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.md")
    documents = loader.load()
    return documents

def get_retriever(k: int):
    """
    Builds an ensemble retriever combining semantic and keyword search.
    """
    # Initialize the embedding model for the vector store
    embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # Initialize the Chroma vector store retriever
    vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embedding)
    vs_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Load documents for the keyword retriever
    markdown_docs = _load_markdown_docs()

    # Initialize the BM25 keyword retriever
    bm25_retriever = BM25Retriever.from_documents(markdown_docs)
    bm25_retriever.k = k

    # Initialize the ensemble retriever
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vs_retriever, bm25_retriever], weights=[0.5, 0.5]
    )

    return ensemble_retriever