import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

load_dotenv()

VECTOR_DIR = ".chroma"
KNOWLEDGE_BASE_DIR = "data/knowledge_base"


def main():
    """
    Main function to ingest data from the knowledge base directory
    into the Chroma vector store.
    """
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Error: Knowledge base directory not found at '{KNOWLEDGE_BASE_DIR}'")
        return

    # Load all .md files from the knowledge base directory
    print(f"Loading documents from '{KNOWLEDGE_BASE_DIR}'...")
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.md", show_progress=True)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    print(f"Split documents into {len(splits)} chunks.")

    # Initialize the embedding model
    print("Initializing embedding model...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # Delete the old vector store if it exists
    if os.path.exists(VECTOR_DIR):
        print(f"Deleting old vector store at '{VECTOR_DIR}'...")
        shutil.rmtree(VECTOR_DIR)

    # Create and persist the new vector store
    print(f"Creating and persisting vector store at '{VECTOR_DIR}'...")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=VECTOR_DIR
    )

    print("\nIngestion complete!")
    print(f"Vector store created with {len(splits)} chunks from the knowledge base.")


if __name__ == "__main__":
    main()