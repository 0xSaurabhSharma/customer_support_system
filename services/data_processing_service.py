# data_processing_services.py

import os
import json
import fitz
import pandas as pd
from datetime import datetime
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from utils.settings_loader import settings
from utils.config_loader import load_config

# --- CONFIGURATION ---
REGISTRY_PATH = "processed_docs.json"
PINECONE_INDEX = load_config()["pinecone_db"]["index_name"]
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# --- REGISTRY FUNCTIONS ---
def load_registry() -> dict:
    """Loads the registry of processed documents."""
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            return json.loads(content) if content else {}
    return {}

def save_registry(processed_docs: dict):
    """Saves the registry of processed documents."""
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2)


# --- DOCUMENT LOADING FUNCTIONS ---
def load_pdf(file_path: str) -> List[Document]:
    """Loads text from a PDF file into a list of Documents."""
    docs = []
    with fitz.open(file_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text()
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={"page_num": page_num, "source": file_path}
                ))
    return docs

def load_csv(file_path: str) -> List[Document]:
    """Loads text from a CSV file into a list of Documents."""
    df = pd.read_csv(file_path)
    docs = []
    for idx, row in df.iterrows():
        docs.append(Document(
            page_content=" ".join(str(v) for v in row.values if pd.notna(v)),
            metadata={"row_index": idx, "source": file_path}
        ))
    return docs


# --- TEXT SPLITTING ---
def chunk_and_add_metadata(docs: List[Document], doc_name: str) -> List[Document]:
    """Chunks documents and adds metadata to each chunk."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_name": doc_name,
            "chunk_index": idx,
        })
    return chunks


# --- PINECONE INTEGRATION ---
def store_in_pinecone(chunks: List[Document], index_name: str) -> Optional[PineconeVectorStore]:
    """
    Stores document chunks in a Pinecone index.
    Initializes PineconeVectorStore using the from_documents method.
    """
    if not chunks:
        print("No chunks to store.")
        return None

    try:
        # from_documents is the preferred method for this task
        vector_store = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=EMBEDDING_MODEL,
            index_name=index_name,
        )
        print(f"Stored {len(chunks)} chunks in Pinecone index '{index_name}'.")
        return vector_store
    except Exception as e:
        print(f"Error storing chunks in Pinecone: {e}")
        return None


# --- MAIN PROCESSING FUNCTION ---
def process_files(files: List[str]) -> None:
    """
    Orchestrates the loading, chunking, and storage of new documents.
    """
    processed_docs = load_registry()
    
    for file_path in files:
        doc_name = os.path.basename(file_path)

        if doc_name in processed_docs:
            print(f"Skipping already processed file: {doc_name}")
            continue

        print(f"Processing file: {doc_name}")
        ext = doc_name.split(".")[-1].lower()
        
        docs = []
        if ext == "pdf":
            docs = load_pdf(file_path)
        elif ext == "csv":
            docs = load_csv(file_path)
        else:
            print(f"Unsupported file type: {doc_name}")
            continue

        chunks = chunk_and_add_metadata(docs, doc_name)
        
        if chunks:
            vector_store = store_in_pinecone(chunks, PINECONE_INDEX)
            if vector_store:
                processed_docs[doc_name] = {
                    "filename": doc_name,
                    "date_added": datetime.now().isoformat()
                }
                save_registry(processed_docs)