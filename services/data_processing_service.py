import os
import json
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from utils.settings_loader import settings
from utils.config_loader import load_config



# ---------- CONFIG ----------
REGISTRY_PATH = "processed_docs.json"
PINECONE_INDEX = load_config()["pinecone_db"]["index_name"]
# PINECONE_INDEX = "document-processing"
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY


# ---------- REGISTRY ----------
def load_registry():
    if os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return {}
            return json.loads(content)
    return {}


def save_registry(processed_docs):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, indent=2)


# ---------- FILE LOADING ----------
def load_pdf(file_path: str) -> List[Document]:
    docs = []
    pdf = fitz.open(file_path)
    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text()
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"page_num": page_num}
            ))
    pdf.close()
    return docs


def load_csv(file_path: str) -> List[Document]:
    df = pd.read_csv(file_path)
    docs = []
    for idx, row in df.iterrows():
        docs.append(Document(
            page_content=" ".join(str(v) for v in row.values if pd.notna(v)),
            metadata={"row_index": idx}
        ))
    return docs


# ---------- TEXT SPLITTING ----------
def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(docs)


def add_metadata(chunks: List[Document], doc_name: str):
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update({
            "doc_name": doc_name,
            "chunk_index": idx
        })
    return chunks


# ---------- PROCESSING ----------
def process_files(files: List[str]) -> List[Document]:
    processed_docs = load_registry()
    all_chunks = []

    for file_path in files:
        doc_name = os.path.basename(file_path)

        if doc_name in processed_docs:
            print(f"Skipping already processed file: {doc_name}")
            continue
        
        print(f"Processing file: {doc_name}")
        
        ext = doc_name.split(".")[-1].lower()
        if ext == "pdf":
            docs = load_pdf(file_path)
        elif ext == "csv":
            docs = load_csv(file_path)
        else:
            print(f"Unsupported file type: {doc_name}")
            continue

        chunks = chunk_docs(docs)
        chunks = add_metadata(chunks, doc_name)
        # all_chunks.extend(chunks)
        
        if chunks:
            try:
                store_in_pinecone(chunks, doc_name)
                processed_docs[doc_name] = {
                    "filename": doc_name,
                    "date_added": datetime.now().isoformat()
                }
                save_registry(processed_docs)
            except Exception as e:
                print(f"Error storing {doc_name}: {e}")

    if not all_chunks:
        print("No new documents to process.")
    return all_chunks


# ---------- STORE IN PINECONE ----------
def store_in_pinecone(chunks: List[Document], doc_name:str, index_name = PINECONE_INDEX):
    if not chunks:
        print("No chunks to store.")
        return None

    pc = Pinecone()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        # google_api_key=settings_loader.GOOGLE_API_KEY
    )

    vector_store = PineconeVectorStore.from_texts(
        texts=[chunk.page_content for chunk in chunks],
        embedding=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
        index_name=index_name
    )

    print(f"Stored {len(chunks)} chunks of {doc_name} in Pinecone.")
    return vector_store
