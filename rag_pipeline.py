# test_rag_pipeline.py

import os
import fitz
import json
from pathlib import Path

# Assuming your services are in a 'services' directory
from services.data_processing_service import process_files, setup_parent_document_retriever
from services.rag_retriever_service import (
    initialize_pinecone_vectorstore,
    create_rag_retriever,
)
from utils.config_loader import load_config
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --- CONFIGURATION ---
DATA_DIR = "data"
REGISTRY_PATH = "processed_docs.json"
PINECONE_INDEX_NAME = load_config()["pinecone_db"]["index_name"]
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def create_dummy_files():
    """Create dummy PDF files for testing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Dummy GST file
    gst_text = "This document outlines the Goods and Services Tax (GST) regulations in India. Key provisions include tax rates and input tax credit rules."
    gst_pdf = fitz.open()
    page = gst_pdf.new_page()
    page.insert_text((72, 72), gst_text)
    gst_pdf.save(os.path.join(DATA_DIR, "gst_rules_2024.pdf"))
    gst_pdf.close()
    
    # Dummy ITR file
    itr_text = "The Income Tax Act (ITR) specifies rules for filing income tax returns. Section 80C provides deductions for certain investments, such as life insurance."
    itr_pdf = fitz.open()
    page = itr_pdf.new_page()
    page.insert_text((72, 72), itr_text)
    itr_pdf.save(os.path.join(DATA_DIR, "itr_act_update.pdf"))
    itr_pdf.close()


def clean_up():
    """Removes dummy files and registry."""
    if os.path.exists(REGISTRY_PATH):
        os.remove(REGISTRY_PATH)
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, f))
        os.rmdir(DATA_DIR)


if __name__ == "__main__":
    print("--- Starting RAG Pipeline Test ---")
    
    # 1. Cleanup old files
    clean_up()
    print("Previous files and registry cleaned.")
    
    # 2. Create dummy data
    create_dummy_files()
    print("Dummy PDF files created.")

    # 3. Process the files using your data processing service
    files_to_process = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith((".pdf"))
    ]
    process_files(files_to_process)
    print("Files processed and stored in Pinecone and local registry.")
    
    # 4. Verify processing with a simple check
    try:
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
            print(f"Registry content: {list(registry.keys())}")
            assert len(registry) == 2, "Expected 2 documents in registry."
            print("Registry check passed. ✅")
    except (FileNotFoundError, AssertionError) as e:
        print(f"Registry check failed: {e}")

    # 5. Test the retrieval part of the pipeline
    print("\n--- Testing Retrieval ---")
    
    # Initialize Pinecone and create a retriever
    vector_store = initialize_pinecone_vectorstore(
        index_name=PINECONE_INDEX_NAME, embedding_model=EMBEDDING_MODEL
    )
    retriever = create_rag_retriever(vector_store)
    
    # Perform a test query
    query = "What is section 80C?"
    print(f"Performing test query: '{query}'")
    
    # The retriever should find the parent document from the ITR file
    retrieved_docs = retriever.invoke(query)
    
    print(f"Retrieved {len(retrieved_docs)} documents.")
    if retrieved_docs:
        print("First retrieved document content snippet:")
        print(retrieved_docs[0].page_content[:200] + "...")
        assert "80C" in retrieved_docs[0].page_content, "Retrieved document does not contain '80C'."
        print("Retrieval test passed. ✅")
    else:
        print("Retrieval test failed: No documents found.")
        
    print("\n--- RAG Pipeline Test Complete ---")