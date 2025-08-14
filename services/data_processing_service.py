# data_processing_services.py
import re
import os
import json
import fitz
from datetime import datetime
from typing import List, Dict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv(override=True)

from utils.settings_loader import settings
from utils.config_loader import load_config
# from services.rag_retriever_service import setup_parent_document_retriever
from services.model_loader_service import ModelLoader

# --- CONFIGURATION ---
REGISTRY_PATH = "processed_docs.json"
PINECONE_INDEX = "gst-act-parent-store"
# PINECONE_INDEX = load_config()["pinecone_db"]["index_name"]
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
model_loader = ModelLoader()
EMBEDDING_MODEL = model_loader.load_embeddings()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
# embedding_function = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDING_MODEL)
# vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)



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


# --- DOCUMENT LEVEL METADATA ---
DOCUMENT_METADATA = {
    "document_title": "The Central Goods and Services Tax Act, 2017",
    "source_url": "https://www.cbic.gov.in/resources//htdocs-cbec/gst/CGST-Act-2020.pdf",
    "version": "As on 30.09.2020"
}





# ------------------------------------------------------------------
#               PRODUCTION DATA PROCESSING SERVICES
# ------------------------------------------------------------------



def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)
        
        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
        
        vectorstore.add_documents(splits)
        # vectorstore.persist()
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        if docs['ids']:
            # Delete the documents with the specified file_id
            vectorstore.delete(ids=docs['ids'])
            print(f"Deleted {len(docs['ids'])} document chunks with file_id {file_id}")
        else:
            print(f"No document chunks found with file_id {file_id}")
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False







# ------------------------------------------------------------------
#               EXPERIMENTALS DATA PROCESSING SERVICES
# ------------------------------------------------------------------


# --- CHUNKING FUNCTIONS ---
def get_text_from_doc(file_path: str) -> str:
    """Loads the PDF and extracts text."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def get_chunks_by_chatpter_and_section(text: str) -> List[Document]:
    """
    Chunks the document based on hierarchical structure and assigns metadata.
    
    Returns:
        A list of Document objects, where each document is a chunk with its metadata.
    """
    chunks = []
    chapter_pattern = r'(CHAPTER [IVXLCDM]+) (.+?)\n'
    chapter_matches = re.finditer(chapter_pattern, text)
    chapters = {m.group(1): m.group(2).strip() for m in chapter_matches}

    section_pattern = r'(\d+)\. (.+?)\n\s*(.*?)(?=\n\d+\. |\Z)'
    section_matches = re.finditer(section_pattern, text, re.DOTALL)
    
    for match in section_matches:
        section_number = match.group(1).strip()
        section_title = match.group(2).strip()
        content = match.group(0).strip()
        
        # Determine the parent chapter based on section number
        chapter_number = "Unknown"
        chapter_title = ""
        for chap_num, chap_name in chapters.items():
            if f'CHAPTER {chap_num}' in content:
                chapter_number = chap_num
                chapter_title = chap_name
                break
        
        # Add metadata to the chunk
        metadata = {
            "type": "Section",
            "document_title": DOCUMENT_METADATA["document_title"],
            "version": DOCUMENT_METADATA["version"],
            "section_number": section_number,
            "section_title": section_title,
            "chapter_number": chapter_number,
            "chapter_title": chapter_title,
        }
        
        chunks.append(Document(page_content=content, metadata=metadata))
    
    return chunks


# --- RETRIEVER SETUP ---
def setup_parent_document_retriever(index_name: str) -> ParentDocumentRetriever:
    """
    Sets up a ParentDocumentRetriever with a Pinecone vector store.
    """
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=EMBEDDING_MODEL)
    docstore = InMemoryStore()

    # Splitter for small, embedded chunks (child documents)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    # Splitter for large, parent documents (not embedded)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    
    return retriever


# --- MAIN PROCESSING FUNCTION ---
def process_files(files: List[str]) -> None:
    """
    Orchestrates the loading, chunking, and storage of new documents
    using the Parent Document Retriever strategy.
    """
    processed_docs = load_registry()
    retriever = setup_parent_document_retriever(index_name=PINECONE_INDEX)
    
    for file_path in files:
        doc_name = os.path.basename(file_path)

        if doc_name in processed_docs:
            print(f"Skipping already processed file: {doc_name}")
            continue

        print(f"Processing file: {doc_name}")
        ext = doc_name.split(".")[-1].lower()
        
        if ext == "pdf":
            try:
                # 1. Get text from the PDF
                text = get_text_from_doc(file_path)
                
                # 2. Chunk the text hierarchically with metadata
                parent_documents = get_chunks_by_chatpter_and_section(text)

                # 3. Add parent documents to the retriever
                retriever.add_documents(parent_documents)

                # 4. Update the registry
                processed_docs[doc_name] = {
                    "filename": doc_name,
                    "date_added": datetime.now().isoformat(),
                    "status": "success"
                }
                save_registry(processed_docs)
                print(f"Successfully processed and stored: {doc_name}")

            except Exception as e:
                print(f"Error storing {doc_name}: {e}")
                processed_docs[doc_name] = {
                    "filename": doc_name,
                    "date_added": datetime.now().isoformat(),
                    "status": "failed",
                    "error": str(e)
                }
                save_registry(processed_docs)
        else:
            print(f"Unsupported file type: {doc_name}")


