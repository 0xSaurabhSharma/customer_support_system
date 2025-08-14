# rag_retriever_service.py

import os
from typing import Literal
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from utils.config_loader import load_config
from utils.settings_loader import settings

# --- CONFIGURATION ---
index_name = "gst-act-parent-store"
# index_name = load_config()["pinecone_db"]["index_name"]
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(model="gemma-3-12b-it")


def initialize_pinecone_vectorstore(
    index_name: str,
    embedding_model: GoogleGenerativeAIEmbeddings
) -> PineconeVectorStore:
    """
    Initializes and returns a PineconeVectorStore object.
    
    Args:
        index_name (str): The name of the Pinecone index.
        embedding_model (GoogleGenerativeAIEmbeddings): The embedding model to use.
        
    Returns:
        PineconeVectorStore: An initialized Pinecone vector store instance.
    """
    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding_model
    )
    return vector_store


def setup_parent_document_retriever(index_name: str) -> ParentDocumentRetriever:
    """
    Sets up a ParentDocumentRetriever with an in-memory store for parent documents.
    
    Args:
        index_name (str): The name of the Pinecone index.
        
    Returns:
        ParentDocumentRetriever: An initialized retriever instance.
    """
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=EMBEDDING_MODEL)
    docstore = InMemoryStore()

    # Splitter for small, embedded chunks
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


def create_rag_retriever(
    vector_store: PineconeVectorStore,
    search_type: Literal["similarity", "mmr"] = "mmr",
    k: int = 4,
    fetch_k: int = 20,
    lambda_mult: float = 0.5
) -> BaseRetriever:
    """
    Creates and configures a LangChain retriever from a Pinecone vector store.
    
    Args:
        vector_store (PineconeVectorStore): The initialized Pinecone vector store.
        search_type (Literal["similarity", "mmr"]): The search type to use.
                                                     Defaults to "mmr".
        k (int): The number of documents to return from the retriever.
        fetch_k (int): The number of documents to fetch before filtering for MMR.
                       Only used if search_type is "mmr".
        lambda_mult (float): The diversity parameter for MMR.
                             A value of 0 is maximum diversity, and 1 is minimum.
                             Only used if search_type is "mmr".
                             
    Returns:
        BaseRetriever: A configured retriever instance.
    """
    if search_type == "mmr":
        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
            },
        )
    else:  # Defaults to "similarity"
        return vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k": k,
            },
        )
        

def create_multi_query_retriever(
    base_retriever: BaseRetriever,
    llm: ChatGoogleGenerativeAI
) -> MultiQueryRetriever:
    """
    Creates a MultiQueryRetriever to generate multiple queries from a single input.

    Args:
        base_retriever (BaseRetriever): The underlying retriever to use for each generated query.
        llm (ChatGoogleGenerativeAI): The LLM to use for generating new queries.

    Returns:
        MultiQueryRetriever: A retriever that generates and runs multiple queries.
    """
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever, llm=llm
    )
    
    
def query_rewriter(query: dict):
    """
    Creates an LCEL chain to rewrite a query based on chat history and invokes it.
    Args:
        query (dict): A dictionary containing the user's input and chat history.
                      Expected keys: 'input' (str) and 'chat_history' (List[BaseMessage]).
    Returns:
        str: The rewritten query as a string.
    """
    
    query_rewrite_template = """You are an AI assistant tasked with generating broader, more general queries to improve context retrieval in a RAG system.
    Given the original query, generate a step-back query that is more general and can help retrieve relevant background information.
    Chat History: {chat_history}
    Original query: {input}
    Step-back query:"""
    
    query_rewrite_prompt = PromptTemplate(
        input_variables=["input", "chat_history"],
        template=query_rewrite_template
    )

    query_rewrite_chain = query_rewrite_prompt | llm
    rewritten_query = query_rewrite_chain.invoke({
        "input": query["input"],
        "chat_history": query["chat_history"]
    })
    return str(rewritten_query.content)
    
    


if __name__ == "__main__":       
    # # --- Test Script for setup_parent_document_retriever ---
    # print("\n--- Testing setup_parent_document_retriever ---")
    # # To test this, you first need to add documents to the retriever.
    # # The ParentDocumentRetriever manages its own documents.
    # from langchain_core.documents import Document

    # parent_docs_to_add = [
    #     Document(
    #         page_content="The new tax law introduces a 15% flat rate on small business profits up to $50,000. It's designed to simplify the tax code and encourage entrepreneurship."
    #     ),
    #     Document(
    #         page_content="A detailed report on the new tax reforms shows that a 28% GST rate applies to electronics like televisions and smartphones, while groceries are exempt."
    #     )
    # ]
    # parent_retriever.add_documents(parent_docs_to_add)
    query = "What does 'appointed day' mean according to the act?"
    rewritten_result = query_rewriter({"input": query, "chat_history": []})
    print("-----------------------------------------")
    print(f"Original: {query}")
    print(f"Rewritten: {rewritten_result}")
    print("-----------------------------------------")


    parent_retriever = setup_parent_document_retriever(index_name)
    parent_docs = parent_retriever.invoke(query)
    print(f"Retrieved {len(parent_docs)} documents with ParentDocumentRetriever.")
    print("-----------------------------------------")
    print(parent_docs)
    print("-----------------------------------------")
    

    print("\n--- Testing create_multi_query_retriever ---")
    vector_store_multi = initialize_pinecone_vectorstore(index_name, EMBEDDING_MODEL)
    base_retriever = vector_store_multi.as_retriever()
    multi_query_retriever = create_multi_query_retriever(base_retriever, llm)
    multi_query_docs = multi_query_retriever.invoke(query)
    print(f"Retrieved {len(multi_query_docs)} documents with MultiQueryRetriever.")
    
    print("-----------------------------------------")
    print(multi_query_docs)
    print("-----------------------------------------")
    
    
    ## => try setup_parent_document_retriever if doc == 0 then multi_query_retriever