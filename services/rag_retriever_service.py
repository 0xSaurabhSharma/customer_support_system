# rag_retriever_service.py

import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.retrievers import BaseRetriever
from typing import Literal
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from utils.config_loader import load_config
from utils.settings_loader import settings

index_name = load_config()["pinecone_db"]["index_name"]
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY





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
    
    
    
# def create_contextual_query_rewriter(llm: ChatGoogleGenerativeAI):
#     """
#     Creates an LCEL chain to rewrite a query based on chat history.

#     Args:
#         llm (ChatGoogleGenerativeAI): The LLM to use for rewriting the query.

#     Returns:
#         RunnableSerializable: An LCEL chain for contextual query rewriting.
#     """
#     prompt = ChatPromptTemplate.from_messages(
#         [
#             MessagesPlaceholder(variable_name="chat_history"),
#             ("user", "{input}"),
#             ("system", "Given the above conversation, generate a search query to look up in order to get a better response to the user's question. Refrain from including any conversational phrases. Just return the search query."),
#         ]
#     )
#     return prompt | llm | StrOutputParser()