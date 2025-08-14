# rag_agent.py

import os
import json
import operator
import fitz
from datetime import datetime
from typing import TypedDict, List, Annotated, Literal
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
# from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from pydantic import BaseModel, Field
from typing import Optional
from pinecone import Pinecone

from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from services.rag_retriever_service import create_contextual_query_rewriter, initialize_pinecone_vectorstore, create_rag_retriever, setup_parent_document_retriever
from utils.settings_loader import settings
from utils.config_loader import load_config



os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
index_name = load_config()["pinecone_db"]["index_name"]
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


if getattr(settings, "LANGSMITH_API_KEY", None):
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
if getattr(settings, "LANGSMITH_PROJECT", None):
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
if getattr(settings, "LANGSMITH_TRACING_V2", None) is not None:
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGSMITH_TRACING_V2)

# =======================================================================================
# --- RAG AGENT IMPLEMENTATION ---
# =======================================================================================

# 1. Pydantic Evaluation Model
class Evaluation(BaseModel):
    """Structured output for the LLM as a judge."""
    relevance_score: int = Field(
        ...,
        description="A score from 1-10 on how relevant the retrieved documents are to the query."
    )
    groundedness_score: int = Field(
        ...,
        description="A score from 1-10 on how well the generated answer is supported by the retrieved documents."
    )
    is_hallucination: bool = Field(
        ...,
        description="True if the answer contains information not supported by the documents, False otherwise."
    )
    feedback: str = Field(
        ...,
        description="Constructive feedback on how to improve the response."
    )


# 2. Graph State Definition
class AgentState(TypedDict):
    """
    A custom state for the LangGraph agent.
    """
    # Changed from 'messages' to 'chat_history' for consistency with your code.
    chat_history: Annotated[List[BaseMessage], operator.add]
    query: str
    retrieved_docs: List[BaseMessage]
    final_answer: str
    evaluation: Optional[Evaluation] = None


# 3. Node Definitions
llm = ChatGoogleGenerativeAI(model="gemma-3-12b-it")
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")



def retrieve_node(state: AgentState):
    """
    Retrieves documents based on the query, using a contextual rewriter.
    """
    # print("---NODE: Retrieval---")
    # query = state["query"]
    # chat_history = state["chat_history"]

    # # Use the contextual query rewriter from your provided code
    # # prompt_rewriter = create_contextual_query_rewriter(llm)
    # # rewritten_prompt = prompt_rewriter.invoke({"chat_history": chat_history, "input": query})
    
    # class NewQuery(BaseModel):
    #     new_query: str = Field("Given the above conversation, generate a concise, keyword-only search query to look up the user's question. For example, if the question is 'What is the capital of France?', a good query would be 'capital of France'. Do not include any conversational phrases or full sentences. Return only the keywords.")
    # structured_llm = llm.with_structured_output(NewQuery)
    # rewritten_prompt = structured_llm.invoke({"chat_history": chat_history, "input": query})
    # print(f"--- Rewritten query: {rewritten_prompt} ---")
        
    # print("--- Initializing mock retriever ---")
    # # vector_store = initialize_pinecone_vectorstore(index_name, EMBEDDING_MODEL)
    # # retriever = create_rag_retriever(vector_store, search_type="mmr")
    # retriever = setup_parent_document_retriever(index_name)
    
    # retrieved_docs = retriever.invoke(rewritten_prompt)

    # print(f"--- Documents retrieved successfully. Found {len(retrieved_docs)} documents. ---")
    # return {"retrieved_docs": retrieved_docs, "query": rewritten_prompt}
    
    print("---NODE: Retrieval---")
    query = state["query"]
    chat_history = state["chat_history"]

    # Use the pre-existing function to rewrite the query
    prompt_rewriter = create_contextual_query_rewriter(llm)
    rewritten_query = prompt_rewriter.invoke({"chat_history": chat_history, "input": query})
    
    print(f"--- Rewritten query: {rewritten_query} ---")
        
    print("--- Initializing mock retriever ---")
    retriever = setup_parent_document_retriever(index_name)
    
    # Pass the string output of the rewriter to the retriever
    retrieved_docs = retriever.invoke(rewritten_query)

    print(f"--- Documents retrieved successfully. Found {len(retrieved_docs)} documents. ---")
    return {"retrieved_docs": retrieved_docs, "query": rewritten_query}
    
    
    



def generate_node(state: AgentState):
    """
    Generates an answer using the retrieved documents and chat history.
    """
    print("---NODE: Generation---")
    query = state["query"]
    docs = state["retrieved_docs"]

    # Combine docs into a single string for context
    context = "\n\n".join([doc.content for doc in docs])

    # Prompt for generation, including chat history for context
    generation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the following context to answer the user's question. If the context does not contain the answer, politely state that you do not have the information.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{query}"),
    ])

    generation_chain = generation_prompt | llm | StrOutputParser()
    final_answer = generation_chain.invoke({"context": context, "chat_history": state["chat_history"], "query": query})

    print("--- Answer generated successfully. ---")
    return {"final_answer": final_answer}


def evaluate_node(state: AgentState):
    """
    Evaluates the generated answer using an LLM as a judge with a Pydantic model.
    """
    print("---NODE: Evaluation---")
    parser = PydanticOutputParser(pydantic_object=Evaluation)
    
    evaluation_prompt = PromptTemplate(
        template="""
        You are an expert AI evaluation judge. Your task is to critique an AI's response
        based on a user query and a set of retrieved documents.

        User Query: {query}
        Retrieved Documents: {retrieved_docs}
        AI Generated Answer: {final_answer}

        Instructions:
        1. Rate the relevance of the documents to the query on a scale of 1 to 10.
        2. Rate the groundedness of the answer (how well it is supported by the documents) on a scale of 1 to 10.
        3. Determine if the answer contains any hallucinations (information not present in the documents).
        4. Provide constructive feedback on the answer.

        {format_instructions}
        """,
        input_variables=["query", "retrieved_docs", "final_answer"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    evaluation_chain = evaluation_prompt | llm | parser

    evaluation_result = evaluation_chain.invoke({
        "query": state["query"],
        "retrieved_docs": "\n\n".join([doc.content for doc in state["retrieved_docs"]]),
        "final_answer": state["final_answer"]
    })

    print("--- Evaluation complete. ---")
    print(f"Evaluation Results:\n{evaluation_result.model_dump_json(indent=2)}")

    return {"evaluation": evaluation_result}


# 4. Conditional Edges
def decide_to_evaluate(state: AgentState):
    """
    Decides whether to proceed to evaluation or end the graph.
    """
    print("---DECISION: Evaluate---")
    # For this example, we always evaluate.
    return "evaluate"


# 5. Build the LangGraph
def build_rag_graph():
    """
    Builds and compiles the LangGraph.
    """
    # Create the checkpointer for persistent memory
    # memory = SqliteSaver.from_file("rag_agent_memory.sqlite")
    memory = MemorySaver()

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_conditional_edges(
        "generate",
        decide_to_evaluate,
        {
            "evaluate": "evaluate",
            "end": END,
        },
    )
    workflow.add_edge("evaluate", END)

    app = workflow.compile(checkpointer=memory)

    return app


# Main execution block
if __name__ == "__main__":
    # if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
    #     print("Warning: GOOGLE_API_KEY and/or PINECONE_API_KEY not set. Using mock keys.")

    # app = build_rag_graph()
    # config = {"configurable": {"thread_id": "1"}}

    # print("\n" + "="*50 + "\n")
    # # Example 1: New conversation
    # query_1 = "What are income tax act rules in india, what is section 2a?"
    # print(f"\n--- Starting a new conversation with query: '{query_1}' ---")
    # inputs = {"query": query_1, "chat_history": [HumanMessage(content=query_1)]}
    # for s in app.stream(inputs, config=config):
    #     print(s)
    
    # print("\n" + "="*50 + "\n")

    # # Example 2: Continuing the conversation in the same thread
    # query_2 = "What is gst return policy and gst for laptop?"
    # print(f"--- Continuing conversation in thread_id='1' with query: '{query_2}' ---")
    # # The `chat_history` will be loaded from the checkpoint
    # inputs = {"query": query_2, "chat_history": [HumanMessage(content=query_2)]}
    # for s in app.stream(inputs, config=config):
    #     print(s)
    
    # print("\n" + "="*50 + "\n")

    # # Example 3: New conversation with a new thread
    # query_3 = "What company has highest profit in sp500?"
    # print(f"--- Starting a new conversation with query: '{query_3}' ---")
    # inputs = {"query": query_3, "chat_history": [HumanMessage(content=query_3)]}
    # for s in app.stream(inputs, config=config):
    #     print(s)

    # print("\n" + "="*50 + "\n")
    
    # chat_history = []
    
    # while True:
    #     user_query = input("\n👤 You: ")
        
    #     if user_query.lower() in ["quit", "exit", "q", "e"]:
    #         print("Goodbye! 👋")
    #         break
        
    #     # Add user's message to the chat history
    #     chat_history.append(HumanMessage(content=user_query))
        
    #     # Prepare inputs for the RAG graph
    #     inputs = {"query": user_query, "chat_history": chat_history}
        
    #     print("\n🤖 AI:")
    #     ai_response_full = ""
    #     try:
    #         # Stream the response from the RAG graph
    #         for s in app.stream(inputs, config=config):
    #             for key, value in s.items():
    #                 if key == "answer":
    #                     response_text = value.get("response", "")
    #                     print(response_text, end="", flush=True)
    #                     ai_response_full += response_text
    #     except Exception as e:
    #         print(f"An error occurred: {e}")
            
    #     # Add the AI's full response to the chat history
    #     chat_history.append(AIMessage(content=ai_response_full))
        
    #     print("\n" + "="*50)
    
    # print("--- Testing the retrieve_node function directly ---")
    # ans = retrieve_node({
    #     "query": "what boat headphone are best under 1000",
    #     "chat_history": [HumanMessage(content="what headphones should i buy?")],
    #     "retrieved_docs": [],
    #     "final_answer": "",
    # })
    
    # # The `retrieve_node` returns the state, so you need to access the docs from the returned dictionary.
    # print("\nResult of retrieve_node:")
    # print(f"Rewritten Query: {ans.get('query')}")
    # print(f"Number of Retrieved Docs: {len(ans.get('retrieved_docs'))}")
    # if ans.get('retrieved_docs'):
    #     print(f"Content of first doc: {ans['retrieved_docs'][0].page_content[:150]}...")
        
    # print("\n" + "="*50 + "\n")
    # print(ans)
    
    
    # --- TEST 5: create_contextual_query_rewriter ---
    print("\n--- Testing create_contextual_query_rewriter ---")
    try:
        rewriter_chain = create_contextual_query_rewriter(llm)
        dummy_chat_history = [
            HumanMessage(content="What are the rules for income tax?"),
            AIMessage(content="I'm sorry, I cannot provide legal advice. However, I can look up general information about income tax rules if you provide a more specific question."),
        ]
        query = "what is the tax rate for electronics in india"
        rewritten_query = rewriter_chain.invoke({"chat_history": dummy_chat_history, "input": query})
        print(f"Original Query: '{query}'")
        print(f"Rewritten Query: '{rewritten_query}'")
    except Exception as e:
        print(f"Failed to create and run contextual query rewriter: {e}")    