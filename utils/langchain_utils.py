import os
import logging
from dotenv import load_dotenv

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from utils.model_loader import ModelLoader
from utils.settings_loader import settings


# ==================================================================
#  GLOBAL SETUP & MODEL INITIALIZATION
# ==================================================================

load_dotenv(override=True)
os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model_loader = ModelLoader()
llm_guard = model_loader.load_safeguard()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
# llm = model_loader.load_llm()


# ==================================================================
#  PROMPTS & CHAINS
# ==================================================================

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is. "
    "⟹ Return **only** the reformulated question (no explanations, no answers)."
)

CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

contextualise_chain = (CONTEXT_PROMPT | llm | StrOutputParser()).with_config(run_name="query-rewrite")

policy_prompt_template = """
You are a chatbot policy enforcer for Boat headphones. Your job is to check a user's question and a RAG-generated answer against the following rules.

**Rules:**
1.  **NO Pricing:** Do not provide any prices, discounts, or special offers. Redirect to the official website for pricing.
2.  **NO Comparisons:** Do not compare Boat products with competitor brands.
3.  **NO Support:** Do not process refunds, returns, or warranty claims. Redirect to the support page.

**Task:**
Analyze the user's question and the RAG-generated response. If the response violates any of the rules, you MUST respond with a concise, policy-compliant message. If the response is safe and does not violate any rules, you MUST respond with the exact word "SAFE".

**User Question:** {user_question}
**RAG-Generated Answer:** {rag_answer}
"""

policy_chain = ChatPromptTemplate.from_template(policy_prompt_template) | llm

@chain
def check_policy(inputs: dict) -> str:
    """
    Runs the custom policy check using a single dictionary input.
    """
    user_question = inputs["user_question"]
    rag_answer = inputs["rag_answer"]

    policy_check = policy_chain.invoke({"user_question": user_question, "rag_answer": rag_answer})

    return policy_check.content


# ==================================================================
#  HELPER FUNCTIONS & MODELS
# ==================================================================

def check_prompt_safety(prompt: str):
    """Safety and Policy security check"""
    safety = llm_guard.invoke(prompt)
    
    if safety == "safe":
        policy = """ some prompt"""
        policy_prompt = ChatPromptTemplate.from_messages([
            ("system", policy),
            ("human", "{input}")
        ])
        chain = policy_prompt | llm 
        
        policy_check = chain.invoke({"input": prompt})
        
        if policy_check == "safe":
            return True
    
    return False

def handle_guardrail_failure(error_type: str) -> str:
    """Returns a user-friendly message for a guardrail failure."""
    if error_type == "safety":
        return "I cannot answer this request as it violates safety guidelines. 🚫"
    elif error_type == "policy":
        return "I cannot fulfill this request due to our business policies. Please rephrase your question or visit our website for more information. 🙏"
    return "An unexpected error occurred."

