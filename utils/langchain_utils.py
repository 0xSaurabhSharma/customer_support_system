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
from prompts.prompt import PROMPT_TEMPLATES




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

contextualize_q_system_prompt = PROMPT_TEMPLATES["contextualize_q_system_prompt"]

CONTEXT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

contextualise_chain = (CONTEXT_PROMPT | llm | StrOutputParser()).with_config(run_name="query-rewrite")



policy_prompt_template = PROMPT_TEMPLATES["policy_prompt_template"]

policy_chain = (ChatPromptTemplate.from_template(policy_prompt_template) | llm).with_config(run_name="Policy Check")

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

# def check_prompt_safety(prompt: str):
#     """Safety and Policy security check"""
#     safety = llm_guard.invoke(prompt)
    
#     if safety == "safe":
#         policy = """ some prompt"""
#         policy_prompt = ChatPromptTemplate.from_messages([
#             ("system", policy),
#             ("human", "{input}")
#         ])
#         chain = (policy_prompt | llm ).with_config(run_name="Guard Check")
        
#         policy_check = chain.invoke({"input": prompt})
        
#         if policy_check == "safe":
#             return True
    
#     return False

def handle_guardrail_failure(error_type: str) -> str:
    """Returns a user-friendly message for a guardrail failure."""
    if error_type == "safety":
        return "I cannot answer this request as it violates safety guidelines. 🚫"
    elif error_type == "policy":
        return "I cannot fulfill this request due to our business policies. Please rephrase your question or visit our website for more information. 🙏"
    return "An unexpected error occurred."

