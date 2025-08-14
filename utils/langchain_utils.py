import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from services.model_loader_service import ModelLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.settings_loader import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY
model_loader = ModelLoader()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite")
# llm = model_loader.load_llm()

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


contextualise_chain = ( CONTEXT_PROMPT | llm | StrOutputParser()).with_config(run_name="query-rewrite")