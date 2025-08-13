# api.py
import os
import logging
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from retrievers.retriever import Retriever
from utils.model_loader import ModelLoader
from prompts.prompt import PROMPT_TEMPLATES

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

router = APIRouter()

# Instantiate long-lived objects once
retriever_obj = Retriever()
model_loader = ModelLoader()
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)


def invoke_chain(query: str) -> str:
    """
    Build and run a simple chain using retriever + prompt + LLM.
    Return plain text output (best-effort).
    """
    # load retriever (returns a retriever-like runnable or object expected by your chain)
    retriever = retriever_obj.load_retriever()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["product_bot"])

    # load llm (provider/config handled in your ModelLoader)
    llm = model_loader.load_llm()

    # Compose a chain using runnables — keep same structure you had
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke chain (may raise; catch and return error)
    try:
        output = chain.invoke(query)
        return str(output)
    except Exception as exc:
        logger.exception("Chain invocation failed")
        return f"[error] {exc}"


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the chat HTML template (templates/chat.html must exist).
    """
    return templates.TemplateResponse("chat.html", {"request": request})


@router.post("/get", response_class=PlainTextResponse)
async def chat(msg: str = Form(...)):
    """
    Receive form POST from web UI (or JSON via `{"msg": "..."}`
    and return the raw assistant text.
    """
    
    result = invoke_chain(msg)
    logger.info("Query: %s | Response length: %d", msg[:80], len(result))
    return result


flow =    """

memory: config & checkpoint in memory    

req: username + query -> graph.invoke(query, config)

graph: state -> graph build -> perform rag + + + 


    
    """