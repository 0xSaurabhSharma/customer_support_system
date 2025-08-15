import os
import logging
import shutil
from dotenv import load_dotenv

from fastapi import APIRouter, Request, Form, UploadFile, File, HTTPException, FastAPI
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from models.chat_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from prompts.prompt import PROMPT_TEMPLATES
from retrievers.retriever import Retriever
from utils.model_loader import ModelLoader
from utils.chat_utils import get_or_create_session_id, history_to_lc_messages, append_message
from utils.langchain_utils import contextualise_chain
from services.db_service import insert_chat_history, get_chat_history, get_all_documents, insert_document_record, delete_document_record
from services.data_processing_service import index_document_to_chroma, delete_doc_from_chroma
from agents.simple_rag_apent import agent
from utils.langchain_utils import check_policy, handle_guardrail_failure


# ==================================================================
#  GLOBAL SETUP
# ==================================================================

load_dotenv(override=True)
logging.basicConfig(filename='agent.log', level=logging.INFO)
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

retriever_obj = Retriever()
model_loader = ModelLoader()
llm_guard = model_loader.load_safeguard()


# ==================================================================
#  HELPER FUNCTIONS
# ==================================================================

async def invoke_chain(query: str) -> str:
    retriever = retriever_obj.load_retriever()
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["product_bot"])
    llm = model_loader.load_llm()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    ).with_config(run_name="Cupport RAG")

    try:
        output = await chain.ainvoke(query) 
        return str(output)
    except Exception as exc:
        logger.exception("Chain invocation failed")
        return f"[error] {exc}"


# ==================================================================
#  FASTAPI ENDPOINTS
# ==================================================================


# ------------------------------------------------------------------
#               AGENT RAG ENDPOINTS
# ------------------------------------------------------------------

@router.post("/chat", response_model=QueryResponse, tags=["agent"])
def chat_agent(query_input: QueryInput):
    try:
        # --- Step 1: Validate Session ---
        session_id = get_or_create_session_id(query_input.session_id)

        # --- Step 2: INPUT Safety Guard ---
        safety_result = llm_guard.invoke(query_input.question)
        if "unsafe" in safety_result.content.lower():
            return QueryResponse(
                answer=handle_guardrail_failure("safety"),
                session_id=session_id,
                model=query_input.model
            )

        # --- Step 3: Get chat history & contextualize ---
        chat_history = get_chat_history(session_id)
        messages = history_to_lc_messages(chat_history)
        standalone_q = contextualise_chain.invoke({
            "chat_history": messages,
            "input": query_input.question,
        })

        # --- Step 4: SAFETY check on contextualized query ---
        contextual_safety = llm_guard.invoke(standalone_q)
        if "unsafe" in contextual_safety.content.lower():
            return QueryResponse(
                answer=handle_guardrail_failure("safety"),
                session_id=session_id,
                model=query_input.model
            )

        # --- Step 5: Run RAG Agent ---
        messages = append_message(messages, HumanMessage(content=standalone_q))
        result = agent.invoke({"messages": messages})

        last_message = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        answer = last_message.content if last_message else "I couldn't generate a response at this time."

        # --- Step 6: OUTPUT Policy Guard ---
        # policy_check_result = check_policy.invoke({
        #     "user_question": query_input.question,
        #     "rag_answer": answer
        # })

        # final_answer = answer if policy_check_result == "SAFE" else policy_check_result

        # --- Step 7: Store in history ---
        insert_chat_history(session_id, query_input.question, answer, query_input.model.value)

        return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")



@router.post("/upload-doc", tags=["agent"])
def upload_and_index_document(file: UploadFile = File(...)):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(temp_file_path, file_id)
        
        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.get("/list-docs", response_model=list[DocumentInfo], tags=["agent"])
def list_documents():
    return get_all_documents()

@router.post("/delete-doc", tags=["agent"])
def delete_document(request: DeleteFileRequest):
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}


# ------------------------------------------------------------------
#               STREAMLIT END POINTS
# ------------------------------------------------------------------

@router.post("/get", response_class=PlainTextResponse)
async def chat(msg: str = Form(...)):
    # --- Step 1: Input Guardrail with Llama Guard ---
    safety_result = llm_guard.invoke(msg)
    if "unsafe" in safety_result.content.lower():
        return handle_guardrail_failure("safety")

    # --- Step 2: Normal RAG Call ---
    rag_answer = await invoke_chain(msg)
    
    # --- Step 3: Output Guardrail with Custom Policy Check ---
    policy_check = check_policy.invoke({"user_question": msg, "rag_answer": rag_answer})

    if policy_check != "SAFE":
        return policy_check 
    else:
        return rag_answer 


# @router.post("/get", response_class=PlainTextResponse)
# async def chat(msg: str = Form(...)):
#     result = invoke_chain(msg)
#     logger.info("Query: %s | Response length: %d", msg[:80], len(result))
#     return result

    
@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

