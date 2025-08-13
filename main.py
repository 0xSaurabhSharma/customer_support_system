# main.py
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from utils.settings_loader import Settings
from api.api import router as api_router

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load settings (your settings_loader should provide a Settings class)
settings = Settings()

# LangSmith environment (optional)
if getattr(settings, "LANGSMITH_API_KEY", None):
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
if getattr(settings, "LANGSMITH_PROJECT", None):
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
if getattr(settings, "LANGSMITH_TRACING_V2", None) is not None:
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGSMITH_TRACING_V2)

app = FastAPI(title="Flipkart Product Assistant")

# static + templates
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# CORS (loose for dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include API router (defined in api.py)
app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)












# import os
# import uvicorn
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import ChatPromptTemplate

# from utils.settings_loader import Settings
# from retrievers.retriever import Retriever
# from utils.model_loader import ModelLoader
# from prompts.prompt import PROMPT_TEMPLATES


# app = FastAPI()
# settings = Settings()


# os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
# os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
# os.environ["LANGSMITH_TRACING_V2"] = str(settings.LANGSMITH_TRACING_V2)


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
# templates = Jinja2Templates(directory="templates")


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# retriever_obj = Retriever()
# model_loader = ModelLoader()


# def invoke_chain(query:str):
    
#     retriever=retriever_obj.load_retriever()
#     prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATES["product_bot"])
#     llm= model_loader.load_llm()
    
#     chain=(
#         {"context": retriever, "question": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )
#     output=chain.invoke(query)
    
#     return output
    
    

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     """
#     Render the chat interface.
#     """
#     return templates.TemplateResponse("chat.html", {"request": request})

# @app.post("/get",response_class=HTMLResponse)
# async def chat(msg:str=Form(...)):
#     result=invoke_chain(msg)
#     print(f"Response: {result}")
#     return result

# if __name__ == "__main__":
#     uvicorn.run(app=app, host="127.0.0.1", port=8000)