from typing import TypedDict, List, Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from services.model_loader_service import ModelLoader

model_loader = ModelLoader()
llm = model_loader.load_llm()


# ── Pydantic schemas 
class RouteDecision(BaseModel):
    route: Literal["rag", "answer", "end"]
    reply: str | None = Field(None, description="Filled only when route == 'end'")

class RagJudge(BaseModel):
    sufficient: bool



# ── LLM instances with structured output where needed 
router_llm = (llm.with_structured_output(RouteDecision)).with_config(run_name="Router Node")
judge_llm  = (llm.with_structured_output(RagJudge)).with_config(run_name="Judge Node")
answer_llm = (llm).with_config(run_name="Answer Node")



# ── Shared state type 
class AgentState(TypedDict, total=False):
    messages: List[BaseMessage]
    route:    Literal["rag", "answer", "end"]
    rag:      str
    web:      str 