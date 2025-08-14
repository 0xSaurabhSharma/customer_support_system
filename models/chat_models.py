from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime

class ModelName(str, Enum):
    # GPT4_1 = "gpt-4.1"
    # GPT4_1_MINI = "gpt-4.1-mini"
    # GEMINI_2_0_FLASH_LITE = "GEMINI_2_0_FLASH_LITE"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    model: ModelName = Field(default=ModelName.GEMINI_2_0_FLASH_LITE)

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: ModelName

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int