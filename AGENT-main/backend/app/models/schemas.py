from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    user_speech_text: str
    detected_language: str
    answer: str
    source_type: str
    source_name: str
