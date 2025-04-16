from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# Authentication schemas
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: UserRole = UserRole.USER


class UserLogin(BaseModel):
    username: str
    password: str


class UserInDB(BaseModel):
    id: str
    username: str
    email: str
    role: UserRole
    hashed_password: str
    created_at: datetime


class User(BaseModel):
    id: str
    username: str
    email: str
    role: UserRole
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str
    role: Optional[str] = None


# Chat schemas
class ChatSession(BaseModel):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatHistory(BaseModel):
    history: List[ChatMessage]


class Source(BaseModel):
    content: str
    score: Optional[str] = None
    text: Optional[str] = None


class HistoryRequest(BaseModel):
    session_id: str


class QueryRequest(BaseModel):
    query: str
    session_id: str


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    history: List[Dict[str, str]]
    source_nodes: List[Dict[str, str]]


class AudioResponse(BaseModel):
    answer: str
    transcribed: str
    history: List[Dict[str, Any]]
    source_nodes: Optional[List] = None
    session_id: str


class ManualsResponse(BaseModel):
    manuals: List[Dict[str, str]]
