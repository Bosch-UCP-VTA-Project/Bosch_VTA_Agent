import os
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timedelta
from duckduckgo_search import DDGS
from llama_index.core.tools import FunctionTool
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
from app.utils.schema import User, TokenData, UserInDB

load_dotenv()

# JWT configuration
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "your-secret-key"
)  # Better to use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Get MongoDB from request
async def get_mongodb(request: Request):
    return request.app.state.mongodb


async def authenticate_user(mongodb, username: str, password: str):
    user_dict = await mongodb.get_user_by_username(username)
    if not user_dict:
        return False

    user = UserInDB(**user_dict)
    if not verify_password(password, user.hashed_password):
        return False

    return user


async def get_current_user(
    token: str = Depends(oauth2_scheme), request: Request = None
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    mongodb = request.app.state.mongodb

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user_dict = await mongodb.get_user_by_username(username)
    if user_dict is None:
        raise credentials_exception

    return User(**user_dict)


def generate_session_id():
    import uuid

    return str(uuid.uuid4())


# RAG utilities
def duckduckgo_search_tool():
    def _ddgs_search(input: str) -> List[Dict[str, Any]]:
        with DDGS() as ddgs:
            results = []
            for r in ddgs.text(input, max_results=3):
                results.append(r)
        return results

    return FunctionTool.from_defaults(
        name="duckduckgo_search",
        description="Search the web for relevant information about automotive questions. Use it to find up-to-date information that might not be in your technical manuals or online resources.",
        fn=_ddgs_search,
    )
