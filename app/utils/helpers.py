import os
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timedelta
from duckduckgo_search import DDGS
from llama_index.core.tools import FunctionTool
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
from app.utils.schema import User, TokenData, UserInDB

load_dotenv()

# Authentication utilities
SECRET_KEY = os.getenv("SECRET_KEY", "super-secret-key-replace-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

# In-memory user database for demo purposes
# In production, replace with a proper database
fake_users_db = {}


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_user(db, username: str) -> Optional[UserInDB]:
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None


def authenticate_user(db, username: str, password: str) -> Optional[UserInDB]:
    user = get_user(db, username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=payload.get("role"))
    except JWTError:
        raise credentials_exception

    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception

    return User(
        id=user.id,
        username=user.username,
        email=user.email,
        role=user.role,
        created_at=user.created_at,
    )


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


def generate_session_id() -> str:
    """Generate a unique session ID for chat interactions"""
    return str(uuid4())
