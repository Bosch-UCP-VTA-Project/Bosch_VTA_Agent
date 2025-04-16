from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from uuid import uuid4
from app.utils.schema import UserCreate, User, Token, UserInDB
from app.utils.helpers import (
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_mongodb,
)

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={401: {"description": "Unauthorized"}},
)


@router.post("/register", response_model=User)
async def register_user(user_data: UserCreate, request: Request):
    """Register a new user and return user information"""
    mongodb = request.app.state.mongodb

    existing_user = await mongodb.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    user_id = str(uuid4())
    hashed_password = get_password_hash(user_data.password)

    db_user = UserInDB(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        role=user_data.role,
        hashed_password=hashed_password,
        created_at=datetime.utcnow(),
    )

    await mongodb.create_user(db_user.dict())

    return User(
        id=user_id,
        username=user_data.username,
        email=user_data.email,
        role=user_data.role,
        created_at=db_user.created_at,
    )


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), request: Request = None
):
    """Get access token for logged in user"""
    mongodb = request.app.state.mongodb
    user = await authenticate_user(mongodb, form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires,
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get information about the current logged in user"""
    return current_user
