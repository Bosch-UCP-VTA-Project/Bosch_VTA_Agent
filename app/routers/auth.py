from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from uuid import uuid4
from app.utils.schema import (
    UserCreate,
    User,
    Token,
    UserInDB,
    UserRole,
)  # Import UserRole
from app.utils.helpers import (
    get_password_hash,
    authenticate_user,
    create_access_token,
    get_current_user,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    get_mongodb,
    oauth2_scheme,  # Import oauth2_scheme
    SECRET_KEY,  # Import SECRET_KEY
    ALGORITHM,  # Import ALGORITHM
    TokenData,  # Import TokenData
    JWTError,  # Import JWTError
    jwt,  # Import jwt
)

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={401: {"description": "Unauthorized"}},
)


# Dependency to get the current user and verify they are an admin
async def get_current_admin_user(
    token: str = Depends(oauth2_scheme), request: Request = None
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    role_exception = HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Operation not permitted. Admin role required.",
    )

    mongodb = request.app.state.mongodb

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        user_role: str = payload.get("role")  # Get role from token
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, role=user_role)
    except JWTError:
        raise credentials_exception

    user_dict = await mongodb.get_user_by_username(token_data.username)
    if user_dict is None:
        raise credentials_exception

    user = User(**user_dict)
    # Check if the role in the token and database matches 'admin'
    if token_data.role != UserRole.ADMIN or user.role != UserRole.ADMIN:
        raise role_exception

    return user


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

    # Use model_dump() instead of dict() for Pydantic v2+
    await mongodb.create_user(db_user.model_dump())

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
        data={"sub": user.username, "role": user.role.value},  # Include role in token
        expires_delta=access_token_expires,
    )

    return {"access_token": access_token, "token_type": "bearer"}


# New endpoint for admin login
@router.post("/admin/token", response_model=Token)
async def admin_login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), request: Request = None
):
    """Admin-specific login endpoint"""
    mongodb = request.app.state.mongodb
    user = await authenticate_user(mongodb, form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if the user has the admin role
    if user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Not an admin user",
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role.value},  # Include role
        expires_delta=access_token_expires,
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get information about the current logged in user"""
    return current_user


# Export the admin dependency for use in other routers
__all__ = ["get_current_admin_user"]
