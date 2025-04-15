from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from typing import Optional
from app.utils.schema import (
    QueryRequest,
    ChatResponse,
    HistoryRequest,
    ChatHistory,
    AudioResponse,
    User,
)
from app.utils.helpers import get_current_user, generate_session_id
from app.utils.rag import AutoTechnicianRAG
from groq import Groq
from datetime import datetime

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    responses={401: {"description": "Unauthorized"}},
)


# Dependency to get the RAG pipeline from the app state
async def get_rag_pipeline(app_state=Depends(lambda: router.app.state)):
    if not hasattr(app_state, "rag_pipeline"):
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    return app_state.rag_pipeline


# Dependency to get the Groq client from the app state
async def get_groq_client(app_state=Depends(lambda: router.app.state)):
    if not hasattr(app_state, "groq_client"):
        raise HTTPException(status_code=500, detail="Groq client not initialized")
    return app_state.groq_client


@router.post("/query", response_model=ChatResponse)
async def query(
    request: QueryRequest,
    current_user: User = Depends(get_current_user),
    rag_pipeline: AutoTechnicianRAG = Depends(get_rag_pipeline),
):
    """Process a text query and return a response with sources"""
    try:
        # Use the provided session_id or generate a new one
        session_id = request.session_id or generate_session_id()

        # Store user information with the query for audit/logging
        user_context = {
            "user_id": current_user.id,
            "username": current_user.username,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Process the query
        result = rag_pipeline.query(request.query, session_id)
        history = rag_pipeline.get_history(session_id)

        return ChatResponse(
            answer=result.answer,
            session_id=session_id,
            history=history,
            source_nodes=result.source_nodes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/history", response_model=ChatHistory)
async def history(
    request: HistoryRequest,
    current_user: User = Depends(get_current_user),
    rag_pipeline: AutoTechnicianRAG = Depends(get_rag_pipeline),
):
    """Get chat history for a specific session"""
    try:
        session_id = request.session_id
        history = rag_pipeline.get_history(session_id)
        return ChatHistory(
            history=history,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving history: {str(e)}"
        )


@router.post("/audio", response_model=AudioResponse)
async def audio_query(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    rag_pipeline: AutoTechnicianRAG = Depends(get_rag_pipeline),
    groq_client: Groq = Depends(get_groq_client),
):
    """Process an audio query, transcribe it, and return a response with sources"""
    try:
        # Read audio content
        audio_content = await audio.read()

        # Transcribe the audio using Groq
        translation = groq_client.audio.translations.create(
            file=("recording.wav", audio_content),
            model="whisper-large-v3",
            prompt="Vehicle repair technical context",
            response_format="json",
            temperature=0.0,
        )

        query_text = translation.text

        # Use the provided session_id or generate a new one
        session_id = session_id or generate_session_id()

        # Store user information with the query
        user_context = {
            "user_id": current_user.id,
            "username": current_user.username,
            "timestamp": datetime.utcnow().isoformat(),
            "input_type": "audio",
        }

        # Process the query
        result = rag_pipeline.query(query_text, session_id)
        history = rag_pipeline.get_history(session_id)

        return AudioResponse(
            answer=result.answer,
            transcribed=query_text,
            history=history,
            source_nodes=result.source_nodes,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing audio query: {str(e)}"
        )


@router.get("/new-session")
async def new_session(current_user: User = Depends(get_current_user)):
    """Generate a new session ID for the user"""
    return {"session_id": generate_session_id()}
