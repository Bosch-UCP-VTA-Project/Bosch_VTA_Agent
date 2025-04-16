from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
import traceback
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
async def get_rag_pipeline(request: Request):
    if not hasattr(request.app.state, "rag_pipeline"):
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    return request.app.state.rag_pipeline


# Dependency to get the Groq client from the app state
async def get_groq_client(request: Request):
    if not hasattr(request.app.state, "groq_client"):
        raise HTTPException(status_code=500, detail="Groq client not initialized")
    return request.app.state.groq_client


@router.post("/query", response_model=ChatResponse)
async def query(
    request: QueryRequest,
    req: Request,
    current_user: User = Depends(get_current_user),
    rag_pipeline: AutoTechnicianRAG = Depends(get_rag_pipeline),
):
    """Process a text query and return a response with sources"""
    try:
        mongodb = req.app.state.mongodb

        # Use the provided session_id or generate a new one
        session_id = request.session_id or generate_session_id()

        # If new session, create session record
        if not request.session_id:
            await mongodb.create_session(
                {
                    "session_id": session_id,
                    "user_id": current_user.id,
                    "username": current_user.username,
                    "created_at": datetime.utcnow().isoformat(),
                }
            )

        # Store user query in MongoDB
        await mongodb.add_chat_message(
            {
                "session_id": session_id,
                "role": "user",
                "content": request.query,
                "user_id": current_user.id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Process the query
        result = rag_pipeline.query(request.query, session_id)

        # Store assistant response in MongoDB
        # Ensure source_nodes are consistently formatted as dictionaries
        source_nodes_data = []
        if result.source_nodes:
            for node in result.source_nodes:
                if isinstance(node, dict):
                    source_nodes_data.append(node)
                elif hasattr(node, "to_dict"):
                    source_nodes_data.append(node.to_dict())
                else:
                    print(f"Warning: Unexpected node type: {type(node)}")
                    # Convert to a reasonable default format if needed
                    source_nodes_data.append({"text": str(node), "score": "0.0"})

        await mongodb.add_chat_message(
            {
                "session_id": session_id,
                "role": "assistant",
                "content": result.answer,
                "source_nodes": source_nodes_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Get chat history from MongoDB
        chat_history = await mongodb.get_chat_history(session_id)

        # Convert MongoDB records to expected format
        formatted_history = []
        for msg in chat_history:
            entry = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
            }
            if msg["role"] == "assistant" and "source_nodes" in msg:
                # Just directly add the source_nodes, schema now accepts Any type
                entry["source_nodes"] = msg["source_nodes"]
            formatted_history.append(entry)

        return ChatResponse(
            answer=result.answer,
            session_id=session_id,
            history=formatted_history,
            source_nodes=result.source_nodes,
        )
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Query processing error: {str(e)}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/history", response_model=ChatHistory)
async def history(
    request: HistoryRequest,
    req: Request,
    current_user: User = Depends(get_current_user),
):
    """Get chat history for a specific session"""
    try:
        mongodb = req.app.state.mongodb
        session_id = request.session_id

        # Get chat history from MongoDB
        chat_history = await mongodb.get_chat_history(session_id)

        # Convert MongoDB records to expected format
        formatted_history = []
        for msg in chat_history:
            entry = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
            }
            if msg["role"] == "assistant" and "source_nodes" in msg:
                # Direct assignment now works with updated schema
                entry["source_nodes"] = msg["source_nodes"]
            formatted_history.append(entry)

        return ChatHistory(
            history=formatted_history,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving history: {str(e)}"
        )


@router.post("/audio", response_model=AudioResponse)
async def audio_query(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    req: Request = None,
    current_user: User = Depends(get_current_user),
    rag_pipeline: AutoTechnicianRAG = Depends(get_rag_pipeline),
    groq_client: Groq = Depends(get_groq_client),
):
    """Process an audio query, transcribe it, and return a response with sources"""
    try:
        mongodb = req.app.state.mongodb

        # Read audio content
        audio_content = await audio.read()

        # Get the filename and actual content type
        filename = audio.filename or "recording.wav"
        content_type = audio.content_type or "audio/wav"

        # Log details to help debug
        print(
            f"Processing audio: filename={filename}, content_type={content_type}, size={len(audio_content)} bytes"
        )

        # Use appropriate file extension based on content type
        file_ext = "wav"
        if "mp3" in content_type:
            file_ext = "mp3"
        elif "ogg" in content_type or "opus" in content_type:
            file_ext = "ogg"

        # Transcribe the audio using Groq
        try:
            translation = groq_client.audio.translations.create(
                file=(f"recording.{file_ext}", audio_content),
                model="whisper-large-v3",
                prompt="Vehicle repair technical context",
                response_format="json",
                temperature=0.0,
            )

            query_text = translation.text
            print(f"Transcription successful: '{query_text}'")
        except Exception as transcription_error:
            print(f"Transcription error: {str(transcription_error)}")
            traceback_str = traceback.format_exc()
            print(traceback_str)
            raise HTTPException(
                status_code=500,
                detail=f"Audio transcription failed: {str(transcription_error)}",
            )

        # Use the provided session_id or generate a new one
        session_id = session_id or generate_session_id()

        # If new session, create session record
        if not session_id:
            await mongodb.create_session(
                {
                    "session_id": session_id,
                    "user_id": current_user.id,
                    "username": current_user.username,
                    "created_at": datetime.utcnow().isoformat(),
                    "input_type": "audio",
                }
            )

        # Store user query in MongoDB
        await mongodb.add_chat_message(
            {
                "session_id": session_id,
                "role": "user",
                "content": query_text,
                "user_id": current_user.id,
                "input_type": "audio",
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Process the query
        result = rag_pipeline.query(query_text, session_id)

        # Store assistant response in MongoDB
        # Check if source_nodes are dict objects or objects with to_dict method
        source_nodes_data = []
        if result.source_nodes:
            for node in result.source_nodes:
                if isinstance(node, dict):
                    source_nodes_data.append(node)
                elif hasattr(node, "to_dict"):
                    source_nodes_data.append(node.to_dict())
                else:
                    print(f"Warning: Unexpected node type: {type(node)}")

        await mongodb.add_chat_message(
            {
                "session_id": session_id,
                "role": "assistant",
                "content": result.answer,
                "source_nodes": source_nodes_data,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )

        # Get chat history from MongoDB
        chat_history = await mongodb.get_chat_history(session_id)

        # Convert MongoDB records to expected format
        formatted_history = []
        for msg in chat_history:
            entry = {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
            }
            if msg["role"] == "assistant" and "source_nodes" in msg:
                # Just directly add the source_nodes array
                entry["source_nodes"] = msg["source_nodes"]
            formatted_history.append(entry)

        return AudioResponse(
            answer=result.answer,
            transcribed=query_text,
            history=formatted_history,
            source_nodes=result.source_nodes,
            session_id=session_id,
        )
    except Exception as e:
        # Enhanced error logging
        error_trace = traceback.format_exc()
        print(f"Audio processing error: {str(e)}\n{error_trace}")
        raise HTTPException(
            status_code=500, detail=f"Error processing audio query: {str(e)}"
        )


@router.get("/new-session")
async def new_session(req: Request, current_user: User = Depends(get_current_user)):
    """Generate a new session ID for the user"""
    session_id = generate_session_id()

    # Create the session in MongoDB
    mongodb = req.app.state.mongodb
    await mongodb.create_session(
        {
            "session_id": session_id,
            "user_id": current_user.id,
            "username": current_user.username,
            "created_at": datetime.utcnow().isoformat(),
        }
    )

    return {"session_id": session_id}


@router.get("/sessions")
async def get_user_sessions(
    req: Request,
    current_user: User = Depends(get_current_user),
):
    """Get all chat sessions for the current user"""
    try:
        mongodb = req.app.state.mongodb
        sessions = await mongodb.get_user_sessions(current_user.id)

        # Format the response
        formatted_sessions = []
        for session in sessions:
            # Get the first message from each session to use as a title
            first_message = await mongodb.get_first_message(session["session_id"])
            title = first_message["content"] if first_message else "New conversation"

            # Truncate title if too long
            if len(title) > 50:
                title = title[:47] + "..."

            formatted_sessions.append(
                {
                    "id": session["session_id"],
                    "title": title,
                    "created_at": session["created_at"],
                }
            )

        return {"sessions": formatted_sessions}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving sessions: {str(e)}"
        )
