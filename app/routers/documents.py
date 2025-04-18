from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from typing import Dict, List  # Import List
from app.utils.schema import (
    ManualsResponse,
    User,
)
from app.routers.auth import get_current_admin_user
from app.utils.rag import AutoTechnicianRAG
from llama_parse import LlamaParse
from datetime import datetime
import os

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(get_current_admin_user)],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden - Admin role required"},
    },
)


# Dependency to get the RAG pipeline from the app state
async def get_rag_pipeline(request: Request):  # Changed from app_state to request
    if not hasattr(request.app.state, "rag_pipeline"):
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    return request.app.state.rag_pipeline


# Dependency to get the LlamaParse instance from the app state
async def get_parser(request: Request):  # Changed from app_state to request
    if not hasattr(request.app.state, "parser"):
        raise HTTPException(status_code=500, detail="Document parser not initialized")
    return request.app.state.parser


@router.post("/upload", response_model=Dict[str, str])
async def upload_document(
    file: UploadFile = File(...),
    request: Request = None,
    # Use the admin user dependency here
    current_user: User = Depends(get_current_admin_user),
    rag_pipeline: AutoTechnicianRAG = Depends(get_rag_pipeline),
    parser: LlamaParse = Depends(get_parser),
):
    """Upload and process a PDF document to add to the knowledge base"""
    # Verify file is a PDF
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are accepted.",
        )

    try:
        mongodb = request.app.state.mongodb

        # Read file content
        file_content = await file.read()

        upload_time = datetime.utcnow().isoformat()

        # Add metadata about who uploaded the file
        extra_info = {
            "file_name": file.filename,
            "uploaded_by": current_user.username,
            "upload_date": upload_time,
        }

        # Parse the PDF into markdown
        # Assuming parser.load_data accepts bytes and extra_info
        # Check LlamaParse documentation if this signature is correct
        markdown_documents = await parser.aload_data(
            file_content, extra_info=extra_info
        )

        # Add the document to the RAG system
        success = rag_pipeline.add_documents(markdown_documents)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add document to the knowledge base",
            )

        # Store document metadata in MongoDB
        await mongodb.add_document(
            {
                "file_name": file.filename,
                "uploaded_by": current_user.username,
                "user_id": current_user.id,
                "upload_date": upload_time,
                "status": "processed",
            }
        )

        return {"file_name": file.filename, "status": "processed"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document upload: {str(e)}"
        )


@router.get(
    "/list", response_model=ManualsResponse
)  # Ensure ManualsResponse matches the MongoDB document structure or create a new response model
async def list_documents(
    request: Request = None,
    # Use the admin user dependency here
    current_user: User = Depends(get_current_admin_user),
    # rag_pipeline: AutoTechnicianRAG = Depends(get_rag_pipeline),  # No longer needed here
):
    """List all available technical manuals/documents from MongoDB"""
    try:
        mongodb = request.app.state.mongodb

        # Fetch documents directly from MongoDB
        mongo_docs: List[Dict] = (
            await mongodb.list_documents()
        )  # Use the MongoDB method

        # Ensure the documents have the 'file_name' key expected by ManualsResponse
        # If your MongoDB documents have a different structure, you might need to transform them
        # or adjust the ManualsResponse model in schema.py
        manuals_list = [
            {"file_name": doc.get("file_name", "Unknown Filename")}
            for doc in mongo_docs
        ]

        return ManualsResponse(
            manuals=manuals_list
        )  # Return the list fetched from MongoDB
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents from MongoDB: {str(e)}"
        )
