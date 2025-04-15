from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Dict
from app.utils.schema import ManualsResponse, User
from app.utils.helpers import get_current_user
from app.utils.rag import AutoTechnicianRAG
from llama_parse import LlamaParse
import os

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    responses={401: {"description": "Unauthorized"}},
)


# Dependency to get the RAG pipeline from the app state
async def get_rag_pipeline(app_state=Depends(lambda: router.app.state)):
    if not hasattr(app_state, "rag_pipeline"):
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    return app_state.rag_pipeline


# Dependency to get the LlamaParse instance from the app state
async def get_parser(app_state=Depends(lambda: router.app.state)):
    if not hasattr(app_state, "parser"):
        raise HTTPException(status_code=500, detail="Document parser not initialized")
    return app_state.parser


@router.post("/upload", response_model=Dict[str, str])
async def upload_document(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
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
        # Read file content
        file_content = await file.read()

        # Add metadata about who uploaded the file
        extra_info = {
            "file_name": file.filename,
            "uploaded_by": current_user.username,
            "upload_date": str(
                os.path.getmtime(file.filename)
                if os.path.exists(file.filename)
                else None
            ),
        }

        # Parse the PDF into markdown
        markdown_content = parser.load_data(file_content, extra_info)

        # Add the document to the RAG system
        success = rag_pipeline.add_documents(markdown_content)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to add document to the knowledge base",
            )

        return {"file_name": file.filename, "status": "processed"}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing document upload: {str(e)}"
        )


@router.get("/list", response_model=ManualsResponse)
async def list_documents(
    current_user: User = Depends(get_current_user),
    rag_pipeline: AutoTechnicianRAG = Depends(get_rag_pipeline),
):
    """List all available technical manuals/documents"""
    try:
        manuals = rag_pipeline.list_manuals()
        return ManualsResponse(manuals=manuals)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {str(e)}"
        )
