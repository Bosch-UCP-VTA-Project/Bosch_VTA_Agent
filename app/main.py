from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import traceback
from groq import Groq
from llama_parse import LlamaParse
from dotenv import load_dotenv
from app.utils.rag import AutoTechnicianRAG
from app.routers import auth, chat, documents
from app.utils.mongodb import MongoDB
from contextlib import asynccontextmanager

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize services on startup
    # Initialize MongoDB
    try:
        app.state.mongodb = MongoDB()
        print("MongoDB connection initialized successfully.")
    except Exception as e:
        print(f"Error initializing MongoDB: {str(e)}")
        raise

    # Initialize document parser
    app.state.parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        verbose=True,
    )

    # Initialize Groq client for audio transcription
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    app.state.groq_client = Groq(api_key=groq_api_key)

    # Initialize RAG pipeline
    manuals_path = os.getenv("MANUALS_PATH", "./data/technical_manuals")
    online_resources_path = os.getenv(
        "ONLINE_RESOURCES_PATH", "./data/online_resources"
    )

    # Ensure data directories exist
    os.makedirs(manuals_path, exist_ok=True)
    os.makedirs(online_resources_path, exist_ok=True)

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if not qdrant_url or not qdrant_api_key:
        raise ValueError(
            "Qdrant configuration missing. Check QDRANT_URL and QDRANT_API_KEY environment variables."
        )

    try:
        app.state.rag_pipeline = AutoTechnicianRAG(
            manuals_path=manuals_path,
            online_resources_path=online_resources_path,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        )
        print("RAG pipeline initialized successfully.")
    except Exception as e:
        print(f"Error initializing RAG pipeline: {str(e)}")
        raise

    yield  # This is where the FastAPI application runs

    # Cleanup on shutdown
    if hasattr(app.state, "mongodb"):
        app.state.mongodb.close()
        print("MongoDB connection closed.")


app = FastAPI(
    title="Bosch VTA Agent",
    description="An advanced automotive technical assistant API with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(documents.router)


@app.get("/", tags=["health"])
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/health", tags=["health"])
async def health_check():
    """Detailed health check of all services"""
    health_status = {
        "api": "healthy",
        "services": {
            "rag_pipeline": hasattr(app.state, "rag_pipeline"),
            "groq_client": hasattr(app.state, "groq_client"),
            "parser": hasattr(app.state, "parser"),
            "mongodb": hasattr(app.state, "mongodb"),
        },
    }

    if not all(health_status["services"].values()):
        return {"status": "degraded", "details": health_status}

    return {"status": "healthy", "details": health_status}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with detailed error logging"""
    error_trace = traceback.format_exc()
    error_message = f"Unhandled exception: {str(exc)}\n{error_trace}"
    print(error_message)

    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "path": request.url.path},
    )


def run_server():
    """Entry point for running the server with uv"""
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )


if __name__ == "__main__":
    run_server()
