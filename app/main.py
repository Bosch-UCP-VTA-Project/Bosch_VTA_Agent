from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from groq import Groq
from llama_parse import LlamaParse
from dotenv import load_dotenv
from app.utils.rag import AutoTechnicianRAG
from app.routers import auth, chat, documents

load_dotenv()

app = FastAPI(
    title="Bosch VTA Agent",
    description="An advanced automotive technical assistant API with RAG capabilities",
    version="1.0.0",
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


# Initialize services on startup
@app.on_event("startup")
async def startup_event():
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
        },
    }

    if not all(health_status["services"].values()):
        return {"status": "degraded", "details": health_status}

    return {"status": "healthy", "details": health_status}

def run_server():
    """Entry point for running the server with uv"""
    import uvicorn

    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )


if __name__ == "__main__":
    run_server()
