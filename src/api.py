
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from .rag_pipeline import RAGPipeline

# --- API SETUP ---
app = FastAPI(
    title="Production-Ready RAG API",
    description="An API for asking questions to a RAG pipeline with reranking.",
    version="1.0.0",
)

# --- DATA MODELS ---
class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    question: str

class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str

# --- LOAD PIPELINE ---
# This is loaded once at startup and reused for all requests.
# This can take a few minutes depending on the model sizes.
print("Loading RAG pipeline...")
pipeline = RAGPipeline()
print("Pipeline loaded and ready.")


# --- API ENDPOINTS ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint providing basic API information."""
    return {"message": "Welcome to the RAG API. Use the /ask endpoint to ask questions."}

@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_question(request: AskRequest):
    """
    Receives a question, processes it through the RAG pipeline, and returns the answer.
    """
    answer = pipeline.ask(request.question)
    return {"answer": answer}

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load server configuration from config.yaml
    import yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    server_config = config.get("server", {})
    host = server_config.get("host", "0.0.0.0")
    port = server_config.get("port", 8000)

    print(f"Starting FastAPI server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
