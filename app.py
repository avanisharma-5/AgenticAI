from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from models import QueryRequest, WorkflowResponse
from crew_system import run_insurance_agents


app = FastAPI(title="Collaborative Insurance Agents API", version="1.0.0")
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "message": "Collaborative Insurance Agents API is running.",
        "frontend": "http://localhost:5173",
        "health": "/health",
        "run_workflow": "/workflow/run",
        "upload_file": "/files/upload",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/workflow/run", response_model=WorkflowResponse)
def run_workflow(request: QueryRequest) -> WorkflowResponse:
    # Create one unique ID per request.
    request_id = str(uuid4())

    # Run the main multi-agent logic.
    output = run_insurance_agents(request.question)
    state = {
        "request_id": request_id,
        "question": request.question,
        "retrieved_facts": [],
        "draft": None,
        "final_output": output,
        "status": "completed",
    }

    # Hand-off protocol event (Researcher -> Writer).
    handoffs = [
        {
            "from_agent": "Researcher Agent",
            "to_agent": "Writer Agent",
            "reason": "Research evidence gathered (RAG PDF or SerpAPI fallback), then synthesized for final response.",
            "payload_preview": request.question[:80],
        }
    ]

    # Basic heuristics for better UI state.
    if output.startswith("I can help with insurance questions"):
        state["status"] = "blocked"
    elif "couldn’t find grounded insurance information" in output or "couldn't find grounded insurance information" in output:
        state["status"] = "no_info"

    return WorkflowResponse(
        request_id=request_id,
        status=state["status"],
        handoffs=handoffs,  # pydantic will coerce dicts
        state=state,
    )


@app.post("/files/upload")
async def upload_file(file: UploadFile = File(...)) -> dict:
    # Save uploaded file with a generated ID prefix.
    file_id = str(uuid4())
    safe_name = Path(file.filename or "upload.bin").name
    dest = UPLOAD_DIR / f"{file_id}__{safe_name}"
    content = await file.read()
    dest.write_bytes(content)
    return {
        "file_id": file_id,
        "filename": safe_name,
        "content_type": file.content_type,
        "size": len(content),
    }
