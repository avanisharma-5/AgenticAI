from pydantic import BaseModel, Field
from typing import Dict, List, Optional


# Input payload sent by UI.
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Insurance user query")
    file_ids: List[str] = Field(default_factory=list, description="Uploaded file IDs")
    context: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Optional extra context from UI inputs",
    )


# One handoff record between agents.
class HandoffEvent(BaseModel):
    from_agent: str
    to_agent: str
    reason: str
    payload_preview: str


# Current workflow state for UI rendering.
class AgentState(BaseModel):
    request_id: str
    question: str
    retrieved_facts: List[str] = Field(default_factory=list)
    draft: Optional[str] = None
    final_output: Optional[str] = None
    status: str = "created"


# Final response envelope from /workflow/run.
class WorkflowResponse(BaseModel):
    request_id: str
    status: str
    handoffs: List[HandoffEvent]
    state: AgentState
