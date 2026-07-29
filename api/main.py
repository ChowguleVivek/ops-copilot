"""
Thin FastAPI wrapper around the orchestrator so the workflow can be
triggered as a web app / hit from Slack, a cron job, curl, etc.

Run:
    uvicorn api.main:app --reload

Then:
    curl -X POST http://localhost:8000/run-triage
"""

import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from agents.orchestrator import run_workflow

app = FastAPI(title="Ops Copilot")

ROOT = Path(__file__).parent.parent
DEFAULT_TICKETS_PATH = ROOT / "data" / "tickets.json"


class Ticket(BaseModel):
    id: str
    subject: str
    body: str
    customer: str = ""


class TriageRequest(BaseModel):
    tickets: list[Ticket] | None = None  # if omitted, uses data/tickets.json


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/run-triage")
async def run_triage(req: TriageRequest = TriageRequest()):
    tickets = (
        [t.model_dump() for t in req.tickets]
        if req.tickets
        else json.loads(DEFAULT_TICKETS_PATH.read_text())
    )
    final_state = await run_workflow(tickets)
    return {
        "results": final_state["results"],
        "report": final_state["report"],
    }
