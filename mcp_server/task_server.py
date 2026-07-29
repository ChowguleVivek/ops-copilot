"""
Task MCP Server
================
A minimal MCP server that exposes a small "task board" as tools an agent
can call. Backed by a local JSON file so it works with zero external
accounts. In a real deployment you'd swap the storage layer for the
Notion/Trello/Linear API and keep the same tool signatures — that's the
whole point of building this as an MCP server rather than a hard-coded
function: any MCP-compatible agent/client can plug into it unchanged.

Run standalone for local testing:
    python mcp_server/task_server.py

Run as an MCP server (stdio transport) for the orchestrator to connect to:
    it is spawned automatically by agents/orchestrator.py via
    langchain_mcp_adapters.client.MultiServerMCPClient
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

TASK_STORE_PATH = Path(os.environ.get("TASK_STORE_PATH", "./data/tasks.json"))

mcp = FastMCP("ops-copilot-tasks")


def _load() -> list[dict]:
    if not TASK_STORE_PATH.exists():
        TASK_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        TASK_STORE_PATH.write_text("[]")
    return json.loads(TASK_STORE_PATH.read_text())


def _save(tasks: list[dict]) -> None:
    TASK_STORE_PATH.write_text(json.dumps(tasks, indent=2))


@mcp.tool()
def create_task(title: str, priority: str = "medium", source_ticket_id: str = "") -> dict:
    """Create a new follow-up task on the team's task board.

    Args:
        title: Short, actionable task description.
        priority: One of "low", "medium", "high".
        source_ticket_id: Optional ID of the support ticket this task
            originated from, for traceability.
    """
    tasks = _load()
    task = {
        "id": str(uuid.uuid4())[:8],
        "title": title,
        "priority": priority,
        "source_ticket_id": source_ticket_id,
        "status": "open",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tasks.append(task)
    _save(tasks)
    return task


@mcp.tool()
def list_tasks(status: str = "open") -> list[dict]:
    """List tasks on the board, optionally filtered by status.

    Args:
        status: "open", "done", or "all".
    """
    tasks = _load()
    if status == "all":
        return tasks
    return [t for t in tasks if t["status"] == status]


@mcp.tool()
def complete_task(task_id: str) -> dict:
    """Mark a task as done.

    Args:
        task_id: The id returned by create_task or list_tasks.
    """
    tasks = _load()
    for t in tasks:
        if t["id"] == task_id:
            t["status"] = "done"
            t["completed_at"] = datetime.now(timezone.utc).isoformat()
            _save(tasks)
            return t
    raise ValueError(f"No task with id {task_id}")


if __name__ == "__main__":
    # stdio transport — this is what MultiServerMCPClient spawns as a subprocess
    mcp.run(transport="stdio")
