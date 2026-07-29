"""
Ops Copilot orchestrator
=========================
A LangGraph state machine:

    router ──▶ [conditional] ──▶ tool_use ──▶ report ──▶ notify ──▶ END
           └───────────────────────────────────▶ report  (skip path)

    router     -> classifies each ticket's priority using the LLM,
                  grounded in policy context pulled from the RAG
                  knowledge base (retrieval_agent.py)
    [routing]  -> conditional edge: if NO ticket needs a follow-up task,
                  skip tool_use entirely and go straight to report
    tool_use   -> for tickets that need a follow-up, calls create_task
                  on the Task MCP server
    report     -> asks the LLM to write a status summary from the
                  structured results
    notify     -> posts that summary to a team channel via post_message
                  on a second, separate MCP server (Notify), showing
                  multi-platform/multi-server integration
    END

Uses a local Ollama model (no API key, no cost) via ChatOllama. Defaults
to `phi3` (lightweight, CPU-friendly). Run `ollama pull phi3` and make
sure `ollama serve` is running before use. If you have more RAM (16GB+)
and want stronger reasoning, set OLLAMA_MODEL=llama3.1 in .env instead.

Run directly for a CLI smoke test:
    python agents/orchestrator.py

Or import `run_workflow(tickets)` from the FastAPI layer (api/main.py).
"""

import asyncio
import json
import os
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, StateGraph

from agents.retrieval_agent import retrieve_policy_context

load_dotenv()

ROOT = Path(__file__).parent.parent
TASK_SERVER_PATH = ROOT / "mcp_server" / "task_server.py"
NOTIFY_SERVER_PATH = ROOT / "mcp_server" / "notify_server.py"

# Shared config for both MCP servers this graph talks to. Both are spawned
# as local stdio subprocesses; in production these `command`/`args` would
# instead point at hosted MCP servers over HTTP/SSE.
MCP_SERVERS = {
    "tasks": {
        "command": "python",
        "args": [str(TASK_SERVER_PATH)],
        "transport": "stdio",
    },
    "notify": {
        "command": "python",
        "args": [str(NOTIFY_SERVER_PATH)],
        "transport": "stdio",
    },
}


class TicketResult(TypedDict):
    ticket_id: str
    subject: str
    priority: str
    needs_task: bool
    task_id: str | None


class GraphState(TypedDict):
    tickets: list[dict]
    # No reducer annotation: each node that touches "results" returns the
    # full, authoritative list for that point in the pipeline (router
    # produces it, tool_use replaces it with task_id populated). Plain
    # TypedDict fields are overwritten on update, not merged, by default.
    results: list[TicketResult]
    report: str


llm = ChatOllama(
    model=os.environ.get("OLLAMA_MODEL", "phi3"),
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
)

# Ollama models are noticeably less reliable than Claude at returning bare,
# unwrapped JSON from a plain prompt -- they'll often add a preamble or
# wrap the object in ```json fences. Ollama's `format="json"` mode forces
# valid-JSON output at the sampling level, so the router uses a separate
# instance with that flag set rather than relying on prompt wording alone.
llm_json = ChatOllama(
    model=os.environ.get("OLLAMA_MODEL", "phi3"),
    base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    temperature=0,
    format="json",
)


def _parse_json_response(text: str) -> dict:
    """Defensive parse: strips ```json fences some local models still add
    even in JSON mode, before falling back to raw json.loads."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


ROUTER_PROMPT = """You are triaging a support ticket according to the team's policy.

Policy context (retrieved from the knowledge base):
{policy_context}

Ticket:
Subject: {subject}
Body: {body}

Classify this ticket's priority as exactly one of: high, medium, low.
Then decide whether it needs a follow-up task created (true/false) per
the policy's task-creation rules.

Respond ONLY with compact JSON: {{"priority": "...", "needs_task": true/false}}
"""


async def router_node(state: GraphState) -> dict:
    """Classify every ticket's priority + task-need, grounded in RAG context."""
    results: list[TicketResult] = []
    for ticket in state["tickets"]:
        context = retrieve_policy_context(
            f"priority rules for ticket: {ticket['subject']} {ticket['body']}"
        )
        prompt = ROUTER_PROMPT.format(
            policy_context=context, subject=ticket["subject"], body=ticket["body"]
        )
        response = await llm_json.ainvoke(prompt)
        parsed = _parse_json_response(response.content)
        results.append(
            TicketResult(
                ticket_id=ticket["id"],
                subject=ticket["subject"],
                priority=parsed["priority"],
                needs_task=parsed["needs_task"],
                task_id=None,
            )
        )
    return {"results": results}


def route_after_router(state: GraphState) -> str:
    """Conditional edge: only spin up the tool_use node (and its MCP
    subprocess) if at least one ticket actually needs a follow-up task.
    Otherwise skip straight to the report — no point paying the cost of
    a tool call for a batch of all-low-priority tickets."""
    if any(r["needs_task"] for r in state["results"]):
        return "tool_use"
    return "report"


def _parse_tool_result(raw) -> dict:
    """MCP tool results come back in different shapes depending on the
    adapter version and whether the server returned structured content
    vs. plain text: sometimes a JSON string, sometimes an already-parsed
    dict, sometimes a list of content blocks like
    [{"type": "text", "text": "..."}]. Normalize all of them to a dict.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, list):
        if not raw:
            return {}
        first = raw[0]
        if isinstance(first, dict):
            # content-block shape: {"type": "text", "text": "<json>"}
            if "text" in first:
                return json.loads(first["text"])
            return first
        if isinstance(first, str):
            return json.loads(first)
    raise ValueError(f"Unexpected MCP tool result shape: {type(raw)} -> {raw!r}")


async def tool_use_node(state: GraphState) -> dict:
    """For tickets flagged needs_task, call the MCP task server's
    create_task tool. Uses MultiServerMCPClient to spawn the MCP server
    as a subprocess over stdio and expose its tools as callables."""
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    create_task = next(t for t in tools if t.name == "create_task")

    updated: list[TicketResult] = []
    for result in state["results"]:
        if result["needs_task"]:
            task = await create_task.ainvoke(
                {
                    "title": f"Follow up: {result['subject']}",
                    "priority": result["priority"],
                    "source_ticket_id": result["ticket_id"],
                }
            )
            task_data = _parse_tool_result(task)
            result = {**result, "task_id": task_data.get("id")}
        updated.append(result)

    # Overwrite rather than append since this node fully supersedes the
    # router's results with task_id populated.
    return {"results": updated}


REPORT_PROMPT = """Write a short, professional status report (markdown,
under 200 words) for the operations team based on this structured ticket
triage data:

{data}

Include: total tickets, breakdown by priority, and a bulleted list of
follow-up tasks created with their IDs.
"""


async def report_node(state: GraphState) -> dict:
    prompt = REPORT_PROMPT.format(data=json.dumps(state["results"], indent=2))
    response = await llm.ainvoke(prompt)
    return {"report": response.content}


async def notify_node(state: GraphState) -> dict:
    """Post the finished report to a team channel via the Notify MCP
    server. Separate MCP server from tool_use's task server, and
    separate node from report_node, so each node has exactly one
    responsibility and each MCP server has exactly one client
    connection lifecycle to reason about."""
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    post_message = next(t for t in tools if t.name == "post_message")

    await post_message.ainvoke(
        {"channel": "#ops-status", "message": state["report"]}
    )
    return {}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("router", router_node)
    graph.add_node("tool_use", tool_use_node)
    graph.add_node("report", report_node)
    graph.add_node("notify", notify_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"tool_use": "tool_use", "report": "report"},
    )
    graph.add_edge("tool_use", "report")
    graph.add_edge("report", "notify")
    graph.add_edge("notify", END)

    return graph.compile()


async def run_workflow(tickets: list[dict]) -> GraphState:
    graph = build_graph()
    final_state = await graph.ainvoke({"tickets": tickets, "results": [], "report": ""})
    return final_state


if __name__ == "__main__":
    tickets = json.loads((ROOT / "data" / "tickets.json").read_text())
    final_state = asyncio.run(run_workflow(tickets))
    print("\n=== TRIAGE RESULTS ===")
    print(json.dumps(final_state["results"], indent=2))
    print("\n=== REPORT ===")
    print(final_state["report"])
    print("\n(posted to #ops-status — see data/slack_log.json)")