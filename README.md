# Ops Copilot — Multi-Agent Support Triage Assistant

A small multi-agent system that triages support tickets, creates
follow-up tasks on a team board, and generates a status report —
end to end, with no human in the loop for the routine cases.

Built to explore agentic workflow orchestration, MCP-based tool
integration, and RAG-grounded decision-making rather than just a
single-shot "chatbot that answers questions."

## Architecture

```
                       ┌──▶ tool_use ──┐
tickets.json ──▶ router┤               ├──▶ report ──▶ notify ──▶ END
                       └───────────────┘
                 (conditional: skips tool_use if
                  no ticket needs a follow-up task)
```

- **Router agent** — for each ticket, retrieves relevant policy text from
  a Chroma vector store (RAG) and asks the LLM to classify priority and
  decide whether a follow-up task is needed, grounded in that policy
  rather than the model's own judgment call.
- **Conditional routing** — a LangGraph conditional edge inspects the
  router's output and skips the `tool_use` node (and the subprocess it
  would spawn) entirely if no ticket needs a follow-up — an explicit
  graph-level branch, not just an `if` inside one node.
- **Tool-use agent** — for tickets that need follow-up, calls a
  `create_task` tool exposed by a custom **MCP server**
  (`mcp_server/task_server.py`). Backed by a local JSON file here, but
  the tool contract is identical to what you'd expose for a real
  Notion/Trello/Linear integration — swap the storage layer, keep the
  agent code unchanged.
- **Report agent** — turns the structured triage results into a short
  markdown status report.
- **Notify agent** — posts that report to a team channel via
  `post_message` on a **second, independent MCP server**
  (`mcp_server/notify_server.py`), demonstrating multi-platform
  integration rather than a single tool wired into everything. Backed by
  a local log file here; swap in `slack_sdk` for a real Slack post.
- **Orchestration** — LangGraph `StateGraph` wires all four agents as
  nodes, with one conditional edge, and explicit typed state passed
  between them.
- **Web layer** — FastAPI exposes the whole pipeline as `POST
  /run-triage`, so it can be triggered from a cron job, Slack slash
  command, or curl.

## Setup

**1. Install [Ollama](https://ollama.com)** (free, runs locally, no API key, no subscription):

```bash
# macOS: brew install ollama
# Windows/Linux: download from https://ollama.com/download

ollama pull llama3.1     # ~4.7GB download, needs ~8GB RAM free
                           # lighter laptop? use `ollama pull mistral` or `phi3`
                           #   and set OLLAMA_MODEL accordingly in .env
ollama serve               # leave this running in its own terminal
```

**2. Set up the Python project:**

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # defaults work out of the box, no keys needed
```

**3. Build the knowledge base index** (run once, or whenever
`knowledge_base/sample_docs/*.md` changes):

```bash
python knowledge_base/ingest.py
```

## Run it

**CLI smoke test** (processes `data/tickets.json`, prints results + report):

```bash
python -m agents.orchestrator
```

**As a web app:**

```bash
uvicorn api.main:app --reload
curl -X POST http://localhost:8000/run-triage
```

## Project layout

```
ops-copilot/
├── mcp_server/
│   ├── task_server.py         # MCP server: create_task, list_tasks, complete_task
│   └── notify_server.py       # MCP server: post_message (mock Slack channel)
├── knowledge_base/
│   ├── sample_docs/            # policy docs the RAG agent grounds decisions in
│   └── ingest.py                # embeds docs into a persisted Chroma collection
├── agents/
│   ├── retrieval_agent.py      # RAG lookup against the Chroma store
│   └── orchestrator.py         # LangGraph graph: router → [tool_use?] → report → notify
├── api/
│   └── main.py                  # FastAPI wrapper: POST /run-triage
├── data/
│   ├── tickets.json             # sample input tickets
│   ├── tasks.json               # created by task_server.py at runtime
│   └── slack_log.json           # created by notify_server.py at runtime
└── requirements.txt
```

## Notes / things to extend further

- **Swap the model**: defaults to a local, free open-source model
  (Llama 3.1) via Ollama — no API key needed. Swap to a frontier model by
  changing `llm`/`llm_json` in `agents/orchestrator.py` to
  `ChatAnthropic(model="claude-sonnet-4-6")` (needs `langchain-anthropic`
  + `ANTHROPIC_API_KEY`), useful for comparing frontier-vs-open-source
  accuracy and latency on the same task — exactly the tradeoff the JD's
  "frontier and open-source models" line is testing for.
- **Make notify conditional too**: currently every run posts to Slack;
  you could branch so only `high`-priority batches trigger a notification.
- **Real integrations**: replace the JSON-file bodies of `create_task`
  and `post_message` with real Notion/Trello and Slack SDK calls — the
  tool signatures agents call don't need to change at all.

## Troubleshooting local models

- If `router_node` throws a JSON parse error, your model may be ignoring
  `format="json"` mode (smaller/older models are less reliable at this).
  Try a larger model (`llama3.1:8b` or bigger) or add a stricter
  few-shot example to `ROUTER_PROMPT`.
- If everything hangs, check `ollama serve` is actually running and
  `OLLAMA_BASE_URL` in `.env` matches it (default `http://localhost:11434`).
- First run of any prompt is slow while Ollama loads the model into
  memory — subsequent calls in the same session are much faster.

## Honesty note for résumé use

This was built and should be run and understood before being described
as a completed project — walk through `agents/orchestrator.py` end to
end so you can explain the router → tool_use → report flow and the MCP
tool-calling mechanics in an interview without hesitation.
