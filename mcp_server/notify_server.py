"""
Notify MCP Server
==================
A second MCP server, separate from the task board, to demonstrate
multi-platform integration (the JD explicitly calls out "integrate AI
systems across platforms, APIs, SaaS tools"). Exposes a single
`post_message` tool that simulates posting to a Slack channel by
appending to a local log file — swap the body of `post_message` for a
real `slack_sdk.WebClient.chat_postMessage` call and every agent that
calls this tool keeps working unchanged.

Run as an MCP server (stdio transport):
    it is spawned automatically by agents/orchestrator.py
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

NOTIFY_LOG_PATH = Path(os.environ.get("NOTIFY_LOG_PATH", "./data/slack_log.json"))

mcp = FastMCP("ops-copilot-notify")


def _load() -> list[dict]:
    if not NOTIFY_LOG_PATH.exists():
        NOTIFY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        NOTIFY_LOG_PATH.write_text("[]")
    return json.loads(NOTIFY_LOG_PATH.read_text())


def _save(entries: list[dict]) -> None:
    NOTIFY_LOG_PATH.write_text(json.dumps(entries, indent=2))


@mcp.tool()
def post_message(channel: str, message: str) -> dict:
    """Post a message to a team channel.

    In this local demo, "posting" means appending to a JSON log file so
    the whole pipeline runs with zero external accounts. Replace the
    body with slack_sdk.WebClient(token=...).chat_postMessage(...) to
    make this a real Slack integration without touching any calling code.

    Args:
        channel: Channel name, e.g. "#ops-status".
        message: The message body to post.
    """
    entries = _load()
    entry = {
        "channel": channel,
        "message": message,
        "posted_at": datetime.now(timezone.utc).isoformat(),
    }
    entries.append(entry)
    _save(entries)
    return entry


if __name__ == "__main__":
    mcp.run(transport="stdio")
