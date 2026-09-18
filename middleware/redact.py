"""Redact sensitive values before the router sends the task off-deployment.

`ModelRouterMiddleware` classifies the latest human message with the TypeSafe
API, so the user's text leaves this deployment before the first model call.
That classification happens in `before_agent`, and the agent schedules every
`before_agent` hook ahead of every `before_model` hook — so `PIIMiddleware`
redacts too late to cover it, wherever it sits in the middleware list. This
middleware redacts in `before_agent` instead and is listed ahead of the router,
so the classifier only ever sees redacted text.
"""

from __future__ import annotations

import operator

from langchain.agents.middleware import AgentState, before_agent
from langchain.agents.middleware.pii import (
    PIIMatch,
    detect_credit_card,
    detect_email,
    detect_ip,
)
from langchain_core.messages import AnyMessage, HumanMessage
from managed_deepagents import ManagedDeepAgentRuntime

_DETECTORS = (detect_email, detect_credit_card, detect_ip)


def _redact(text: str) -> str:
    """Replace every detected value with a `[REDACTED_TYPE]` placeholder."""
    matches: list[PIIMatch] = [
        match for detect in _DETECTORS for match in detect(text)
    ]
    # Rewrite back to front so each replacement leaves earlier offsets intact.
    redacted = text
    for match in sorted(matches, key=operator.itemgetter("start"), reverse=True):
        placeholder = f"[REDACTED_{match['type'].upper()}]"
        redacted = redacted[: match["start"]] + placeholder + redacted[match["end"] :]
    return redacted


def redact_pii_middleware():
    """Redact human messages in agent state before the router reads them."""

    @before_agent
    def redact_pii(
        state: AgentState,
        runtime: ManagedDeepAgentRuntime,
    ) -> dict | None:
        # Returning a message that carries an existing message's id replaces it
        # in state rather than appending. `add_messages` has already assigned an
        # id to everything in state, so no message here can be missing one.
        replacements: list[AnyMessage] = []
        for message in state["messages"]:
            if not isinstance(message, HumanMessage):
                continue
            if not isinstance(message.content, str):
                continue
            redacted = _redact(message.content)
            if redacted != message.content:
                replacements.append(message.model_copy(update={"content": redacted}))

        if not replacements:
            return None
        return {"messages": replacements}

    return redact_pii
