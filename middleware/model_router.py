"""Model routing for this agent, powered by TypeSafe classification.

`ModelRouterMiddleware` asks TypeSafe one `Choice` question about the latest
human message before the run starts, then pins every model call in that run to
the route it picked. It fails closed: a classification error, a missing human
message, or a route TypeSafe invented all raise instead of quietly falling back
to another model.

The class is experimental and its API may change without notice.
"""

from __future__ import annotations

from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)

# Every route resolves through the LangSmith LLM gateway, so they authenticate
# with LANGSMITH_GATEWAY_API_KEY (falling back to LANGSMITH_API_KEY) like the
# agent's own model does, and need no separate provider keys.
FAST_MODEL = "langsmith:openai/gpt-5-mini"
POWERFUL_MODEL = "langsmith:anthropic/claude-sonnet-4-6"
SUPER_POWERFUL_MODEL = "langsmith:anthropic/claude-opus-5"


def model_router_middleware() -> ModelRouterMiddleware:
    """Route each run to the cheapest model that can handle its task."""
    return ModelRouterMiddleware(
        choices={
            "fast": ModelChoice(
                model=FAST_MODEL,
                criteria=(
                    "Requests answerable in one step: lookups, short factual "
                    "answers, reformatting, and summaries of material the user "
                    "supplied. Route here when the work itself is trivial, not "
                    "when the request merely arrives with its own context."
                ),
            ),
            "powerful": ModelChoice(
                model=POWERFUL_MODEL,
                criteria=(
                    "Ordinary multi-step work: research across a few sources, "
                    "analysis that weighs known trade-offs, code the agent has "
                    "to reason about, and plans carried across several tool "
                    "calls. The path to an answer is clear even though the work "
                    "takes several steps. A request can belong here even when "
                    "it arrives complete in a single message."
                ),
            ),
            "super_powerful": ModelChoice(
                model=SUPER_POWERFUL_MODEL,
                criteria=(
                    "The hardest tasks, where a weaker model would plausibly "
                    "get it wrong: novel or open-ended problems with no "
                    "established approach, subtle correctness or safety "
                    "requirements, conflicting evidence that has to be "
                    "reconciled, long multi-stage plans whose early choices "
                    "constrain later ones, or work the user has flagged as "
                    "high-stakes."
                ),
            ),
        },
        instructions=(
            "Choose the least costly route that can still complete the task "
            "well. Judge by how much reasoning the task needs, not by whether "
            "the request carries its own context — one self-contained message "
            "can still describe hours of hard work. Choose `fast` only when "
            "the answer is one step. Choose `powerful` when the task takes "
            "several steps but the approach is clear. Reserve `super_powerful` "
            "for tasks that genuinely need expert-level reasoning — escalate "
            "on difficulty and stakes, not on length, topic, or the user's "
            "tone."
        ),
    )
