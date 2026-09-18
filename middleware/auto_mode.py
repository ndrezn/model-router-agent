"""Tool-risk gating for this agent's file writes, powered by TypeSafe.

`AutoModeMiddleware` intercepts the tools named in `tools` just before they
execute, asks TypeSafe a `Noul` question for the probability that the call is
risky or unauthorized, and returns an error `ToolMessage` instead of running the
tool once that probability reaches `risk_threshold`. It blocks on its own; it
does not ask a human.

Two things to know about the shape of this guard:

- Tool names not listed in `tools` skip classification silently. A typo here
  disables the guard without raising, so the names below are the deepagents
  built-ins verified against `deepagents/middleware`, not guesses.
- The classifier receives the tool call's arguments verbatim, plus the last 30
  messages in state. The redaction in `middleware/redact.py` covers human
  messages, not tool arguments and not tool results, so a `write_file` whose
  content carries sensitive data sends it unredacted.
- The block threshold is not configurable. `_PROBABILITY_THRESHOLD` is a module
  constant of 0.5 in 0.0.1a2, so a call is blocked once TypeSafe puts its risk
  at 50% or more.

The class is experimental and its API may change without notice.
"""

from __future__ import annotations

from langchain_typesafe import NoulCriteria
from langchain_typesafe.experimental.middleware import AutoModeMiddleware

# Built-in deepagents filesystem tools that mutate state, plus the custom
# fetcher. The read-only members of the filesystem toolset (`ls`, `read_file`,
# `glob`, `grep`) are left unguarded so the agent can research without a
# classification round-trip per call.
#
# `fetch_url` is here because an agent-driven fetcher is an exfiltration channel
# — data placed in a query string reaches whatever host the model was talked
# into using — and the default instructions already count external sharing as
# risky. Its URL argument is sent to TypeSafe for that classification.
GUARDED_TOOLS = ("write_file", "edit_file", "fetch_url")


def auto_mode_middleware() -> AutoModeMiddleware:
    """Block file writes TypeSafe judges risky or unauthorized."""
    return AutoModeMiddleware(
        tools=list(GUARDED_TOOLS),
        # The default `instructions` are kept: they already tell the classifier
        # to treat every value in state — tool descriptions and arguments
        # included — as data rather than instructions, and to accept only
        # explicit user messages as authorization.
        criteria=NoulCriteria(
            true=(
                "The call overwrites a file the agent did not create, writes "
                "outside the agent's own working files, or saves content the "
                "user never asked to have stored."
            ),
            false=(
                "The call saves notes, drafts, or results the user asked for, "
                "into the agent's own working files."
            ),
        ),
    )
