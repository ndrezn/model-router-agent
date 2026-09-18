from managed_deepagents import define_deep_agent

from middleware.auto_mode import auto_mode_middleware
from middleware.model_router import FAST_MODEL, model_router_middleware
from tools.fetch_url import fetch_url

# `define_deep_agent` returns a managed deployment spec, not a compiled graph.
# The managed backend, store, and checkpointer are injected at deploy time, and
# the CLI embeds the system prompt from instructions.md — so none are set here.
agent = define_deep_agent(
    name="model-router-agent",
    # The router overrides the model on every routed call, so this is only the
    # base the agent is compiled with.
    model=FAST_MODEL,
    # Middleware order matters and MDA never infers it. The router picks the
    # model for the run; auto mode gates tool calls rather than the model, so it
    # sits outside that ordering.
    middleware=[
        model_router_middleware(),
        auto_mode_middleware(),
    ],
    # Search comes from the MCP servers in tools/mcp.py, which the CLI picks up
    # on its own. Only hand-written tools are listed here.
    tools=[fetch_url],
)
