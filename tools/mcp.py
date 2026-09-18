"""MCP servers this agent loads tools from.

Both are credential-free, so neither declares a `connection`: Exa is a LangSmith
managed tool server, and the LangChain docs server is public. `mda connections
list` shows no search credential in this workspace, which is why neither is a
Tavily-style key.
"""

from managed_deepagents import define_mcp

mcp = define_mcp(
    servers={
        # Web search. `include_tools` is omitted on purpose — an unrecognized
        # raw tool name silently yields no tool, and this server's names are
        # not documented in the SDK, so every tool it loads is exposed.
        "exa": {
            "transport": "http",
            "url": "https://api.smith.langchain.com/v1/managed-tools/servers/exa/mcp",
        },
        # Docs search, worth having given this project is built on unreleased
        # LangChain middleware. Selected by raw MCP tool name; MDA adds the
        # server-name prefix at runtime.
        "langchainDocs": {
            "transport": "http",
            "url": "https://docs.langchain.com/mcp",
            "include_tools": ["search_docs_by_lang_chain"],
        },
    },
)
