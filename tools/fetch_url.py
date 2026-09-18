"""Fetch a specific web page so the agent can read and cite it.

A URL the model supplies is untrusted input, so this validates before it
connects: http(s) only, and every address the host resolves to must be a global
one. Redirects are deliberately not followed, each hop has to come back through
this tool, and the response is size-capped.

One residual risk this does not close: DNS rebinding. The host is resolved for
the check and resolved again by the connect, and a record that changes between
the two would slip past. Closing it means pinning the checked address for the
connection itself.
"""

from __future__ import annotations

import asyncio
import ipaddress
from urllib.parse import urlparse

import httpx
from langchain.tools import tool

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_BYTES = 200_000
_TIMEOUT_SECONDS = 30.0


async def _assert_public_host(host: str) -> None:
    """Reject a host resolving anywhere outside the public internet.

    Args:
        host: Hostname parsed out of the requested URL.

    Raises:
        ValueError: If the host does not resolve, or any record is non-global.
    """
    loop = asyncio.get_running_loop()
    try:
        records = await loop.getaddrinfo(host, None)
    except OSError as error:
        msg = f"Could not resolve {host!r}."
        raise ValueError(msg) from error

    # Allowlist rather than deny list: every record has to be global. `is_global`
    # already excludes private, loopback, link-local, reserved, and multicast
    # ranges, so a newly assigned special-use range needs no change here.
    for record in records:
        # Strip any IPv6 scope id (`fe80::1%eth0`), which `ip_address` rejects.
        literal = record[4][0].split("%")[0]
        if not ipaddress.ip_address(literal).is_global:
            msg = f"Refusing to fetch {host!r}: it resolves to a non-public address."
            raise ValueError(msg)


@tool(parse_docstring=True)
async def fetch_url(url: str) -> str:
    """Fetch the text of a public web page.

    Redirects are not followed. When the page redirects, the new location is
    returned so it can be fetched in a second call.

    Args:
        url: Absolute http or https URL to fetch.
    """
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        msg = f"Only http and https URLs are supported, got {parsed.scheme!r}."
        raise ValueError(msg)
    if not parsed.hostname:
        msg = "URL must include a hostname."
        raise ValueError(msg)

    await _assert_public_host(parsed.hostname)

    body = bytearray()
    async with httpx.AsyncClient(
        timeout=_TIMEOUT_SECONDS,
        # Following redirects here would let a public URL bounce to a private
        # one after the check above, so each hop is handed back to the model.
        follow_redirects=False,
    ) as client:
        async with client.stream("GET", url) as response:
            if response.is_redirect:
                location = response.headers.get("location", "(none)")
                return (
                    f"HTTP {response.status_code} redirect to {location}. "
                    "Call fetch_url again with that URL to follow it."
                )
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) >= _MAX_BYTES:
                    break

    text = bytes(body[:_MAX_BYTES]).decode("utf-8", errors="replace")
    if len(body) >= _MAX_BYTES:
        text += f"\n\n[truncated at {_MAX_BYTES} bytes]"
    return text
