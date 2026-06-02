"""Query alias expansion rules for code search."""

from __future__ import annotations

_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "arg": ("argument",),
    "args": ("argument", "arguments"),
    "argument": ("arg", "args"),
    "arguments": ("arg", "args", "argument"),
    "auth": ("authentication", "authorization"),
    "authentication": ("auth",),
    "authorization": ("auth",),
    "cfg": ("config", "configuration"),
    "config": ("cfg", "configuration"),
    "configuration": ("cfg", "config"),
    "context": ("ctx",),
    "ctx": ("context",),
    "database": ("db",),
    "db": ("database",),
    "env": ("environment",),
    "environment": ("env",),
    "err": ("error",),
    "error": ("err",),
    "errors": ("err", "error"),
    "message": ("msg",),
    "messages": ("msg", "message"),
    "msg": ("message",),
    "opt": ("option",),
    "opts": ("option", "options"),
    "option": ("opt", "opts"),
    "options": ("opt", "opts", "option"),
    "param": ("parameter",),
    "params": ("parameter", "parameters"),
    "parameter": ("param", "params"),
    "parameters": ("param", "params", "parameter"),
    "req": ("request",),
    "request": ("req",),
    "requests": ("req", "request"),
    "res": ("response",),
    "resp": ("response",),
    "response": ("resp", "res"),
    "responses": ("resp", "res", "response"),
}


def query_aliases(term: str) -> tuple[str, ...]:
    """Return code-search aliases for one normalized query term."""
    return _QUERY_ALIASES.get(term.lower(), ())
