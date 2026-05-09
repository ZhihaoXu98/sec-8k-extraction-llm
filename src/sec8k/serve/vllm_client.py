"""Async OpenAI-compatible client targeting the local vLLM server.

Wraps the OpenAI SDK with a retry/timeout policy tuned for local serving and exposes a
narrow surface (``complete``, ``stream``) so the rest of the codebase doesn't depend
on the vendor SDK directly.
"""
