"""Shared pytest fixtures.

Provides a tiny EDGAR sample, a stubbed vLLM client, and an isolated temporary config
so unit tests stay hermetic and never hit the network or the GPU.
"""
