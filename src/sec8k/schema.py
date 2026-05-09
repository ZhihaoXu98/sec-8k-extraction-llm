"""Pydantic v2 models for 8-K Items, extracted fields, and FastAPI request/response payloads.

Single source of truth for the extraction schema. Any change here is a contract change for
training data, the inference server, the eval harness, and downstream consumers, so it
must land in its own PR with a migration note.
"""
