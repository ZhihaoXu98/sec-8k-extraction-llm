"""FastAPI app factory.

Lifespan hooks initialise and tear down the vLLM client; middleware registration
(logging, request id, Langfuse tracing) and route registration happen here.
"""
