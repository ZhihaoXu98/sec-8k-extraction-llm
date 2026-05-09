"""JSON-schema-guided decoding via vLLM structured outputs (xgrammar backend).

Builds the ``response_format`` / guided-decoding payload from the Pydantic models in
:mod:`sec8k.schema`, so model output is constrained to a valid extraction record.
"""
