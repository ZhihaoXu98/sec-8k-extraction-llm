"""LLM-as-judge for free-text fields.

Carries an explicit rubric, calibrates against a small human-labelled subset, and uses
tie-breaking (e.g. dual-judge or self-consistency) to keep agreement high enough to
trust on aggregate.
"""
