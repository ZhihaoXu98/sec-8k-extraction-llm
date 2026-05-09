"""Langfuse client init and span decorators.

Tags every trace with the prompt version (from :mod:`sec8k.prompts`) and the model
revision; captures latency and token-cost metadata so we can compare runs across
experiments and serving builds.
"""
