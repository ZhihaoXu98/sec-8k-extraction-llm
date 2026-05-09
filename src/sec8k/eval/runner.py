"""Eval orchestrator.

Loads an eval dataset, hits a configured model endpoint with bounded concurrency,
persists per-example artefacts (input, raw output, parsed prediction, judge verdict),
and writes a timestamped report under ``eval_results/``.
"""
