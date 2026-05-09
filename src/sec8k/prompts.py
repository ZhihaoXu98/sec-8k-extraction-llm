"""Versioned system and user prompt templates plus render helpers.

Every prompt carries an explicit version string that is included in Langfuse traces and
training-data records, so we can reproduce or roll back model behaviour bound to a
specific prompt revision.
"""
