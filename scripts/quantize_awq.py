"""CLI: merge SFT/DPO LoRA into the base model, then AWQ INT4 quantize for vLLM.

Output goes to ``$MODEL_AWQ_DIR`` (default ``models/awq-int4/``, gitignored) ready to
be loaded by vLLM.
"""
