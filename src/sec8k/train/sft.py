"""QLoRA SFT entry point.

Built on TRL ``SFTTrainer`` + PEFT + ``accelerate``; reads ``configs/sft.yaml``. The base
model is loaded in 4-bit via ``bitsandbytes`` so the run fits in 12 GB on a 4070.
"""
