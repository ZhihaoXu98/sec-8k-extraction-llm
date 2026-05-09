"""DPO entry point.

Built on TRL ``DPOTrainer``; trains over the SFT adapter using preference pairs produced
by :mod:`sec8k.data` (judge scores or rule perturbations). Reads ``configs/dpo.yaml``.
"""
