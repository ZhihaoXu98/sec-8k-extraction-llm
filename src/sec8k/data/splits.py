"""Train/val/test splitter with leakage guards.

Splits respect filer-CIK and time-window boundaries so a filer never appears in two
splits and the test set strictly post-dates the training set.
"""
