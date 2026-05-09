"""Custom data collator for prompt-completion training.

Masks the prompt portion in the labels so the loss is computed only on the completion
tokens; required to keep extraction quality stable when prompts are long relative to
completions.
"""
