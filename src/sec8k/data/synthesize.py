"""Teacher-model synthesis path.

Generates synthetic ``(prompt, completion)`` pairs from a stronger reference model to
bootstrap SFT data when human-labelled examples are scarce; outputs are tagged with the
teacher model id and prompt version for provenance.
"""
