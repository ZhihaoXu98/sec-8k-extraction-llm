# Config

Runtime configuration is layered, lowest-precedence first:

1. `config/default.yaml` (this directory) — committed defaults.
2. Environment variables — see `.env.example` at the repo root.
3. CLI flags — highest precedence.

Training-time configuration lives separately under `src/sec8k/train/configs/`
(`sft.yaml`, `dpo.yaml`) so it can be versioned alongside code that consumes it.
