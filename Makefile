PYTHON_VERSION := 3.11
VENV := .venv

.PHONY: venv install install-data install-train install-serve install-ui \
        test test-cov lint format format-check typecheck check \
        precommit-install ingest build-sft build-dpo train-sft train-dpo \
        quantize eval serve ui clean

venv:
	uv venv --python $(PYTHON_VERSION)

install: venv
	uv pip install -e ".[dev]"

install-data:
	uv pip install -e ".[dev,data]"

install-train:
	uv pip install -e ".[dev,data,train]"

install-serve:
	uv pip install -e ".[dev,data,serve,quant]"

install-ui:
	uv pip install -e ".[dev,ui]"

test:
	uv run pytest

test-cov:
	uv run pytest --cov=sec8k --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

typecheck:
	uv run mypy src tests

check: lint format-check typecheck test

precommit-install:
	uv run pre-commit install

ingest:
	uv run python scripts/ingest_edgar.py

build-sft:
	uv run python scripts/build_sft_dataset.py

build-dpo:
	uv run python scripts/build_dpo_dataset.py

train-sft:
	bash scripts/train_sft.sh

train-dpo:
	bash scripts/train_dpo.sh

quantize:
	uv run python scripts/quantize_awq.py

eval:
	uv run python scripts/run_eval.py

serve:
	bash scripts/serve_local.sh

ui:
	uv run streamlit run frontend/app.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .ruff_cache .mypy_cache .pytest_cache dist build *.egg-info
