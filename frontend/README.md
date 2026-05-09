# Streamlit demo UI

Install the UI extra and launch the app:

```bash
uv pip install -e ".[dev,ui]"
uv run streamlit run frontend/app.py
```

The app expects the FastAPI service from `src/sec8k/serve/` to be running and reachable at the address configured in `.env` (see `.env.example`).
