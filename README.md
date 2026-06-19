# public-data-quality-be

Backend service for public data quality analysis and validation workflows.

## Structure

- `agents/`: domain-specific workflow and validation agents
- `core/`: shared core logic
- `data/`: local sample and reference datasets
- `graph.py`, `service.py`, `web.py`: application entry modules

## Notes

- `.env` files are ignored by Git.
- Python cache and common local artifacts are ignored via `.gitignore`.
- LLM integration uses Ollama chat by default:
  `http://127.0.0.1:11434/api/chat`
- Start Ollama and pull the models before enabling LLM agents:
  `ollama pull gemma4:e2b`
  `ollama pull gemma4:e4b`
- The default strategy uses `gemma4:e2b` for fast routing, then uses `gemma4:e4b` for strong/precision validation.
- Override it with `OLLAMA_FAST_MODEL`/`OLLAMA_STRONG_MODEL` or the web form's model values.
