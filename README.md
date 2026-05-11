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
- LLM integration uses the OpenAI GPT API by default:
  `https://api.openai.com/v1/chat/completions`
- Set `OPENAI_API_KEY` in `.env` before enabling LLM agents.
- The default model is `gpt-4o-mini`. Override it with `LLM_MODEL` or the web form's `llm_model` value.
