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
