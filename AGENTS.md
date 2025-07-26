# AGENTS.md

## Build, Lint, and Test Commands

- **Install dependencies:** `uv sync`
- **Activate environment:** `source .venv/bin/activate`
- **Run all tests:** `uv run pytest`
- **Run a single test:** `uv run pytest path/to/test_file.py::test_function`
- **Format code:** `uv run black .`
- **Sort imports:** `uv run isort .`

## Code Style Guidelines

- **Imports:** Use absolute imports. Group: standard library, third-party, local. Use `isort` for sorting.
- **Formatting:** Use `black` for consistent formatting (PEP8 compliant).
- **Types:** Prefer explicit type hints for function signatures and variables.
- **Naming:** Use `snake_case` for functions/variables, `PascalCase` for classes, and UPPER_CASE for constants.
- **Error Handling:** Use exceptions for error cases. Avoid bare `except`; catch specific exceptions.
- **Project Structure:** Place main code in `main.py`. Add more modules in `src/` if needed. Tests should go in `tests/`.
- **Comments:** Write clear, concise comments. Use docstrings for all public functions/classes.
- **Version Control:** Respect `.gitignore` for Python, build, and environment files.
- **Python Version:** Requires Python 3.11+.
- **Dependencies:** List all dependencies in `pyproject.toml`.
- **Test Coverage:** Prefer `pytest` for all tests. Name test functions as `test_*`.
- **Documentation:** Add docs in `docs/` as needed.
