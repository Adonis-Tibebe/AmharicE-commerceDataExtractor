# Tests

This directory contains all tests for the project.

- `unit/` - Unit tests for individual functions and modules (fast, isolated)
- `integration/` - Integration tests for workflows and external dependencies (may require credentials)

## Running Tests

To run all unit tests:
```bash
pytest tests/unit
```

To run all integration tests (locally, with credentials):
```bash
pytest tests/integration
```

**Note:**  
- Integration tests may require a valid `.env` file and Telegram session.
- In this project it is run locally to avoide pushing session data and credentials to git.