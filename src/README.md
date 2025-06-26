# Source Code (`src/`)

This directory contains the main source code for the project.

## Structure

- `services/`  
  Contains service modules, such as:
  - `telegram_scrapper.py`: Module for establishing a connection with the Telegram server and scraping data from channels.

- `utils/`  
  Utility functions for data processing, including:
  - `utils.py`: Functions for data cleaning, normalization, tokenization, and other helper routines.

- `models/`  
  Model-related code, including:
  - `ner_model.py`: Functions for loading, saving, and running NER models and pipelines.

- `core/`  
  Core training and evaluation logic, including:
  - `train.py`: Functions for fine-tuning and evaluating NER models.

## Notes

- Each module is documented with inline comments and docstrings.
- Designed for modularity and reuse