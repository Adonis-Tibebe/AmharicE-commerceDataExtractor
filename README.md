# Amharic E-commerce Data Extractor

A Python project for scraping, preprocessing, and labeling Amharic e-commerce data from Telegram channels for NLP tasks.

## Features

- Telegram channel scraping
- Data cleaning and normalization
- Hashtag extraction
- Data labeling workflow for NER
- Model FineTuning
- Model Comparison and Interpretation
- Model Interpretation (SHAP and LIME)
- Vendor ScoreBoard Generation for business discision

## Installation

```bash
git clone https://github.com/yourusername/amharic-ecommerce-extractor.git
cd amharic-ecommerce-extractor
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

See the `notebooks/` directory for example workflows and the `src/` directory for core modules.

## Project Structure

- `src/` - Source code modules
- `notebooks/` - Jupyter notebooks for data Ingestion, Normalization,Preprocessing and labeling, Model finetuning, model comparison, Model interpretaton, verndor scoreboard creation
- `data/` - Raw and processed data
- `tests/` - Unit and integration tests
- `config/` - Configuration files