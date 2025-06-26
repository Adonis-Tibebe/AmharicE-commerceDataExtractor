# Jupyter Notebooks

This directory contains Jupyter notebooks for:

- Data Ingestion and Preprocessing
- Data Labeling Workflow
- Model Training and Evaluation
- Model Interpretation and Business Analytics

**Notebooks:**

- `Data_ingestin_and_preprocessing.ipynb`  
  Contains a reproducible workflow for ingesting data from Telegram using the python telethon package and core modules from `src/`, as well as preprocessing steps.

- `Data_labeling.ipynb`  
  Workflow for loading preprocessed data and saving text entities in JSON format for labeling through Label Studio.

- `Fine_tune_NER.ipynb`  
  Notebook for fine-tuning the NER model on labeled Amharic e-commerce data.

- `Model_comparison_and_interpretation.ipynb`  
  Compares different NER models and provides initial interpretation of their performance.

- `Model_Interpretation.ipynb`  
  Explains the predictions of the fine-tuned NER model using LIME and SHAP, and provides detailed interpretation and recommendations.

- `Vendor_score_board.ipynb`  
  Analyzes Telegram vendor channels, computes business metrics (activity, engagement, price), and generates a vendor lending score board for business