# Home Credit Defaulter Check

This project builds an engineered Home Credit dataset and provides a Streamlit interface for estimating loan-default risk with a trained CatBoost classifier.

## What is included

- `data_merger.py` - reads the original Home Credit CSV tables, validates key uniqueness, creates aggregate and engineered features, and writes the combined dataset.
- `Notebooks/base.ipynb` - exploratory analysis and model-development notebook.
- `Notebooks/catboost_model.cbm` - trained CatBoost model used by the application.
- `streamlit app/main.py` - interactive risk-assessment application.
- `data.zip` - compressed source dataset.
- `output.zip` - compressed generated output data, including `train_main.csv`.
- `requirements.txt` - Python dependencies.

The uncompressed `data/` and `output/` directories are intentionally excluded from Git. Extract the ZIP archives locally when running the pipeline or application.

## Setup

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Extract the archives in the project root so the following paths exist:

```text
data/
output/train_main.csv
Notebooks/catboost_model.cbm
```

## Build the engineered dataset

The merger expects the original Home Credit CSV files inside `data/`. Use an explicit output directory so the command works on any machine:

```bash
python data_merger.py --data-dir data --out-dir output
```

The pipeline combines application, bureau, previous-application, POS cash, installment, and credit-card data. It validates identifiers and merges, handles sentinel values, creates numeric and categorical aggregates, and writes:

```text
output/train_main.csv
```

## Run the Streamlit application

After extracting `output.zip` (or generating `output/train_main.csv`), start the app from the project root:

```bash
streamlit run "streamlit app/main.py"
```

The sidebar supports two modes:

- **Existing applicant** - looks up an applicant ID in the engineered dataset.
- **Quick assessment** - starts from a baseline record and lets you enter a new application profile.

The app displays an estimated default probability, a risk band, important model features, and basic loan context. This is decision-support software, not an automatic approval or rejection system; use it with affordability checks, policy rules, and human review.

## Data and model paths

The application uses these defaults:

```text
Notebooks/catboost_model.cbm
output/train_main.csv
```

To use different locations, set `MODEL_PATH` and `TRAIN_DATA_PATH` before starting Streamlit. For example, in PowerShell:

```powershell
$env:MODEL_PATH = "C:\path\to\catboost_model.cbm"
$env:TRAIN_DATA_PATH = "C:\path\to\train_main.csv"
streamlit run "streamlit app/main.py"
```

## Repository notes

Large ZIP archives are stored with Git LFS. The local virtual environment, IDE metadata, uncompressed data, and generated output directories are ignored by Git.
