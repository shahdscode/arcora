# Arcora

An end-to-end NBA basketball analytics and game prediction platform powered by machine learning. Arcora forecasts game scores and winning probabilities using historical game data, advanced statistical modeling, explainable AI, and interactive analytics.

---

## Project Structure

```text
arcora/
├── data/
│   ├── raw/                  # Immutable raw NBA datasets (Nathan Lauga Kaggle dataset)
│   └── processed/            # Standardized, validated canonical Parquet datasets
├── docs/                     # Dataset source, coverage, audit, and schema documentation
├── logs/                     # Pipeline execution and validation logs
├── notebooks/                # Exploratory data analysis (EDA) Jupyter notebooks
├── reports/                  # Data quality reports, EDA findings, and visualizations
│   └── figures/              # Generated statistical plots and charts
├── scripts/                  # Standalone CLI scripts for acquisition, audit, ingestion, validation, and EDA
├── src/
│   └── arcora/               # Core Python package
│       ├── __init__.py
│       ├── cli.py            # Unified CLI entry point
│       ├── config.py         # Environment and path configuration
│       ├── data/             # Data acquisition, auditing, and canonical ingestion pipeline
│       ├── eda/              # Exploratory data analysis and visualization generators
│       ├── models/           # Canonical data model and schema definitions
│       └── validation/       # Automated data quality and integrity checks
├── tests/                    # Automated pytest suite
├── .env.example              # Template for environment variables
├── .gitignore                # Git exclusion rules
├── LICENSE                   # MIT License
├── pyproject.toml            # Project metadata, dependencies, and tool configurations
├── requirements.txt          # Pinned/minimum dependency list
└── README.md                 # Project documentation and setup guide
```

---

## Prerequisites

- **Python**: 3.12 or higher
- **Git**: For version control

---

## Installation & Setup

### 1. Clone the repository

```bash
git clone git@github.com:shahdscode/arcora.git
cd arcora
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv
# Or using your default Python 3.12+ interpreter:
python3 -m venv .venv

source .venv/bin/activate
```

### 3. Install dependencies

Install the package in editable mode along with development dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

### 4. Configure environment variables

Copy the example environment file and customize if needed:

```bash
cp .env.example .env
```

---

## Usage & Week 1 Workflows

Arcora provides both a unified CLI (`arcora` or `python -m arcora.cli`) and standalone scripts under `scripts/`:

### 1. Acquire & Verify Historical NBA Dataset
```bash
python scripts/acquire_data.py
```
Downloads (if not already present) and verifies the Nathan Lauga NBA Games dataset under `data/raw/`, generating `data/raw/manifest.json` and coverage summaries.

### 2. Audit Raw Dataset Schemas & Quality
```bash
python scripts/audit_data.py
```
Inspects all raw CSV files (`games.csv`, `games_details.csv`, `players.csv`, `ranking.csv`, `teams.csv`), quantifying missing values, duplicates, foreign-key relationships, and anomalies.

### 3. Run Canonical Data Ingestion Pipeline
```bash
python scripts/ingest_data.py
```
Transforms raw CSV files into standardized, deduplicated canonical Parquet datasets (`teams.parquet`, `players.parquet`, `games.parquet`, `player_game_stats.parquet`, `standings.parquet`) under `data/processed/`.

### 4. Execute Automated Data Quality Checks
```bash
python scripts/validate_data.py
```
Runs automated integrity, schema, type, score, and foreign-key checks on the processed canonical datasets and writes `reports/data_quality_report.md` and `reports/data_quality_summary.json`.

### 5. Run Exploratory Data Analysis (EDA)
```bash
python scripts/run_eda.py
```
Computes historical distributions, home-court advantage metrics, seasonal scoring trends, and anomaly analyses, generating publication-ready charts in `reports/figures/` and executing `notebooks/01_nba_exploratory_data_analysis.ipynb`.

---

## Testing & Code Quality

Run the test suite with `pytest`:

```bash
pytest
```

Run linting and formatting checks with `ruff`:

```bash
ruff check src tests scripts
```

Run static type checking with `mypy`:

```bash
mypy src
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
