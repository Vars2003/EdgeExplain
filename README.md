# EdgeExplain

EdgeExplain is a production-grade, 100% offline Edge AI Data Intelligence Platform and Data Scientist application. It ingests, profiles, and reasons about datasets, extracts natural-language insights, analyzes feature dependencies, highlights structural risks, and provides preprocessing and machine learning model recommendations locally on the user's computer without using any external cloud APIs.

## Key Features

1. **Pure Data Sovereignty (Offline First)**: Zero network calls. Your business files and database schemas are processed entirely in-memory on your local device.
2. **Semantic Feature Parser**: Identifies variables beyond raw data types (e.g. continuous/discrete numeric, nominal/ordinal categorical, identifiers, coordinates, binary, datetime, text).
3. **Evidence-Based Reasoning Engine**: Every classification and preprocessing suggestion includes a confidence score, structural evidence, and a logical justification trace.
4. **Generalized Multi-Type Dependency Network**: Employs Pearson/Spearman for numeric relationships, Chi-Square/Cramer's V for categorical relationships, ANOVA for numeric-categorical relations, and Mutual Information for non-linear mixed associations.
5. **Interactive Interpretability**: Users can select target features to fit local Decision Tree models offline, extracting and plotting feature importances.
6. **Optional ydata-profiling Exports**: Generates deep exploratory reports as standalone HTML files on demand.
7. **Future-Proof AI Hooks**: Contains placeholder packages and prompts preparing the app for local offline Generative AI (LLM GGUF model files) integrations.

## Project Structure

```
edgeai/
├── app.py                  # Main entry point and page router
├── README.md               # Setup and documentation
├── requirements.txt        # Package dependencies
├── .gitignore              # Git exclusion configurations
├── config/
│   └── settings.py         # Global limits, styles, domain vocabularies
├── core/
│   # --- Analytics Layer ---
│   ├── loader.py           # Auto-format detection and reading
│   ├── profiler.py         # Basic metrics, Quality, and ML Readiness score
│   ├── cleaner.py          # Duplicates, outlier filtering, imputations
│   ├── statistics.py       # Math functions (Mean, Median, Skewness, Kurtosis)
│   ├── dependency.py       # Generalized associations and network builders
│   ├── visualizer.py       # Plotly histogram, scatter, bar, box, and graphs
│   # --- AI Layer ---
│   ├── intelligence.py     # Semantic column classification & domain inference
│   ├── reasoning.py        # Logic justification engine
│   ├── insights.py         # Natural-language findings compiler
│   ├── recommendation.py   # Preprocessing recommendation engine
│   ├── algorithm_selector.py # ML model ranking engine
│   └── explain.py          # Custom feature importance and SHAP placeholders
├── utils/
│   ├── logger.py           # Structured system logging
│   └── helpers.py          # Formatting utilities and css injections
├── knowledge/              # Context base schemas for local LLM usage
│   ├── preprocessing.json
│   ├── algorithms.json
│   ├── statistics.json
│   └── feature_types.json
├── ai/                     # Local LLM conversation hooks (Phase 2)
│   ├── llm.py
│   ├── chat.py
│   ├── explainer.py
│   └── prompts.py
├── pages/                  # Streamlit dashboard multi-page sheets
│   ├── 1_Home.py
│   ├── 2_Upload.py
│   └── 3_Dataset_Overview.py
├── assets/                 # Image and logo placeholders
├── data/                   # Saved datasets (auto-populated)
├── reports/                # Exported reports (auto-populated)
└── logs/                   # System logging files (auto-populated)
```

## Setup & Running Instructions

### Prerequisites
- Python 3.12+ installed.
- Pip (Python Package Installer).

### Installation

1. Navigate to the project root directory:
   ```bash
   cd C:\Users\varsh\OneDrive\Desktop\edgeai
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Execution

Launch the Streamlit dashboard by running:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser at `http://localhost:8501`.
