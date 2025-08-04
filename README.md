# Legal Sentences Filtering Pipeline (Faiss & Fuzzy)

## Overview
Two complementary notebooks for filtering and augmenting legal sentence data:

- **Fuzzy**: Lightweight, regex-based topic filtering on textual columns (e.g., “Tema - subtema”, “resuelve”, “sintesis”) that displays a word cloud for quick understanding and summarizes sentences and details via parallelized API calls. Call intervals and token usage are optimized to avoid hitting rate or token limits, which is critical if the service is paid.
  
- **Faiss**: vector index search instead of the old fuzzy.

Both expect an input Excel of past legal sentences and output filtered subsets for further review or LLM processing.

## Requirements

Install the dependencies (see `requirements.txt`), e.g.:

```sh
pip install -r requirements.txt
```

or inside the jupyter notebook cell via: !pip install -r requirements.txt

Minimal required packages include:
- pandas
- numpy
- matplotlib
- spacy
- sentence-transformers
- faiss-cpu (or `faiss-gpu` if using GPU)
- groq
- python-dotenv
- wordcloud

## Setup

1. **Input data**  
   Place the source Excel at `./Datos/sentencias_pasadas.xlsx`. It must contain at least the columns:
   - `Tipo` (will be dropped)
   - `Tema - subtema`
   - `resuelve`
   - `sintesis`

2. **Environment variables**  
   Create a `.env` file in the working directory with API keys used by the augmentation logic. Example:
   ```env
   llama3_70b_8192=...
   gemma2_9b_it=...
   llama_instant=...
   llama3_8b=...
   ```
   Then the notebooks load them via `python-dotenv` and fetch with `os.getenv`.

## Usage

### Common initial steps (both notebooks)

```python
import pandas as pd
from pathlib import Path

# Load and pre-process
legal_data_path = r".\Datos\sentencias_pasadas.xlsx"
legal_data = pd.read_excel(legal_data_path).drop(columns="Tipo")
```

### Fuzzy notebook
 
  - Provides `filter_topic(df, topics, cols=None)` which is a **strict regex-based** filter.
  - `topics`: list of base terms (e.g., `["escolar"]`)
  - Search in default columns: `["Tema - subtema", "resuelve", "sintesis"]`   
  - LLM augmentation (e.g., placeholder `LLM_aumentation(...)`).
  - Visualizations like word clouds.
  - Logging of retained/excluded counts.


Example:
```python
TOPIC = ["escolar"]
filtered, len_df, len_filtered = filter_topic(legal_data, TOPIC)
```

- You can chain filters by applying new topics to the `filtered` result and export:
  ```python
  filtered.to_csv("legal_data_filtrado.csv", index=False, encoding="utf-8-sig")
  ```

### Faiss notebook

- Starts similarly by loading `legal_data` and dropping `Tipo`.
- Reuses or redefines filtering logic for specific topics like `"red social"`.

Example:
```python
TOPIC = ["PIAR", "Plan Individualizado de Ajustes Razonables"]
filtered_PIAR, _, len_PIAR = filter_topic(legal_data, TOPIC)
print(f"Retained: {len_PIAR}")
```

## Output

- Filtered subsets are stored as DataFrames like `filtered_acoso_escolar`, `filtered_PIAR`, etc.
- Notebooks suggest saving to CSV with UTF-8-sig:
  ```python
  filtered_acoso_escolar_con_LLMs.to_csv("filtered_acoso_escolar_con_LLMs.csv", encoding="utf-8-sig")
  ```

## Word Cloud

![Word cloud of key terms](Datos/word_cloud.PGN)

*Figure: Word cloud summarizing the most frequent and semantically related terms, providing a quick topical overview.*


## Tips

- The regex filter lowercases and strips accents; normalize topics accordingly.
- Validate presence of environment variables before use.

## Example Workflow Summary

1. Populate `.env` with required API keys.
2. Ensure `sentencias_pasadas.xlsx` is present.
3. Run `Consultoría 2_Fuzzy.ipynb` to get base filtered data aumented via LLM.
4. Use `Consultoría 2_Faiss.ipynb` for a modern vector index search alternative.
5. Export final results.
