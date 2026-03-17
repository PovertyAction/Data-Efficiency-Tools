# Data Efficiency Tools

Presentation demos for the **IPA Rwanda country office** showcasing modern data processing tools through practical benchmarks.

## Overview

Three self-contained demonstrations comparing traditional vs. modern approaches using a synthetic 500,000-row Rwanda household survey dataset:

1. **Parquet vs CSV** - File format comparison (write/read speed, storage size, query performance)
2. **Polars vs Pandas** - DataFrame library comparison (5 common operations)
3. **DuckDB vs PostgreSQL** - Database comparison (4 analytical queries + Parquet scanning)

All demos use the same reproducible synthetic dataset representing a typical IPA Rwanda household survey with districts, treatment arms, survey rounds, and income data.

## Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://github.com/astral-sh/uv) package manager
- PostgreSQL binaries (for demo3 only) - [Download here](https://www.postgresql.org/download/)

### Installation

```bash
# Install/update all dependencies
uv sync
```

### Running Demos

Open the Jupyter notebooks in VS Code or Jupyter Lab:

- `demo1_parquet_vs_csv.ipynb`
- `demo2_polars_vs_pandas.ipynb`
- `demo3_duckdb_vs_sql.ipynb`

## Demo Details

### Demo 1: Parquet vs CSV
Compares Parquet and CSV file formats across:
- Write performance
- Read performance
- File size efficiency
- Query performance (filtered reads)

### Demo 2: Polars vs Pandas
Benchmarks five common data operations:
- Filtering rows
- Grouping and aggregation
- Joining datasets
- Creating new columns
- Sorting

### Demo 3: DuckDB vs PostgreSQL
Compares embedded analytical database (DuckDB) vs. traditional database (PostgreSQL):
- Aggregation queries
- Filtering and grouping
- Complex joins
- Window functions
- Direct Parquet file scanning (DuckDB only)

## Dataset

All demos generate identical synthetic data with:
- **500,000 rows**
- **10 Rwandan districts** across 5 provinces (Kigali, Eastern, Western, Northern, Southern)
- **3 treatment arms**: control, treatment_A, treatment_B
- **3 survey rounds**: baseline, midline, endline
- **Monthly income** (USD): exponential distribution reflecting rural household income
- **GPS coordinates**: realistic bounds for Rwanda

Seed is fixed (`SEED = 42`) for reproducibility.

## Output

Each demo notebook produces:
- Inline benchmark results and visualizations
- Interactive HTML charts saved to the project directory
- Temporary data files (cleaned up automatically)

## Dependencies

Core libraries:
- `pandas` - Traditional DataFrame library
- `polars` - Modern high-performance DataFrame library
- `pyarrow` - Parquet file support
- `duckdb` - Embedded analytical database
- `psycopg2-binary` + `testing.postgresql` - PostgreSQL support
- `plotly` - Interactive visualizations
- `numpy` - Data generation

See [pyproject.toml](pyproject.toml) for complete dependency list.

## Development

The notebooks can be edited directly in VS Code or Jupyter Lab. All code is contained in the `.ipynb` files.

## Environment

- **Python**: 3.12 (Windows)
- **Venv**: `.venv/` (auto-detected by VS Code)
- **Package manager**: UV via `pyproject.toml`

## Files

| File | Description |
|------|-------------|
| `demo1_parquet_vs_csv.ipynb` | Parquet vs CSV benchmark |
| `demo2_polars_vs_pandas.ipynb` | Polars vs Pandas benchmark |
| `demo3_duckdb_vs_sql.ipynb` | DuckDB vs PostgreSQL benchmark |
| `pyproject.toml` | Dependencies and project config |
| `README.md` | This file |

## Notes

- Demo 3 uses `testing.postgresql` to automatically spin up a temporary PostgreSQL server - no manual setup required
- All benchmarks include interactive Plotly charts saved as HTML files
- Results are reproducible across runs due to fixed random seed

## License

For use by IPA Rwanda country office demonstrations.
