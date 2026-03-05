# Data Efficiency Tools — Project Context

Presentation demos for the **IPA Rwanda country office** on modern data processing tools.
Three self-contained benchmarks comparing old vs new approaches on a synthetic 500k-row
Rwanda household survey dataset.

---

## Environment

- **Python**: 3.12 (Windows, managed by UV)
- **Venv**: `.venv/` in this folder — VS Code picks it up automatically
- **Dependency management**: UV via `pyproject.toml`

```bash
uv sync                  # install / update all dependencies
uv run --script <file>   # run a .py script in an isolated env (no venv needed)
```

The `.venv/` is a **Windows venv** (contains `Lib/` and `Scripts/`, not `lib/bin/`).
It cannot be modified or deleted from the Linux VM — only the user's machine can touch it.

---

## Files

| File | Purpose |
|---|---|
| `demo1_parquet_vs_csv.py` | Benchmark: Parquet vs CSV (write/read/size/query) |
| `demo1_parquet_vs_csv.ipynb` | Notebook version of demo1 |
| `demo2_polars_vs_pandas.py` | Benchmark: Polars vs Pandas (5 operations) |
| `demo2_polars_vs_pandas.ipynb` | Notebook version of demo2 |
| `demo3_duckdb_vs_sql.py` | Benchmark: DuckDB vs PostgreSQL (4 queries + Parquet scan) |
| `demo3_duckdb_vs_sql.ipynb` | Notebook version of demo3 |
| `pyproject.toml` | All dependencies — edit here, then `uv sync` |
| `CLAUDE.md` | This file |

**Do not delete the `.py` files** — the `.ipynb` files are generated from them.
To regenerate a notebook after editing a `.py`, run the converter snippet in `CLAUDE.md` (below).

---

## Dataset

All three demos generate the **same synthetic Rwanda household survey** with 500,000 rows:

- `household_id` — `RW_XXXXXXX` format
- `district` — 10 representative Rwandan districts across 5 provinces
- `treatment_arm` — `control`, `treatment_A`, `treatment_B`
- `survey_round` — `baseline`, `midline`, `endline`
- `monthly_income_usd` — exponential distribution, scale=45 (reflects ~$30–60/month rural median)
- GPS bounds: lat −1.0 to −2.8, lon 28.9 to 30.9

Seed is fixed (`SEED = 42`) — results are reproducible across machines.

---

## Dependencies

```toml
pandas>=2.2           # demo1, demo2, demo3
polars>=0.20          # demo2
pyarrow>=15.0         # Parquet engine for all demos
plotly>=5.20          # interactive HTML charts
numpy>=1.26           # data generation
kaleido>=0.2          # static image export (optional)
nbformat>=4.2.0       # notebook format validation
duckdb>=0.10          # demo3
psycopg2-binary>=2.9  # demo3 PostgreSQL client
testing.postgresql>=1.3  # demo3 — spins up a temporary PostgreSQL server
sqlalchemy>=2.0       # demo3 utility
```

---

## Demo 3 PostgreSQL Requirement

`demo3` uses `testing.postgresql` to automatically start a **temporary PostgreSQL server**
in a temp directory — no manual server setup needed. It starts, benchmarks, and tears down.

**Prerequisite:** PostgreSQL binaries must be in PATH.
- **Windows**: install from https://www.postgresql.org/download/ and add
  `C:\Program Files\PostgreSQL\<version>\bin` to your system PATH.
- **Mac**: `brew install postgresql`
- **Linux**: `apt install postgresql`

The script will print a clear error if binaries are not found.

---

## Regenerating Notebooks from .py Files

After editing any `.py`, regenerate its notebook with this snippet
(run from the project folder):

```python
import json, re, uuid, ast

def py_percent_to_ipynb(src_path, out_path):
    with open(src_path) as f: raw = f.read()
    parts = re.split(r'^(# %% ?(?:\[markdown\])?.*)\n', raw, flags=re.MULTILINE)
    cells = []
    def make_id(): return uuid.uuid4().hex[:8]
    preamble = parts[0].strip()
    if preamble:
        cells.append({"cell_type": "code", "id": make_id(), "metadata": {},
                      "source": [l+"\n" for l in preamble.splitlines()],
                      "outputs": [], "execution_count": None})
    i = 1
    while i < len(parts) - 1:
        marker, content = parts[i], parts[i+1]; i += 2
        is_md = "[markdown]" in marker
        lines = content.rstrip("\n").splitlines()
        while lines and not lines[-1].strip(): lines.pop()
        if not lines: continue
        if is_md:
            src = [(l[2:] if l.startswith("# ") else ("" if l=="#" else l))+"\n" for l in lines]
            if src: src[-1] = src[-1].rstrip("\n")
            cells.append({"cell_type": "markdown", "id": make_id(), "metadata": {}, "source": src})
        else:
            while lines and not lines[0].strip(): lines.pop(0)
            src = [l+"\n" for l in lines]
            if src: src[-1] = src[-1].rstrip("\n")
            cells.append({"cell_type": "code", "id": make_id(), "metadata": {},
                          "source": src, "outputs": [], "execution_count": None})
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3",
                                      "language": "python", "name": "python3"},
                       "language_info": {"name": "python", "version": "3.12.0"}},
          "cells": cells}
    with open(out_path, "w") as f: json.dump(nb, f, indent=1)
    print(f"Written {out_path}: {len(cells)} cells")

py_percent_to_ipynb("demo1_parquet_vs_csv.py",   "demo1_parquet_vs_csv.ipynb")
py_percent_to_ipynb("demo2_polars_vs_pandas.py",  "demo2_polars_vs_pandas.ipynb")
py_percent_to_ipynb("demo3_duckdb_vs_sql.py",     "demo3_duckdb_vs_sql.ipynb")
```

After regeneration, manually update the **Setup cell** (cell index 1) in each notebook
to replace any raw UV `/// script` block with the human-readable setup instructions.
See the existing notebooks for the correct wording.

---

## VM Constraints (for Claude sessions)

- **No root / no sudo** in the Linux VM
- **No internet from the VM** (pip, apt, UV package downloads all blocked)
- `.venv` is a Windows venv — **do not attempt to delete or recreate it from the VM**
- Use `uv run --script <file>` to test scripts if packages are already cached; otherwise
  edits must be verified via `ast.parse()` syntax checks only
- The workspace folder is mounted at `/sessions/sweet-relaxed-babbage/mnt/data-efficiency-tools`
