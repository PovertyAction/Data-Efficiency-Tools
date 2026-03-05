# %% [markdown]
# # Demo 3: DuckDB vs PostgreSQL — Analytical Query Performance
#
# This demo compares **DuckDB** against a real **PostgreSQL** instance on
# analytical workloads typical of survey data processing.
#
# A temporary, self-contained PostgreSQL server is spun up automatically
# using `testing.postgresql` — no manual server setup required. It starts,
# runs the benchmarks, and tears itself down cleanly.
#
# ### What we benchmark
# All queries run against the same 500,000-row Rwanda household survey dataset:
#
# 1. **Ingestion** — bulk-loading a DataFrame into each engine
# 2. **GROUP BY aggregation** — district × treatment arm summary statistics
# 3. **Filtered scan** — multi-condition WHERE + GROUP BY (the analytics workhorse)
# 4. **Window function** — per-district income ranking
#
# ### DuckDB bonus: direct Parquet scanning
# DuckDB can query a Parquet file **on disk without any ingestion** — compared
# against PostgreSQL's full ingest + query cold-start time.
#
# ### MotherDuck
# [MotherDuck](https://motherduck.com) is the managed cloud service built on DuckDB.
# Same SQL dialect, same Python API — see the final cell for a connection example.
#
# ### Prerequisite
# PostgreSQL must be installed. The demo **auto-detects** the binaries from common
# installation paths — no manual PATH configuration needed.
# Download from https://www.postgresql.org/download/ if not already installed.

# %% [markdown]
# ## Setup
#
# **Running as a notebook:** select the `.venv` kernel in VS Code and run all cells with *Run All*.
#
# **Running as a standalone script** (no kernel needed):
# ```bash
# uv run --script demo3_duckdb_vs_sql.py
# ```

# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "duckdb>=0.10",
#   "psycopg2-binary>=2.9",
#   "testing.postgresql>=1.3",
#   "sqlalchemy>=2.0",
#   "pandas>=2.2",
#   "pyarrow>=15.0",
#   "plotly>=5.20",
#   "numpy>=1.26",
# ]
# ///

# %% [markdown]
# ## 1. Generate Synthetic Rwanda Household Survey Data

# %%
import io
import sys
import time
import os
import tempfile
import numpy as np
import pandas as pd
import duckdb
import psycopg2
import psycopg2.extras
import testing.postgresql
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEED     = 42
N_ROWS   = 500_000
N_REPEATS = 3
rng = np.random.default_rng(SEED)

print(f"Generating {N_ROWS:,} synthetic household survey records...")

# Rwanda: representative 10 of 30 districts across 5 provinces
DISTRICTS = ["Gasabo", "Kicukiro", "Nyarugenge",   # Kigali City
             "Musanze", "Gicumbi",                  # Northern
             "Huye", "Nyanza",                      # Southern
             "Nyagatare", "Rwamagana",              # Eastern
             "Rubavu"]                              # Western
ENUMERATORS = [f"ENUM_{i:03d}" for i in range(1, 51)]
TREATMENTS  = ["control", "treatment_A", "treatment_B"]
ROUNDS      = ["baseline", "midline", "endline"]

df = pd.DataFrame({
    "household_id":           [f"RW_{i:07d}" for i in rng.integers(1, 200_001, N_ROWS)],
    "survey_date":            pd.to_datetime(
                                  rng.integers(
                                      pd.Timestamp("2022-01-01").value,
                                      pd.Timestamp("2024-12-31").value,
                                      N_ROWS,
                                  )
                              ).astype(str),   # plain date string for PG compatibility
    "district":               rng.choice(DISTRICTS, N_ROWS),
    "enumerator_id":          rng.choice(ENUMERATORS, N_ROWS),
    "survey_round":           rng.choice(ROUNDS, N_ROWS),
    "treatment_arm":          rng.choice(TREATMENTS, N_ROWS),
    "hh_size":                rng.integers(1, 9, N_ROWS).astype(int),
    "head_age":               rng.integers(18, 75, N_ROWS).astype(int),
    "head_female":            rng.integers(0, 2, N_ROWS).astype(int),
    "monthly_income_usd":     rng.exponential(scale=45, size=N_ROWS).round(2),
    "monthly_expenditure_usd":rng.exponential(scale=40, size=N_ROWS).round(2),
    "food_insecure":          rng.integers(0, 2, N_ROWS).astype(int),
    "owns_land":              rng.integers(0, 2, N_ROWS).astype(int),
    "has_mobile_money":       rng.integers(0, 2, N_ROWS).astype(int),
    "asset_index":            rng.uniform(0, 10, N_ROWS).round(3),
    "consumption_score":      rng.normal(50, 15, N_ROWS).clip(0, 100).round(2),
    "interview_duration_min": rng.integers(15, 90, N_ROWS).astype(int),
    "gps_latitude":           rng.uniform(-2.8, -1.0, N_ROWS).round(6),
    "gps_longitude":          rng.uniform(28.9, 30.9, N_ROWS).round(6),
    "notes":                  rng.choice(
                                  ["", "revisit needed", "proxy respondent",
                                   "partial completion", "data quality flag", ""],
                                  N_ROWS,
                              ),
})

tmpdir = tempfile.mkdtemp()
parquet_path = os.path.join(tmpdir, "survey.parquet")
df.to_parquet(parquet_path, index=False, engine="pyarrow")
print(f"Dataset shape: {df.shape}")

# %% [markdown]
# ## 2. Start Temporary PostgreSQL Server
#
# `testing.postgresql` initialises a throw-away cluster in a temp directory,
# starts `postgres` on a random port, and tears it down at the end of the demo.
#
# The cell below **auto-detects** the PostgreSQL `bin/` directory in common
# installation paths — no manual PATH editing required.

# %%
import glob
import shutil

def _find_pg_bin() -> str | None:
    """Return the path to the PostgreSQL bin/ directory, or None if not found."""
    # Already in PATH?
    if shutil.which("initdb"):
        return None   # already fine, no injection needed

    candidates: list[str] = []

    if sys.platform == "win32":
        candidates = glob.glob(r"C:\Program Files\PostgreSQL\*\bin") + \
                     glob.glob(r"C:\Program Files (x86)\PostgreSQL\*\bin")
    elif sys.platform == "darwin":
        candidates = glob.glob("/opt/homebrew/opt/postgresql@*/bin") + \
                     glob.glob("/opt/homebrew/opt/postgresql/bin") + \
                     glob.glob("/usr/local/opt/postgresql@*/bin") + \
                     glob.glob("/usr/local/opt/postgresql/bin")
    else:  # Linux
        candidates = glob.glob("/usr/lib/postgresql/*/bin") + \
                     glob.glob("/usr/pgsql-*/bin")

    # Sort descending so the highest version is tried first
    candidates.sort(reverse=True)
    return candidates[0] if candidates else None

pg_bin = _find_pg_bin()
if pg_bin:
    os.environ["PATH"] = pg_bin + os.pathsep + os.environ.get("PATH", "")
    print(f"Auto-detected PostgreSQL binaries: {pg_bin}")
elif not shutil.which("initdb"):
    print("ERROR: PostgreSQL binaries not found.")
    print("Install PostgreSQL from https://www.postgresql.org/download/")
    print("On Windows the bin/ dir is usually: C:\\Program Files\\PostgreSQL\\<version>\\bin")
    sys.exit(1)

print("\nStarting temporary PostgreSQL instance...")
try:
    pg_instance = testing.postgresql.Postgresql()
    pg_dsn = pg_instance.dsn()
    print(f"  PostgreSQL running on port {pg_dsn['port']}  (managed by testing.postgresql)")
except Exception as e:
    print(f"\nERROR: Could not start PostgreSQL — {e}")
    sys.exit(1)

# %% [markdown]
# ## 3. Helper: Timed Benchmark Runner

# %%
def benchmark(fn, label, repeats=N_REPEATS):
    """Run fn() `repeats` times, return best elapsed time in seconds."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    print(f"  {label}: {best:.3f}s  (best of {repeats})")
    return best, result

# %% [markdown]
# ## 4. Benchmark 1: Data Ingestion
#
# How long does it take to bulk-load the DataFrame into each engine?
#
# - **PostgreSQL**: `COPY FROM STDIN` via a CSV buffer — the fastest bulk-load path
# - **DuckDB**: `CREATE TABLE … AS SELECT * FROM df` — zero-copy Arrow handoff

# %%
print("\n── Ingestion ──")

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS survey (
    household_id           TEXT,
    survey_date            TEXT,
    district               TEXT,
    enumerator_id          TEXT,
    survey_round           TEXT,
    treatment_arm          TEXT,
    hh_size                INTEGER,
    head_age               INTEGER,
    head_female            INTEGER,
    monthly_income_usd     DOUBLE PRECISION,
    monthly_expenditure_usd DOUBLE PRECISION,
    food_insecure          INTEGER,
    owns_land              INTEGER,
    has_mobile_money       INTEGER,
    asset_index            DOUBLE PRECISION,
    consumption_score      DOUBLE PRECISION,
    interview_duration_min INTEGER,
    gps_latitude           DOUBLE PRECISION,
    gps_longitude          DOUBLE PRECISION,
    notes                  TEXT
)
"""

def ingest_postgres():
    conn = psycopg2.connect(**pg_dsn)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS survey")
    cur.execute(CREATE_TABLE_SQL)
    # COPY FROM STDIN is the fastest bulk-load method
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    cur.copy_expert("COPY survey FROM STDIN WITH (FORMAT CSV, HEADER TRUE)", buf)
    conn.commit()
    cur.close()
    conn.close()

def ingest_duckdb():
    con = duckdb.connect()
    con.execute("CREATE OR REPLACE TABLE survey AS SELECT * FROM df")
    return con

pg_ingest_time, _       = benchmark(ingest_postgres, "PostgreSQL (COPY FROM STDIN)")
duck_ingest_time, duck_con = benchmark(ingest_duckdb, "DuckDB (CREATE TABLE AS SELECT)")

# %% [markdown]
# ## 5. Benchmark 2: GROUP BY Aggregation
#
# Summarise income and consumption by district and treatment arm — the most
# common query pattern in programme reporting.
#
# ```sql
# SELECT district, treatment_arm,
#        COUNT(*)                    AS n,
#        AVG(monthly_income_usd)     AS avg_income,
#        AVG(consumption_score)      AS avg_consumption,
#        SUM(food_insecure)          AS n_food_insecure
# FROM survey
# GROUP BY district, treatment_arm
# ORDER BY district, treatment_arm
# ```

# %%
print("\n── GROUP BY aggregation ──")

QUERY_GROUPBY = """
SELECT district, treatment_arm,
       COUNT(*)                    AS n,
       AVG(monthly_income_usd)     AS avg_income,
       AVG(consumption_score)      AS avg_consumption,
       SUM(food_insecure)          AS n_food_insecure
FROM survey
GROUP BY district, treatment_arm
ORDER BY district, treatment_arm
"""

def run_pg_groupby():
    conn = psycopg2.connect(**pg_dsn)
    result = pd.read_sql_query(QUERY_GROUPBY, conn)
    conn.close()
    return result

pg_groupby_time,   _ = benchmark(run_pg_groupby,                          "PostgreSQL")
duck_groupby_time, _ = benchmark(lambda: duck_con.execute(QUERY_GROUPBY).df(), "DuckDB")

# %% [markdown]
# ## 6. Benchmark 3: Filtered Scan
#
# Multi-condition filter + aggregation — the kind of query used to identify
# the most vulnerable households for targeting:
#
# *"Among food-insecure households at endline without mobile money, what is
# the average income and asset index by district?"*
#
# ```sql
# SELECT district,
#        COUNT(*)                AS n,
#        AVG(monthly_income_usd) AS avg_income,
#        AVG(asset_index)        AS avg_assets
# FROM survey
# WHERE food_insecure    = 1
#   AND survey_round     = 'endline'
#   AND has_mobile_money = 0
# GROUP BY district
# ORDER BY avg_income ASC
# ```

# %%
print("\n── Filtered scan ──")

QUERY_FILTER = """
SELECT district,
       COUNT(*)                AS n,
       AVG(monthly_income_usd) AS avg_income,
       AVG(asset_index)        AS avg_assets
FROM survey
WHERE food_insecure    = 1
  AND survey_round     = 'endline'
  AND has_mobile_money = 0
GROUP BY district
ORDER BY avg_income ASC
"""

def run_pg_filter():
    conn = psycopg2.connect(**pg_dsn)
    result = pd.read_sql_query(QUERY_FILTER, conn)
    conn.close()
    return result

pg_filter_time,   _ = benchmark(run_pg_filter,                          "PostgreSQL")
duck_filter_time, _ = benchmark(lambda: duck_con.execute(QUERY_FILTER).df(), "DuckDB")

# %% [markdown]
# ## 7. Benchmark 4: Window Function
#
# Rank every treatment_A household within its district by income, and compute
# deviation from the district mean — a common step before outlier flagging or
# targeting analysis.
#
# ```sql
# SELECT household_id, district, monthly_income_usd,
#        AVG(monthly_income_usd) OVER (PARTITION BY district) AS district_avg,
#        monthly_income_usd
#          - AVG(monthly_income_usd) OVER (PARTITION BY district) AS deviation,
#        RANK() OVER (
#            PARTITION BY district ORDER BY monthly_income_usd DESC
#        ) AS income_rank
# FROM survey
# WHERE treatment_arm = 'treatment_A'
# ```

# %%
print("\n── Window function ──")

QUERY_WINDOW = """
SELECT household_id, district, monthly_income_usd,
       AVG(monthly_income_usd) OVER (PARTITION BY district) AS district_avg,
       monthly_income_usd
         - AVG(monthly_income_usd) OVER (PARTITION BY district) AS deviation,
       RANK() OVER (
           PARTITION BY district ORDER BY monthly_income_usd DESC
       ) AS income_rank
FROM survey
WHERE treatment_arm = 'treatment_A'
"""

def run_pg_window():
    conn = psycopg2.connect(**pg_dsn)
    result = pd.read_sql_query(QUERY_WINDOW, conn)
    conn.close()
    return result

pg_window_time,   _ = benchmark(run_pg_window,                          "PostgreSQL")
duck_window_time, _ = benchmark(lambda: duck_con.execute(QUERY_WINDOW).df(), "DuckDB")

# %% [markdown]
# ## 8. DuckDB Superpower: Direct Parquet Scanning
#
# DuckDB can query a Parquet file **on disk with zero ingestion** — the file
# is treated as a virtual table. We compare the full cold-start cost:
# PostgreSQL must ingest before querying; DuckDB does not.

# %%
print("\n── Cold-start: PostgreSQL (ingest + query) vs DuckDB direct Parquet scan ──")

QUERY_PARQUET = f"""
SELECT district, treatment_arm,
       COUNT(*)                AS n,
       AVG(monthly_income_usd) AS avg_income
FROM '{parquet_path}'
GROUP BY district, treatment_arm
ORDER BY district, treatment_arm
"""

def pg_cold_start():
    """Full pipeline: ingest via COPY → run GROUP BY query."""
    conn = psycopg2.connect(**pg_dsn)
    cur  = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS survey_tmp")
    cur.execute(CREATE_TABLE_SQL.replace("survey", "survey_tmp"))
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    cur.copy_expert("COPY survey_tmp FROM STDIN WITH (FORMAT CSV, HEADER TRUE)", buf)
    conn.commit()
    result = pd.read_sql_query(QUERY_GROUPBY.replace("survey", "survey_tmp"), conn)
    cur.execute("DROP TABLE survey_tmp")
    conn.commit()
    cur.close()
    conn.close()
    return result

def duckdb_direct_parquet():
    """No ingestion — query the Parquet file on disk directly."""
    return duckdb.execute(QUERY_PARQUET).df()

pg_cold_time,     _ = benchmark(pg_cold_start,        "PostgreSQL (ingest + query)")
duck_parquet_time, r = benchmark(duckdb_direct_parquet, "DuckDB direct Parquet scan")

print(f"\nDuckDB Parquet result preview:\n{r.head(4).to_string(index=False)}")

# %% [markdown]
# ## 9. Shut Down the Temporary PostgreSQL Server

# %%
pg_instance.stop()
print("PostgreSQL temporary instance stopped and cleaned up.")

# %% [markdown]
# ## 10. Visualize Results

# %%
PG_COLOR   = "#E05C5C"
DUCK_COLOR = "#3A86FF"
labels     = ["PostgreSQL\n(row-store)", "DuckDB\n(columnar)"]

benchmarks = [
    ("Ingestion",             pg_ingest_time,  duck_ingest_time,
     "COPY FROM STDIN (CSV buffer) vs CREATE TABLE AS SELECT * FROM df",
     "Bulk-loading 500k rows into the engine"),
    ("GROUP BY\nAggregation", pg_groupby_time, duck_groupby_time,
     "SELECT district, treatment_arm, COUNT(*), AVG(income), … GROUP BY …",
     "Summarising all rows by district × treatment arm"),
    ("Filtered\nScan",        pg_filter_time,  duck_filter_time,
     "WHERE food_insecure=1 AND round='endline' AND has_mobile_money=0 … GROUP BY district",
     "Multi-condition filter + aggregation (targeting query)"),
    ("Window\nFunction",      pg_window_time,  duck_window_time,
     "RANK() OVER (PARTITION BY district ORDER BY income DESC) + deviation from mean",
     "Per-district income ranking across treatment_A households"),
]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[b[0].replace("\n", " ") for b in benchmarks],
    vertical_spacing=0.25,
    horizontal_spacing=0.12,
)

CAPTION_STYLE = dict(xref="paper", yref="paper", showarrow=False,
                     align="center", font=dict(size=10, color="#888"))
CALLOUT_STYLE = dict(xref="paper", yref="paper", showarrow=False,
                     align="center", font=dict(size=11, color="#2a6dd9"))

annotations = []
positions   = [(1, 1, 0.22, 1.06, 0.985),
               (1, 2, 0.78, 1.06, 0.985),
               (2, 1, 0.22, 0.465, 0.455),
               (2, 2, 0.78, 0.465, 0.455)]

for (row, col, px, py_call, py_cap), (title, pg_t, dq_t, query_desc, caption) in \
        zip(positions, benchmarks):
    speedup = pg_t / dq_t
    callout = (f"DuckDB is {speedup:.1f}× faster" if speedup >= 1
               else f"PostgreSQL is {1/speedup:.1f}× faster")

    fig.add_trace(go.Bar(
        x=labels,
        y=[pg_t, dq_t],
        marker_color=[PG_COLOR, DUCK_COLOR],
        text=[f"{pg_t:.3f}s", f"{dq_t:.3f}s"],
        textposition="outside",
        showlegend=False,
        customdata=[f"PostgreSQL — {query_desc}", f"DuckDB — {query_desc}"],
        hovertemplate="<b>%{x}</b><br>%{y:.3f}s<br><br><i>%{customdata}</i><extra></extra>",
    ), row=row, col=col)

    annotations.append(dict(x=px, y=py_call, text=callout, **CALLOUT_STYLE))
    annotations.append(dict(x=px, y=py_cap, text=f"<i>{caption}</i>", **CAPTION_STYLE))

fig.update_layout(
    title=dict(
        text=(
            "<b>DuckDB vs PostgreSQL: Analytical Query Performance</b>"
            f"<br><sup>{N_ROWS:,} synthetic household survey rows (Rwanda) "
            "— hover bars for exact queries</sup>"
        ),
        font=dict(size=17),
    ),
    height=760,
    plot_bgcolor="white",
    paper_bgcolor="white",
    annotations=annotations,
    margin=dict(t=130, b=40),
)
fig.update_yaxes(gridcolor="#eee")
fig.show()
fig.write_html("demo3_duckdb_vs_sql_results.html")
print("Chart saved to demo3_duckdb_vs_sql_results.html")

# %% [markdown]
# ## 11. DuckDB Direct Parquet Scan vs PostgreSQL Cold Start

# %%
fig2 = go.Figure()

fig2.add_trace(go.Bar(
    name="PostgreSQL",
    x=["Cold-start (ingest + query)"],
    y=[pg_cold_time],
    marker_color=PG_COLOR,
    text=[f"{pg_cold_time:.2f}s"],
    textposition="outside",
    customdata=["COPY FROM STDIN into temp table → GROUP BY query"],
    hovertemplate="<b>%{x}</b><br>%{y:.3f}s<br><br><i>%{customdata}</i><extra></extra>",
))

fig2.add_trace(go.Bar(
    name="DuckDB",
    x=["Cold-start (ingest + query)"],
    y=[duck_parquet_time],
    marker_color=DUCK_COLOR,
    text=[f"{duck_parquet_time:.2f}s"],
    textposition="outside",
    customdata=[f"SELECT … FROM 'survey.parquet' — no ingestion step"],
    hovertemplate="<b>%{x}</b><br>%{y:.3f}s<br><br><i>%{customdata}</i><extra></extra>",
))

speedup = pg_cold_time / duck_parquet_time
fig2.update_layout(
    title=dict(
        text=(
            "<b>Cold-start: PostgreSQL (ingest + query) vs DuckDB direct Parquet scan</b>"
            f"<br><sup>DuckDB is <b>{speedup:.1f}×</b> faster — it skips ingestion entirely "
            "by querying the Parquet file on disk</sup>"
        ),
        font=dict(size=16),
    ),
    barmode="group",
    height=450,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    margin=dict(t=120, b=40),
    yaxis=dict(title="seconds", gridcolor="#eee"),
)
fig2.show()
fig2.write_html("demo3_parquet_scan_results.html")
print("Chart saved to demo3_parquet_scan_results.html")

# %% [markdown]
# ## 12. MotherDuck — DuckDB in the Cloud
#
# [MotherDuck](https://motherduck.com) is the managed cloud platform built on DuckDB.
# It runs the same SQL dialect and Python API — the only change is the connection string.
#
# ### What it adds over local DuckDB
#
# | Feature | DuckDB (local) | MotherDuck (cloud) |
# |---|---|---|
# | Setup | `pip install duckdb` | Sign up at motherduck.com |
# | Storage | Local files / in-memory | Persistent cloud databases |
# | Sharing | Copy files manually | Share databases by name |
# | Scale | Your laptop's RAM | Serverless cloud compute |
# | SQL dialect | DuckDB SQL | Identical DuckDB SQL |
# | Python API | `duckdb.connect()` | `duckdb.connect("md:")` |
#
# ### Connecting (requires a free MotherDuck account)
#
# ```python
# import duckdb
#
# # Authenticate — token stored in ~/.motherduck/token after first login
# con = duckdb.connect("md:")
#
# # Upload your local survey data to the cloud
# con.execute("CREATE DATABASE IF NOT EXISTS rwanda_surveys")
# con.execute("USE rwanda_surveys")
# con.execute(f"CREATE OR REPLACE TABLE survey AS SELECT * FROM '{parquet_path}'")
#
# # Now any teammate with access can run the same queries
# result = con.execute("""
#     SELECT district, AVG(monthly_income_usd) AS avg_income
#     FROM survey
#     GROUP BY district
#     ORDER BY avg_income DESC
# """).df()
# ```
#
# ### Hybrid execution
# MotherDuck can join a **cloud table with a local Parquet file** in a single query —
# the planner decides what runs locally vs in the cloud automatically:
#
# ```python
# con.execute(f"""
#     SELECT cloud.district, local.monthly_income_usd
#     FROM rwanda_surveys.survey AS cloud
#     JOIN read_parquet('{parquet_path}') AS local USING (household_id)
# """)
# ```

# %% [markdown]
# ## Summary
#
# | Benchmark | PostgreSQL | DuckDB | Why DuckDB wins |
# |-----------|------------|--------|-----------------|
# | Ingestion | COPY FROM STDIN | Arrow zero-copy | No serialisation overhead |
# | GROUP BY aggregation | Row scan + hash agg | Vectorised columnar scan | Only reads needed columns; SIMD batch processing |
# | Filtered scan | Seq scan + filter | Pushdown into column chunks | Skips irrelevant data earlier |
# | Window function | Sort + partition | Parallel partition execution | Multi-core by default |
# | Cold-start (Parquet) | Ingest + query | Direct scan | No ingestion step at all |
#
# ### When PostgreSQL is still the right choice
# - **Transactional workloads** (INSERT/UPDATE/DELETE at high frequency)
# - **Concurrent multi-user writes** (ACID guarantees, row-level locking)
# - **Complex permissions and row-level security**
# - **Existing infrastructure** already running PostgreSQL
#
# DuckDB and PostgreSQL are complementary: PostgreSQL stores and serves operational
# data; DuckDB (or MotherDuck) handles the analytics layer on top.
