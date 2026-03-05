# %% [markdown]
# # Demo 1: Parquet vs CSV — Storage & Speed
#
# This demo compares **Parquet** and **CSV** file formats using a synthetic
# household survey dataset (~500,000 rows) that mimics typical field research data.
#
# We benchmark:
# - **Write time** — how long it takes to save the dataset
# - **Read time** — how long it takes to load it back
# - **File size** — how much disk space each format uses
# - **Filtered query time** — a common operation (filter + aggregate)

# %% [markdown]
# ## Setup
#
# This script uses [UV](https://docs.astral.sh/uv/) for dependency management.
# Run it with:
# ```bash
# uv run --script demo1_parquet_vs_csv.py
# ```
# UV will automatically install all required packages into an isolated environment.

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "pyarrow>=15.0",
#   "plotly>=5.20",
#   "numpy>=1.26",
#   "kaleido>=0.2",
# ]
# ///

# %% [markdown]
# ## 1. Generate Synthetic Household Survey Data

# %%
import time
import os
import tempfile
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEED = 42
N_ROWS = 500_000
rng = np.random.default_rng(SEED)

print(f"Generating {N_ROWS:,} synthetic household survey records...")

# Rwanda's 30 districts span 5 provinces; we use a representative 10
DISTRICTS = ["Gasabo", "Kicukiro", "Nyarugenge",   # Kigali City
             "Musanze", "Gicumbi",                  # Northern
             "Huye", "Nyanza",                      # Southern
             "Nyagatare", "Rwamagana",              # Eastern
             "Rubavu"]                              # Western
ENUMERATORS = [f"ENUM_{i:03d}" for i in range(1, 51)]
TREATMENTS = ["control", "treatment_A", "treatment_B"]
SURVEY_ROUNDS = ["baseline", "midline", "endline"]

df = pd.DataFrame({
    "household_id":     [f"RW_{i:07d}" for i in rng.integers(1, 200_001, N_ROWS)],
    "survey_date":      pd.to_datetime(
                            rng.integers(
                                pd.Timestamp("2022-01-01").value,
                                pd.Timestamp("2024-12-31").value,
                                N_ROWS
                            )
                        ),
    "district":         rng.choice(DISTRICTS, N_ROWS),
    "enumerator_id":    rng.choice(ENUMERATORS, N_ROWS),
    "survey_round":     rng.choice(SURVEY_ROUNDS, N_ROWS),
    "treatment_arm":    rng.choice(TREATMENTS, N_ROWS),
    "hh_size":          rng.integers(1, 9, N_ROWS),      # avg ~4.3 in Rwanda
    "head_age":         rng.integers(18, 75, N_ROWS),
    "head_female":      rng.integers(0, 2, N_ROWS),
    # Monthly income in USD (median rural HH ~$30–60/month)
    "monthly_income_usd":   rng.exponential(scale=45, size=N_ROWS).round(2),
    "monthly_expenditure_usd": rng.exponential(scale=40, size=N_ROWS).round(2),
    "food_insecure":    rng.integers(0, 2, N_ROWS),
    "owns_land":        rng.integers(0, 2, N_ROWS),
    "has_mobile_money": rng.integers(0, 2, N_ROWS),
    "asset_index":      rng.uniform(0, 10, N_ROWS).round(3),
    "consumption_score": rng.normal(50, 15, N_ROWS).clip(0, 100).round(2),
    "interview_duration_min": rng.integers(15, 90, N_ROWS),
    # Rwanda GPS bounds: lat -1.0 to -2.8, lon 28.9 to 30.9
    "gps_latitude":     rng.uniform(-2.8, -1.0, N_ROWS).round(6),
    "gps_longitude":    rng.uniform(28.9, 30.9, N_ROWS).round(6),
    "notes":            rng.choice(
                            ["", "revisit needed", "proxy respondent",
                             "partial completion", "data quality flag", ""],
                            N_ROWS
                        ),
})

print(f"Dataset shape: {df.shape}")
print(df.dtypes)
print(df.head(3))

# %% [markdown]
# ## 2. Benchmark: Write Times

# %%
tmpdir = tempfile.mkdtemp()
csv_path     = os.path.join(tmpdir, "survey.csv")
parquet_path = os.path.join(tmpdir, "survey.parquet")

# --- CSV write ---
t0 = time.perf_counter()
df.to_csv(csv_path, index=False)
csv_write_time = time.perf_counter() - t0

# --- Parquet write ---
t0 = time.perf_counter()
df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")
parquet_write_time = time.perf_counter() - t0

print(f"CSV    write: {csv_write_time:.2f}s")
print(f"Parquet write: {parquet_write_time:.2f}s")

# %% [markdown]
# ## 3. Benchmark: File Sizes

# %%
csv_size_mb     = os.path.getsize(csv_path) / 1_048_576
parquet_size_mb = os.path.getsize(parquet_path) / 1_048_576

print(f"CSV     size: {csv_size_mb:.1f} MB")
print(f"Parquet size: {parquet_size_mb:.1f} MB")
print(f"Size reduction: {(1 - parquet_size_mb / csv_size_mb) * 100:.1f}%")

# %% [markdown]
# ## 4. Benchmark: Read Times

# %%
# --- CSV read ---
t0 = time.perf_counter()
_ = pd.read_csv(csv_path)
csv_read_time = time.perf_counter() - t0

# --- Parquet read ---
t0 = time.perf_counter()
_ = pd.read_parquet(parquet_path, engine="pyarrow")
parquet_read_time = time.perf_counter() - t0

print(f"CSV    read: {csv_read_time:.2f}s")
print(f"Parquet read: {parquet_read_time:.2f}s")

# %% [markdown]
# ## 5. Benchmark: Filtered Query (Read Only Needed Columns + Filter)
#
# A key Parquet advantage: **predicate pushdown** and **column pruning**.
# Here we read only 3 columns and filter rows — Parquet never loads the rest.

# %%
# --- CSV: must read entire file first ---
t0 = time.perf_counter()
df_csv = pd.read_csv(csv_path, usecols=["district", "treatment_arm", "monthly_income_usd"])
result_csv = (
    df_csv[df_csv["treatment_arm"] == "treatment_A"]
    .groupby("district")["monthly_income_usd"]
    .mean()
)
csv_query_time = time.perf_counter() - t0

# --- Parquet: column pruning + filter ---
t0 = time.perf_counter()
df_pq = pd.read_parquet(
    parquet_path,
    engine="pyarrow",
    columns=["district", "treatment_arm", "monthly_income_usd"],
    filters=[("treatment_arm", "==", "treatment_A")],
)
result_pq = df_pq.groupby("district")["monthly_income_usd"].mean()
parquet_query_time = time.perf_counter() - t0

print(f"CSV    query: {csv_query_time:.2f}s")
print(f"Parquet query: {parquet_query_time:.2f}s")

# %% [markdown]
# ## 6. Visualize Results

# %%
labels = ["CSV", "Parquet"]
colors = ["#E05C5C", "#3A86FF"]

# Operation descriptions shown in hover tooltips and captions
OPERATIONS = {
    "write": {
        "csv":     "df.to_csv(survey.csv)",
        "parquet": "df.to_parquet(survey.parquet, compression='snappy')",
    },
    "size": {
        "csv":     f"Plain text, one row per line\n({N_ROWS:,} rows × 20 columns)",
        "parquet": f"Columnar + Snappy compression\n({N_ROWS:,} rows × 20 columns)",
    },
    "read": {
        "csv":     "pd.read_csv(survey.csv)  — full file, all columns",
        "parquet": "pd.read_parquet(survey.parquet)  — full file, all columns",
    },
    "query": {
        "csv":     (
            "pd.read_csv(usecols=[district, treatment_arm, monthly_income_usd])\n"
            "→ filter treatment_arm == 'treatment_A'\n"
            "→ groupby district → mean income"
        ),
        "parquet": (
            "pd.read_parquet(columns=[district, treatment_arm, monthly_income_usd],\n"
            "                filters=[treatment_arm == 'treatment_A'])  ← pushed to disk\n"
            "→ groupby district → mean income"
        ),
    },
}

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "💾 Write Time (seconds)",
        "📦 File Size (MB)",
        "📖 Full Read Time (seconds)",
        "🔍 Filtered Query Time (seconds)",
    ],
    vertical_spacing=0.22,
    horizontal_spacing=0.12,
)

def bar_trace(values, hover_texts, row, col):
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition="outside",
            showlegend=False,
            customdata=hover_texts,
            hovertemplate="<b>%{x}</b><br>%{y:.3f}<br><br><i>%{customdata}</i><extra></extra>",
        ),
        row=row, col=col,
    )

bar_trace(
    [csv_write_time, parquet_write_time],
    [OPERATIONS["write"]["csv"], OPERATIONS["write"]["parquet"]],
    row=1, col=1,
)
bar_trace(
    [csv_size_mb, parquet_size_mb],
    [OPERATIONS["size"]["csv"], OPERATIONS["size"]["parquet"]],
    row=1, col=2,
)
bar_trace(
    [csv_read_time, parquet_read_time],
    [OPERATIONS["read"]["csv"], OPERATIONS["read"]["parquet"]],
    row=2, col=1,
)
bar_trace(
    [csv_query_time, parquet_query_time],
    [OPERATIONS["query"]["csv"], OPERATIONS["query"]["parquet"]],
    row=2, col=2,
)

def pct_improvement(old, new):
    return f"Parquet is {(old/new):.1f}× faster" if new < old else f"CSV is {(new/old):.1f}× faster"

def size_improvement(old, new):
    return f"{(1 - new/old)*100:.0f}% smaller"

# Subplot captions: what operation was run (italic, grey, below each title)
CAPTION_STYLE = dict(xref="paper", yref="paper", showarrow=False,
                     font=dict(size=10, color="#888"), align="center")

annotations = [
    # Row 1 captions
    dict(x=0.22, y=0.985, text="<i>df.to_csv() vs df.to_parquet(snappy)</i>",        **CAPTION_STYLE),
    dict(x=0.78, y=0.985, text=f"<i>{N_ROWS:,} rows × 20 cols — plain text vs columnar+compressed</i>", **CAPTION_STYLE),
    # Row 2 captions
    dict(x=0.22, y=0.455, text="<i>pd.read_csv() vs pd.read_parquet() — all 20 columns</i>",             **CAPTION_STYLE),
    dict(x=0.78, y=0.455, text="<i>3 cols selected + filter pushed to disk (Parquet only)</i>",           **CAPTION_STYLE),
    # Improvement callouts
    dict(x=0.22, y=1.06, text=pct_improvement(csv_write_time, parquet_write_time),
         font=dict(size=11, color="#2a6dd9"), **{k: v for k, v in CAPTION_STYLE.items() if k != "font"}),
    dict(x=0.78, y=1.06, text=size_improvement(csv_size_mb, parquet_size_mb),
         font=dict(size=11, color="#2a6dd9"), **{k: v for k, v in CAPTION_STYLE.items() if k != "font"}),
    dict(x=0.22, y=0.475, text=pct_improvement(csv_read_time, parquet_read_time),
         font=dict(size=11, color="#2a6dd9"), **{k: v for k, v in CAPTION_STYLE.items() if k != "font"}),
    dict(x=0.78, y=0.475, text=pct_improvement(csv_query_time, parquet_query_time),
         font=dict(size=11, color="#2a6dd9"), **{k: v for k, v in CAPTION_STYLE.items() if k != "font"}),
]

fig.update_layout(
    title=dict(
        text=(
            f"<b>Parquet vs CSV: Storage & Speed Comparison</b>"
            f"<br><sup>{N_ROWS:,} synthetic household survey rows (Rwanda) — hover bars for exact operations</sup>"
        ),
        font=dict(size=18),
    ),
    height=760,
    plot_bgcolor="white",
    paper_bgcolor="white",
    annotations=annotations,
    margin=dict(t=130, b=40),
)
fig.update_yaxes(gridcolor="#eee")

fig.show()
fig.write_html("demo1_parquet_vs_csv_results.html")
print("Chart saved to demo1_parquet_vs_csv_results.html")

# %% [markdown]
# ## Summary
#
# | Metric | CSV | Parquet | Improvement |
# |---|---|---|---|
# | Write time | — | — | See chart |
# | File size | — | — | See chart |
# | Read time | — | — | See chart |
# | Query time | — | — | See chart |
#
# ### Why Parquet wins:
# - **Columnar storage**: only reads columns you actually need
# - **Built-in compression**: Snappy/Gzip encoding reduces size dramatically
# - **Type preservation**: dates stay dates, ints stay ints (no re-parsing)
# - **Predicate pushdown**: filters applied before data hits memory
