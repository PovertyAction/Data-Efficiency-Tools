# %% [markdown]
# # Demo 2: Polars vs Pandas — Processing Speed
#
# This demo compares **Polars** and **Pandas** on common data processing tasks
# using the same synthetic household survey dataset (~500,000 rows).
#
# We benchmark five operations that appear constantly in survey data work:
# 1. **Read** a Parquet file
# 2. **Filter** rows by condition
# 3. **GroupBy + Aggregate** (mean income by district × treatment arm)
# 4. **String operations** (extract/clean a text column)
# 5. **Join** two DataFrames (simulate merging in a lookup table)

# %% [markdown]
# ## Setup
#
# Run with:
# ```bash
# uv run --script demo2_polars_vs_pandas.py
# ```

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas>=2.2",
#   "polars>=0.20",
#   "pyarrow>=15.0",
#   "plotly>=5.20",
#   "numpy>=1.26",
#   "kaleido>=0.2",
# ]
# ///

# %% [markdown]
# ## 1. Generate & Save the Synthetic Dataset
#
# We generate the same household survey data as Demo 1 and save it as Parquet.
# Both libraries will read the same file for a fair comparison.

# %%
import time
import os
import tempfile
import numpy as np
import pandas as pd
import polars as pl
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

df_pandas = pd.DataFrame({
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

tmpdir = tempfile.mkdtemp()
parquet_path = os.path.join(tmpdir, "survey.parquet")
df_pandas.to_parquet(parquet_path, index=False, engine="pyarrow")
print(f"Saved to {parquet_path}")

# %% [markdown]
# ## 2. Helper: Timed Benchmark Runner
#
# We run each operation multiple times and take the best time to reduce noise.

# %%
N_REPEATS = 3

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
# ## 3. Benchmark 1: Read Parquet

# %%
print("\n── Read ──")

pandas_read_time, df_pd = benchmark(
    lambda: pd.read_parquet(parquet_path, engine="pyarrow"),
    "pandas"
)

polars_read_time, df_pl = benchmark(
    lambda: pl.read_parquet(parquet_path),
    "polars"
)

# %% [markdown]
# ## 4. Benchmark 2: Filter Rows

# %%
print("\n── Filter (treatment_A, monthly_income_usd > 40) ──")

pandas_filter_time, _ = benchmark(
    lambda: df_pd[
        (df_pd["treatment_arm"] == "treatment_A") &
        (df_pd["monthly_income_usd"] > 40)
    ],
    "pandas"
)

polars_filter_time, _ = benchmark(
    lambda: df_pl.filter(
        (pl.col("treatment_arm") == "treatment_A") &
        (pl.col("monthly_income_usd") > 40)
    ),
    "polars"
)

# %% [markdown]
# ## 5. Benchmark 3: GroupBy + Aggregate

# %%
print("\n── GroupBy district × treatment_arm → mean income ──")

pandas_groupby_time, _ = benchmark(
    lambda: (
        df_pd
        .groupby(["district", "treatment_arm"])["monthly_income_usd"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    ),
    "pandas"
)

polars_groupby_time, _ = benchmark(
    lambda: (
        df_pl
        .group_by(["district", "treatment_arm"])
        .agg([
            pl.col("monthly_income_usd").mean().alias("mean"),
            pl.col("monthly_income_usd").median().alias("median"),
            pl.col("monthly_income_usd").std().alias("std"),
            pl.col("monthly_income_usd").count().alias("count"),
        ])
    ),
    "polars"
)

# %% [markdown]
# ## 6. Benchmark 4: String Operations
#
# Extract the numeric part from `household_id` (e.g. "HH_0012345" → 12345)
# and clean/uppercase the `notes` column.

# %%
print("\n── String ops (extract ID number + uppercase notes) ──")

pandas_str_time, _ = benchmark(
    lambda: df_pd.assign(
        hh_num=df_pd["household_id"].str.extract(r"(\d+)")[0].astype(int),
        notes_clean=df_pd["notes"].str.strip().str.upper(),
    ),
    "pandas"
)

polars_str_time, _ = benchmark(
    lambda: df_pl.with_columns([
        pl.col("household_id").str.extract(r"(\d+)", 1).cast(pl.Int64).alias("hh_num"),
        pl.col("notes").str.strip_chars().str.to_uppercase().alias("notes_clean"),
    ]),
    "polars"
)

# %% [markdown]
# ## 7. Benchmark 5: Join (simulate merging in a lookup table)
#
# Merge in a district-level metadata table (region, population density, etc.)

# %%
print("\n── Join (district metadata lookup) ──")

DISTRICT_META_PD = pd.DataFrame({
    "district": DISTRICTS,
    # Province each district belongs to
    "province": ["Kigali City", "Kigali City", "Kigali City",
                 "Northern",    "Northern",
                 "Southern",    "Southern",
                 "Eastern",     "Eastern",
                 "Western"],
    # Approximate population density (persons/km²) — Kigali is very dense
    "pop_density_km2": [2800, 3100, 5200, 180, 140, 220, 160, 35, 190, 310],
    # Approximate urban population share
    "urban_pct":       [0.82, 0.87, 0.91, 0.28, 0.22, 0.31, 0.19, 0.12, 0.25, 0.41],
})

DISTRICT_META_PL = pl.from_pandas(DISTRICT_META_PD)

pandas_join_time, _ = benchmark(
    lambda: df_pd.merge(DISTRICT_META_PD, on="district", how="left"),
    "pandas"
)

polars_join_time, _ = benchmark(
    lambda: df_pl.join(DISTRICT_META_PL, on="district", how="left"),
    "polars"
)

# %% [markdown]
# ## 8. Visualize Results

# %%
operations = ["Read", "Filter", "GroupBy\n+ Agg", "String\nOps", "Join"]
pandas_times = [pandas_read_time, pandas_filter_time, pandas_groupby_time,
                pandas_str_time,  pandas_join_time]
polars_times = [polars_read_time, polars_filter_time, polars_groupby_time,
                polars_str_time,  polars_join_time]

PANDAS_COLOR = "#E05C5C"
POLARS_COLOR = "#3A86FF"

# ── Panel 1: grouped bars ──────────────────────────────────────────────────────
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=[
        "Operation Time (seconds) — lower is better",
        "Polars Speedup vs Pandas (×) — higher is better",
    ],
    column_widths=[0.62, 0.38],
    horizontal_spacing=0.1,
)

fig.add_trace(go.Bar(
    name="pandas",
    x=operations,
    y=pandas_times,
    marker_color=PANDAS_COLOR,
    text=[f"{v:.3f}s" for v in pandas_times],
    textposition="outside",
), row=1, col=1)

fig.add_trace(go.Bar(
    name="polars",
    x=operations,
    y=polars_times,
    marker_color=POLARS_COLOR,
    text=[f"{v:.3f}s" for v in polars_times],
    textposition="outside",
), row=1, col=1)

speedups = [p / q for p, q in zip(pandas_times, polars_times)]
speedup_colors = [POLARS_COLOR if s >= 1 else PANDAS_COLOR for s in speedups]

fig.add_trace(go.Bar(
    name="Speedup (×)",
    x=operations,
    y=speedups,
    marker_color=speedup_colors,
    text=[f"{s:.1f}×" for s in speedups],
    textposition="outside",
    showlegend=False,
), row=1, col=2)

# Reference line at 1× (parity)
fig.add_hline(
    y=1.0, line_dash="dot", line_color="gray",
    annotation_text="1× (parity)", annotation_position="bottom right",
    row=1, col=2,
)

fig.update_layout(
    title=dict(
        text=f"<b>Polars vs Pandas</b> — {N_ROWS:,} household survey rows",
        font=dict(size=18),
    ),
    barmode="group",
    height=520,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=100, b=60),
)
fig.update_yaxes(gridcolor="#eee")

fig.show()
fig.write_html("demo2_polars_vs_pandas_results.html")
print("Chart saved to demo2_polars_vs_pandas_results.html")

# %% [markdown]
# ## Summary
#
# | Operation | Pandas | Polars | Speedup |
# |-----------|--------|--------|---------|
# | Read | — | — | see chart |
# | Filter | — | — | see chart |
# | GroupBy + Agg | — | — | see chart |
# | String Ops | — | — | see chart |
# | Join | — | — | see chart |
#
# ### Why Polars is faster:
# - **Rust core** — compiled, zero-copy memory model
# - **Lazy evaluation** — builds a query plan and optimises before executing
# - **Multi-threaded by default** — uses all CPU cores automatically
# - **Apache Arrow memory layout** — the same columnar format as Parquet,
#   so reading Parquet into Polars involves minimal copying
# - **Expressive API** — method chains are translated into optimised execution plans,
#   not intermediate DataFrames
