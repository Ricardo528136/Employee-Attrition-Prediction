# Imports & Settings
import os
import sys
import json
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import plotly.express as px
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

from scipy import stats
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option('display.max_columns', 100)

# Ensure the script is run from the correct directory -- EDIT THESE PATHS
# Path to the CSV file
CSV_PATH = Path(data\WA_Fn-UseC_-HR-Employee-Attrition.csv)
if not CSV_PATH.exists():
    print(f"CSV file not found at {CSV_PATH}. Please check the path.")
    sys.exit(1)

# EDA outputs (tables, cleaned data, plots)
OUTPUT_DIR = Path("./outputs")
FIG_DIR = OUTPUT_DIR / "figures"
TABLE_DIR = OUTPUT_DIR / "tables"
CLEAN_DIR = OUTPUT_DIR / "cleaned"

for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, CLEAN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Random seed for reproducibility
RANDOM_SEED = 7

# Target column
TARGET_COL = "Attrition"

# Optional ID-like columns that should not be used in modeling
ID_COLS = [
    "EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"
]

# Categorical columns, leave empty to auto-detect
FORCE_CATEGORICAL = []

# Numerical columns, leave empty to auto-detect
FORCE_NUMERIC = []

# Utility functions

def load_data(csv_path: Path) -> pd.DataFrame:
    """Load data from a CSV file."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}.")
    df = pd.read_csv(csv_path)
    return df

def basic_overview(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Generate quick shape, dtypes, head, and target distribution."""
    print("\nBasic Overview:")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nData Types:\n", df.dtypes)
    print("\nHead:\n", df.head())
    if target_col in df.columns:
        print("\nTarget distribution (counts & percentages):")
        counts = df[target_col].value_counts(dropna=False)
        perc = counts / len(df) * 100
        display(pd.DataFrame({"count": counts, "percentage": perc.round(2)}))

def find_constant_columns(df: pd.DataFrame):
    """Find columns with a single unique value."""
    const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    return const_cols

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute missing value counts and percentages for each column."""
    ms = df.isna().sum().to_frame(name='missing_count')
    ms['missing_pct'] = ms['missing_count'] / len(df) * 100
    ms = ms.sort_values(by='missing_pct', ascending=False)
    return ms

def infer_variable_types(df: pd.DataFrame, force_cat=None, force_num=None, id_cols=None, max_cat_unique=20):
    """Heuristically classify columns as categorical vs numeric.
    - Explicit overrides respected first.
    - Low-unique-count numerics (<= max_cat_unique) treated as categorical.
    - Object dtype => categorical.
    - Exclude id_cols from both (returned separately).
    Returns: cat_cols, num_cols, id_cols_final
    """
    force_cat = set(force_cat or [])
    force_num = set(force_num or [])
    id_cols = set(id_cols or [])

    cat_cols = []
    num_cols = []

    for col in df.columns:
        if col in id_cols:
            continue
        if col in force_cat:
            cat_cols.append(col)
            continue
        if col in force_num:
            num_cols.append(col)
            continue
        dtype = df[col].dtype
        nunique = df[col].nunique(dropna=False)
        if dtype == 'object' or dtype.name == 'category':
            cat_cols.append(col)
        elif np.issubdtype(dtype, np.number):
            if nunique <= max_cat_unique:
                cat_cols.append(col)
            else:
                num_cols.append(col)
        else:
            # If dtype is not recognized, default to categorical
            cat_cols.append(col)
    
    return cat_cols, num_cols, list(id_cols)

def summarize_categorical(df: pd.DataFrame, cat_cols, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Return tidy table of categorical counts + attrition rate per category."""
    records = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False)
        for level, count in vc.items():
            rec = {
                'variable': col,
                'level': level,
                'count': count,
                'percentage': count / len(df) * 100
            }
            if target_col in df.columns:
                mask = df[col] == level
                rate = df.loc[mask, target_col].eq('Yes').mean() * 100
                rec['attrition_rate'] = rate
            records.append(rec)
    out = pd.DataFrame.from_records(records)
    return out

def summarize_numeric(df: pd.DataFrame, num_cols, target_col: str = TARGET_COL) -> pd.DataFrame:
    """Return tidy table of numeric stats + attrition"""
    summaries = []
    for col in num_cols:
        s = df[col]
        summary = {
            'variable': col,
            'n': s.count(),
            'mean': s.mean(),
            'std': s.std(),
            'min': s.min(),
            '25%': s.quantile(0.25),
            '50%': s.median(),
            '75%': s.quantile(0.75),
            'max': s.max(),
        }
        if target_col in df.columns:
            grp = df.groupby(target_col)[col].agg(['mean', 'median', 'std', 'count'])
            for lvl in grp.index:
                summary[f'{lvl}_mean'] = grp.loc[lvl, 'mean']
                summary[f'{lvl}_median'] = grp.loc[lvl, 'median']
                summary[f'{lvl}_std'] = grp.loc[lvl, 'std']
                summary[f'{lvl}_count'] = grp.loc[lvl, 'count']
        summaries.append(summary)
    return pd.DataFrame(summaries)

def chi_square_of_cats(df: pd.DataFrame, cat_col: str, target_col: str = TARGET_COL):
    """Perform Chi-Square test of independence between a categorical column and the target."""
    contingency = pd.crosstab(df[cat_col], df[target_col], dropna=False)
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return {
        "variable": cat_col,
        "chi2": chi2,
        "pvalue": p,
        "dof": dof
    }

def encode_target_binary(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.Series:
    """Return binary target: 1 if Yes, 0 if No (case-insensitive)."""
    return df[target_col].str.strip().str.lower().eq('yes').astype(int)

# Plotting Helpers

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)

def plot_target_distribution(df: pd.DataFrame, target_col: str = TARGET_COL, save=True):
    """Plot target distribution as a count plot."""
    plt.figure(figsize=(4,4))
    ax = sns.countplot(data=df, x=target_col, order=df[target_col].value_counts().index)
    plt.title("Attrition Counts")
    plt.ylabel("Count")
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / f"target_distribution_{target_col}.png", dpi=150)
    plt.show()

def plot_numeric_by_target(df: pd.DataFrame, num_cols, target_col: str = TARGET_COL, kind="box", save=True):
    """Plot numeric columns by target using box or violin plots."""
    for col in num_cols:
        plt.figure(figsize=(5,4))
        if kind == "violin":
            sns.violinplot(data=df, x=target_col, y=col, inner="quartile")
        elif kind == "hist":
            sns.histplot(data=df, x=col, hue=target_col, bins=30, kde=True, stat="density", common_norm=False)
        else:
            sns.boxplot(data=df, x=target_col, y=col)
        plt.title(f"{col} by {target_col}")
        plt.tight_layout()
        if save:
            plt.savefig(FIG_DIR / f"{col}_by_{target_col}_{kind}.png", dpi=150)
        plt.show()
    
def plot_cat_attrition_rate(df: pd.DataFrame, cat_cols, target_col: str = TARGET_COL, top_n=20, save=True):
    """Plot attrition rate by categorical columns. For large-cardinality columns, only show top N levels."""
    for col in cat_cols:
        if col == target_col:
            continue
        stats_df = df.groupby(col)[target_col].apply(lambda s: s.eq('Yes').mean()*100).reset_index(name="attrition_rate_pct")
        counts = df[col].value_counts()
        if len(stats_df) > top_n:
            stats_df = stats_df[stats_df[col].isin(top_levels)]
        plt.figure(figsize=(6, 4))
        sns.barplot(data=stats_df, x=col, y="attrition_rate_pct", orient="h")
        plt.title(f"Attrition Rate by {col}")
        plt.xlabel("Attrition Rate (%)")
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save:
            plt.savefig(FIG_DIR / f"{col}_attrition_rate.png", dpi=150)
        plt.show()
    
def plot_correlation_heatmap(df: pd.DataFrame, num_cols, target_col: str = TARGET_COL, save=True):
    """Plot correlation heatmap for numeric columns"""
    df_corr = df[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    if save:
        plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.show()

def plot_interactive_scatter(df: pd.DataFrame, fields, color=TARGET_COL):
    """Plot an interactive scatter plot using Plotly."""
    if not _HAS_PLOTLY:
        print("Plotly is not installed. Skipping interactive plots.")
        return
    fig = px.scatter(df, dimensions=fields, color=color, title='Scatter Matrix')
    fig.update_traces(diagonal_visible=False)
    fig.show()

# Save Helpers

def save_table(df: pd.DataFrame, name: str):
    """Save a DataFrame."""
    csv_path = TABLE_DIR / f"{name}.csv"
    xlsx_path = TABLE_DIR / f"{name}.xlsx"
    df.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as xlw:
        df.to_excel(xlw, index=False, sheet_name="Sheet1")
    print(f"Saved {name} to {csv_path} and {xlsx_path}")

# Main EDA Function

def run_eda():
    df_raw = load_data(CSV_PATH)

    # Basic Overview
    basic_overview(df_raw, target_col=TARGET_COL)

    # Find constant columns
    const_cols = find_constant_columns(df_raw)
    if const_cols:
        print(f"\nConstant columns (single unique value, likely not useful): {const_cols}")
    
    # Missing values summary
    missing_df = missing_summary(df_raw)
    print("\nMissing Values Summary:")
    display(missing_df)
    save_table(missing_df.reset_index().rename(columns={'index': 'variable'}), "missing_summary")

    # Infer variable types
    cat_cols, num_cols, id_cols = infer_variable_types(
        df_raw, 
        force_cat=FORCE_CATEGORICAL, 
        force_num=FORCE_NUMERIC, 
        id_cols=ID_COLS
        
    )