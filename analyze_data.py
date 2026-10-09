"""Reproducible analysis of the supplied AI content dataset."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).resolve().parent
PERCENT = ["AI Adoption Rate (%)", "Job Loss Due to AI (%)",
           "Revenue Increase Due to AI (%)", "Human-AI Collaboration Rate (%)",
           "Consumer Trust in AI (%)", "Market Share of AI Companies (%)"]
VOLUME = "AI-Generated Content Volume (TBs per year)"
NUMERIC = PERCENT + [VOLUME]
KEY = ["Country", "Year", "Industry"]
EXPECTED = KEY + NUMERIC + ["Top AI Tools Used", "Regulation Status"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(EXPECTED) - set(df.columns)
    if missing or df.empty:
        raise ValueError(f"Empty data or missing columns: {sorted(missing)}")
    for col in NUMERIC + ["Year"]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    return df


def summarize(df: pd.DataFrame) -> dict:
    pairs = []
    for left, right in combinations(NUMERIC, 2):
        clean = df[[left, right]].dropna()
        if len(clean) < 4 or clean[left].nunique() < 2 or clean[right].nunique() < 2:
            continue
        r, p = stats.pearsonr(clean[left], clean[right])
        z = np.arctanh(r)
        delta = stats.norm.ppf(.975) / np.sqrt(len(clean) - 3)
        pairs.append({"left": left, "right": right, "n": len(clean), "r": float(r),
                      "p": float(p), "ci95": [float(v) for v in np.tanh([z-delta, z+delta])]})
    if pairs:
        for pair, q in zip(pairs, stats.false_discovery_control([x["p"] for x in pairs], method="bh")):
            pair["q_bh"] = float(q)
    trends = {}
    for col in [PERCENT[0], PERCENT[2]]:
        clean = df[["Year", col]].dropna()
        fit = stats.linregress(clean["Year"], clean[col])
        trends[col] = {"slope_percentage_points_per_year": float(fit.slope), "p": float(fit.pvalue)}
    return {
        "source": "data/Global_AI_Content_Impact_Dataset.csv", "rows": len(df),
        "columns": len(df.columns), "missing_cells": int(df.isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_country_year_industry_rows": int(df.duplicated(KEY).sum()),
        "countries": int(df.Country.nunique()), "industries": int(df.Industry.nunique()),
        "years": {str(k): int(v) for k, v in df.Year.value_counts().sort_index().items()},
        "numeric_summary": {c: {"mean": float(df[c].mean()), "median": float(df[c].median()),
                                "std": float(df[c].std()), "min": float(df[c].min()),
                                "max": float(df[c].max())} for c in NUMERIC},
        "year_means": {str(year): {"n": len(group),
            "adoption_mean": float(group[PERCENT[0]].mean()),
            "revenue_mean": float(group[PERCENT[2]].mean())}
            for year, group in df.groupby("Year")},
        "trends": trends, "correlations": pairs,
    }


def create_figures(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep", font_scale=.95)
    blue, orange = "#245E7A", "#C47C3B"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, col, title in zip(axes, PERCENT[:3], ["AI adoption", "Job loss", "Revenue increase"]):
        sns.histplot(df[col].dropna(), bins=12, color=blue, ax=ax)
        ax.axvline(df[col].mean(), color=orange, lw=2, label=f"Mean {df[col].mean():.1f}%")
        ax.set(title=title, xlabel="Reported rate (%)", ylabel="Observations (count)")
        ax.legend(frameon=False)
    fig.suptitle(f"Distribution of reported AI impact measures | n = {len(df)}")
    fig.savefig(out / "01_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, col, title, color in zip(axes, [PERCENT[0], PERCENT[2]],
                                      ["AI adoption by year", "Revenue increase by year"], [blue, orange]):
        grouped = df.groupby("Year")[col].agg(["mean", "std", "count"])
        ci = stats.t.ppf(.975, grouped["count"]-1) * grouped["std"] / np.sqrt(grouped["count"])
        ax.errorbar(grouped.index, grouped["mean"], yerr=ci, color=color, marker="o", capsize=4)
        ax.set(title=title, xlabel="Year", ylabel="Mean reported rate (%)", xticks=grouped.index)
        ax.set_ylim(bottom=0)
        for year, row in grouped.iterrows():
            ax.annotate(f"n={int(row['count'])}", (year, row["mean"]), xytext=(0, 11),
                        textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("Annual sample means with 95% t intervals (unweighted rows)")
    fig.savefig(out / "02_year_means.png", dpi=180)
    plt.close(fig)

    labels = {PERCENT[0]: "Adoption", VOLUME: "Content volume", PERCENT[1]: "Job loss",
              PERCENT[2]: "Revenue", PERCENT[3]: "Collaboration", PERCENT[4]: "Trust",
              PERCENT[5]: "Market share"}
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    sns.heatmap(df[list(labels)].rename(columns=labels).corr(), vmin=-1, vmax=1, center=0,
                cmap="vlag", annot=True, fmt=".2f", square=True, linewidths=.4, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title(f"Pairwise Pearson correlations | numeric measures, n = {len(df)}")
    fig.savefig(out / "03_correlations.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    sns.regplot(data=df, x=PERCENT[0], y=PERCENT[2], ci=None,
                scatter_kws={"alpha": .55, "s": 28}, line_kws={"color": orange}, ax=ax)
    ax.set(title=f"Adoption versus reported revenue increase | n = {len(df)}",
           xlabel="AI adoption rate (%)", ylabel="Revenue increase due to AI (%)")
    fig.savefig(out / "04_adoption_revenue.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/Global_AI_Content_Impact_Dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    df = load_data(args.data)
    result = summarize(df)
    create_figures(df, args.output)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Analyzed {result['rows']} rows; wrote summary and 4 figures to {args.output.resolve()}")


if __name__ == "__main__":
    main()
