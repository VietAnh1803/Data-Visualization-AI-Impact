"""Reproducible analysis of the supplied AI content dataset."""

from __future__ import annotations

import argparse
import hashlib
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
        "industry_means": {industry: {"n": len(group),
            "adoption_mean": float(group[PERCENT[0]].mean()),
            "revenue_mean": float(group[PERCENT[2]].mean())}
            for industry, group in df.groupby("Industry")},
        "trends": trends, "correlations": pairs,
    }


def create_figures(df: pd.DataFrame, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=.95)
    blue, orange, red = "#08726f", "#a65018", "#a5414d"
    for lang in ("en", "vi"):
        vi = lang == "vi"
        suffix = "_vi" if vi else ""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        titles = (["Ứng dụng AI", "Mất việc", "Tăng doanh thu"] if vi else
                  ["AI adoption", "Job loss", "Revenue increase"])
        for ax, col, title, color in zip(axes, PERCENT[:3], titles, [blue, red, orange]):
            sns.histplot(df[col].dropna(), bins=12, color=color, ax=ax)
            ax.axvline(df[col].mean(), color="#18313a", lw=2,
                       label=("Trung bình" if vi else "Mean") + f" {df[col].mean():.1f}%")
            ax.set(title=title, xlabel="Tỷ lệ ghi nhận (%)" if vi else "Reported rate (%)",
                   ylabel="Số dòng" if vi else "Records")
            ax.legend(frameon=False)
        fig.suptitle(("Phân bố ba chỉ số được ghi nhận" if vi else
                      "Distributions of three reported measures") + f" | n = {len(df)}")
        fig.savefig(out / f"01_distributions{suffix}.png", dpi=180)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        titles = (["Ứng dụng AI theo năm", "Tăng doanh thu theo năm"] if vi else
                  ["AI adoption by year", "Revenue increase by year"])
        for ax, col, title, color in zip(axes, [PERCENT[0], PERCENT[2]], titles, [blue, orange]):
            grouped = df.groupby("Year")[col].agg(["mean", "std", "count"])
            ci = stats.t.ppf(.975, grouped["count"]-1) * grouped["std"] / np.sqrt(grouped["count"])
            ax.errorbar(grouped.index, grouped["mean"], yerr=ci, color=color, marker="o", capsize=4)
            ax.set(title=title, xlabel="Năm" if vi else "Year",
                   ylabel="Trung bình (%)" if vi else "Mean reported rate (%)", xticks=grouped.index)
            ax.set_ylim(bottom=0)
            for year, row in grouped.iterrows():
                ax.annotate(f"n={int(row['count'])}", (year, row["mean"]), xytext=(0, 11),
                            textcoords="offset points", ha="center", fontsize=8)
        fig.suptitle("Trung bình năm và khoảng tin cậy 95%" if vi else
                     "Annual sample means with 95% t intervals")
        fig.savefig(out / f"02_year_means{suffix}.png", dpi=180)
        plt.close(fig)

        labels = {PERCENT[0]: "Ứng dụng" if vi else "Adoption",
                  VOLUME: "Nội dung" if vi else "Content volume",
                  PERCENT[1]: "Mất việc" if vi else "Job loss",
                  PERCENT[2]: "Doanh thu" if vi else "Revenue",
                  PERCENT[3]: "Hợp tác" if vi else "Collaboration",
                  PERCENT[4]: "Niềm tin" if vi else "Trust",
                  PERCENT[5]: "Thị phần" if vi else "Market share"}
        correlations = df[list(labels)].rename(columns=labels).corr()
        mask = np.triu(np.ones(correlations.shape, dtype=bool))
        fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
        sns.heatmap(correlations, mask=mask, vmin=-.25, vmax=.25, center=0,
                    cmap="RdBu_r", annot=True, fmt=".3f", square=True,
                    linewidths=.7, linecolor="#ffffff", ax=ax,
                    cbar_kws={"label": "Pearson r", "shrink": .75})
        ax.grid(False)
        ax.set_title(("Tương quan giữa bảy chỉ số" if vi else
                      "Correlations among seven measures") + f" | n = {len(df)}")
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)
        fig.savefig(out / f"03_correlations{suffix}.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        sns.regplot(data=df, x=PERCENT[0], y=PERCENT[2], ci=None,
                    scatter_kws={"alpha": .55, "s": 28, "color": blue},
                    line_kws={"color": orange}, ax=ax)
        ax.set(title=("Ứng dụng AI và tăng doanh thu" if vi else
                      "Adoption versus reported revenue increase") + f" | n = {len(df)}",
               xlabel="Ứng dụng AI (%)" if vi else "AI adoption rate (%)",
               ylabel="Tăng doanh thu được ghi nhận (%)" if vi else "Reported revenue increase (%)")
        fig.savefig(out / f"04_adoption_revenue{suffix}.png", dpi=180)
        plt.close(fig)

        industry = df.groupby("Industry").agg(
            adoption=(PERCENT[0], "mean"), revenue=(PERCENT[2], "mean"), n=("Industry", "size"))
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        ax.axvline(df[PERCENT[0]].mean(), color=blue, ls="--", lw=1, alpha=.6)
        ax.axhline(df[PERCENT[2]].mean(), color=orange, ls="--", lw=1, alpha=.6)
        offsets = {"Manufacturing": (8, 13), "Legal": (8, -11),
                   "Education": (8, 13), "Healthcare": (8, -11),
                   "Marketing": (-82, -12), "Finance": (8, -11)}
        for name, row in industry.iterrows():
            color = orange if name == "Gaming" else blue if name == "Media" else "#8ca4a8"
            ax.scatter(row.adoption, row.revenue, s=row.n * 17, color=color,
                       alpha=.85, edgecolor="white", linewidth=1, zorder=3)
            ax.annotate(f"{name} (n={int(row.n)})", (row.adoption, row.revenue),
                        xytext=offsets.get(name, (6, 5)), textcoords="offset points", fontsize=8)
        ax.set(title="Trung bình theo ngành: ứng dụng và doanh thu" if vi else
               "Industry means: adoption and revenue increase",
               xlabel="Ứng dụng AI trung bình (%)" if vi else "Mean AI adoption (%)",
               ylabel="Tăng doanh thu trung bình (%)" if vi else "Mean revenue increase (%)")
        ax.margins(x=.18, y=.2)
        fig.savefig(out / f"05_industry_means{suffix}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/Global_AI_Content_Impact_Dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    df = load_data(args.data)
    result = summarize(df)
    result["source_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    create_figures(df, args.output)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Analyzed {result['rows']} rows; wrote summary and 10 figure variants to {args.output.resolve()}")


if __name__ == "__main__":
    main()
