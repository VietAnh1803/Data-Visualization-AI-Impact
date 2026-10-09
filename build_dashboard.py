"""Embed the current CSV and report figures in the offline HTML dashboard."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from analyze_data import load_data


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Global_AI_Content_Impact_Dataset.csv"
DASHBOARD = ROOT / "reports" / "dashboard.html"
START = "<!-- DATA_START -->"
END = "<!-- DATA_END -->"
FIGURES_START = "<!-- FIGURES_START -->"
FIGURES_END = "<!-- FIGURES_END -->"
FIGURE_STEMS = ("01_distributions", "02_year_means", "03_correlations",
                "04_adoption_revenue", "05_industry_means")
# The editorial interpretation in dashboard.html was reviewed for this CSV version.
NARRATIVE_SHA256 = "53b52d3d5fef30db9f575852ddcf6cbe347e049aef4de145f689f456fbf7a514"


def main() -> None:
    source = DASHBOARD.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if len(pattern.findall(source)) != 1:
        raise ValueError("dashboard.html must contain exactly one embedded data block")

    sha256 = hashlib.sha256(DATA.read_bytes()).hexdigest()
    if sha256 != NARRATIVE_SHA256:
        raise ValueError("CSV changed; review the dashboard narrative before rebuilding")
    summary = json.loads((ROOT / "reports/figures/summary.json").read_text(encoding="utf-8"))
    if summary.get("source_sha256") != sha256:
        raise ValueError("Figures do not match the CSV; run analyze_data.py first")
    payload = {"rows": load_data(DATA).to_dict(orient="records"), "sha256": sha256}
    # Prevent CSV text from closing the JSON script element.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    block = f'{START}\n<script id="dataset" type="application/json">{encoded}</script>\n{END}'
    source = pattern.sub(lambda _: block, source)

    figures_pattern = re.compile(re.escape(FIGURES_START) + r".*?" + re.escape(FIGURES_END), re.DOTALL)
    if len(figures_pattern.findall(source)) != 1:
        raise ValueError("dashboard.html must contain exactly one embedded figures block")
    figures = {}
    for stem in FIGURE_STEMS:
        for suffix in ("", "_vi"):
            filename = f"{stem}{suffix}.png"
            image = (ROOT / "reports/figures" / filename).read_bytes()
            figures[stem + suffix] = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
    figure_json = json.dumps(figures, separators=(",", ":"))
    figures_block = (f'{FIGURES_START}\n<script id="figures" type="application/json">'
                     f'{figure_json}</script>\n{FIGURES_END}')
    source = figures_pattern.sub(lambda _: figures_block, source)
    DASHBOARD.write_text(source, encoding="utf-8")
    print(f"Embedded {len(payload['rows'])} rows and {len(figures)} figure variants in {DASHBOARD}")


if __name__ == "__main__":
    main()
