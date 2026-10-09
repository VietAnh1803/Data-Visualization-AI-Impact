"""Embed the current CSV in the offline, single-file HTML dashboard."""

from __future__ import annotations

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


def main() -> None:
    source = DASHBOARD.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.DOTALL)
    if len(pattern.findall(source)) != 1:
        raise ValueError("dashboard.html must contain exactly one embedded data block")

    payload = {
        "rows": load_data(DATA).to_dict(orient="records"),
        "sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
    }
    # Prevent CSV text from closing the JSON script element.
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    block = f'{START}\n<script id="dataset" type="application/json">{encoded}</script>\n{END}'
    DASHBOARD.write_text(pattern.sub(lambda _: block, source), encoding="utf-8")
    print(f"Embedded {len(payload['rows'])} rows in {DASHBOARD}")


if __name__ == "__main__":
    main()
