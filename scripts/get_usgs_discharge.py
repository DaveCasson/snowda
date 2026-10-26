#!/usr/bin/env python3
"""Download the full historic daily discharge record for a USGS station.

Uses the NWIS daily-values service (https://waterservices.usgs.gov/nwis/dv/).
Discharge is parameter 00060, statistic 00003 (daily mean), converted from
ft3/s to m3/s.

The default station is Chena River near Two Rivers, Alaska (15493000).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

API = "https://waterservices.usgs.gov/nwis/dv/"
CFS_TO_CMS = 0.028316846592
NO_DATA = -999999.0
COLUMNS = [
    "date",
    "station_number",
    "station_name",
    "discharge_cms",
    "discharge_symbol",
]


def fetch_usgs_daily_means(site: str) -> list[dict]:
    """Return every published daily mean discharge for ``site``, oldest first."""
    query = urlencode(
        {
            "format": "json",
            "sites": site,
            "parameterCd": "00060",
            "statCd": "00003",
            "startDT": "1800-01-01",
        }
    )
    with urlopen(f"{API}?{query}", timeout=120) as response:
        payload = json.load(response)

    series = payload["value"]["timeSeries"]
    if not series or not series[0]["values"][0]["value"]:
        raise SystemExit(f"No daily discharge found for USGS site {site}.")

    chosen = series[0]
    station_name = chosen["sourceInfo"]["siteName"]
    station_number = chosen["sourceInfo"]["siteCode"][0]["value"]
    rows = []
    for sample in chosen["values"][0]["value"]:
        raw = sample["value"]
        discharge = None
        if raw not in ("", None):
            cfs = float(raw)
            if cfs > NO_DATA:
                discharge = round(cfs * CFS_TO_CMS, 3)
        symbol = ",".join(sample.get("qualifiers") or []) or None
        rows.append(
            {
                "date": sample["dateTime"][:10],
                "station_number": station_number,
                "station_name": station_name,
                "discharge_cms": discharge,
                "discharge_symbol": symbol,
            }
        )
    return rows


def write_usgs_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the complete historic daily discharge record "
        "for a USGS station. Default is Chena River near Two Rivers, AK."
    )
    parser.add_argument(
        "site",
        nargs="?",
        default="15493000",
        help="USGS site number (default: 15493000, Chena River near Two Rivers).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <site>_daily_discharge.csv).",
    )
    args = parser.parse_args()

    site = args.site.strip()
    output = args.output or Path(f"{site}_daily_discharge.csv")
    rows = fetch_usgs_daily_means(site)
    write_usgs_csv(rows, output)

    with_discharge = sum(row["discharge_cms"] is not None for row in rows)
    print(
        f"Wrote {len(rows)} days ({with_discharge} with discharge) "
        f"for {rows[0]['station_name']} ({site}) "
        f"from {rows[0]['date']} to {rows[-1]['date']} -> {output}"
    )


if __name__ == "__main__":
    main()
