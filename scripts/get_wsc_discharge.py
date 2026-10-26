#!/usr/bin/env python3
"""Download the full historic daily discharge record for a Canadian station.

Uses the MSC GeoMet collection ``hydrometric-daily-mean``
(https://api.weather.gc.ca/). Discharge is the HYDAT daily mean in m3/s.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

API = "https://api.weather.gc.ca/collections/hydrometric-daily-mean/items"
PAGE_SIZE = 10000
COLUMNS = [
    "date",
    "station_number",
    "station_name",
    "discharge_cms",
    "discharge_symbol",
]


def fetch_wsc_daily_means(station: str) -> list[dict]:
    """Return every daily-mean feature for ``station``, oldest first."""
    query = urlencode(
        {
            "STATION_NUMBER": station,
            "limit": PAGE_SIZE,
            "sortby": "DATE",
            "f": "json",
        }
    )
    url = f"{API}?{query}"
    rows: list[dict] = []
    expected = None

    while url:
        with urlopen(url, timeout=120) as response:
            payload = json.load(response)
        if expected is None:
            expected = payload.get("numberMatched", 0)
            if not payload.get("features"):
                raise SystemExit(f"No daily-mean data found for station {station}.")
        for feature in payload["features"]:
            props = feature["properties"]
            discharge = props["DISCHARGE"]
            rows.append(
                {
                    "date": props["DATE"],
                    "station_number": props["STATION_NUMBER"],
                    "station_name": props["STATION_NAME"],
                    # GeoMet returns HYDAT discharge as float32.
                    "discharge_cms": None if discharge is None else round(discharge, 3),
                    "discharge_symbol": props["DISCHARGE_SYMBOL_EN"],
                }
            )
        url = next(
            (link["href"] for link in payload.get("links", []) if link.get("rel") == "next"),
            None,
        )

    if expected is not None and len(rows) != expected:
        raise SystemExit(
            f"Incomplete download for {station}: got {len(rows)} of {expected} days."
        )
    return rows


def write_wsc_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the complete historic daily discharge record "
        "for a Canadian hydrometric station (MSC GeoMet / HYDAT)."
    )
    parser.add_argument(
        "station",
        nargs="?",
        default="09BA001",
        help="Water Survey of Canada station number (default: 09BA001).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <station>_daily_discharge.csv).",
    )
    args = parser.parse_args()

    station = args.station.strip().upper()
    output = args.output or Path(f"{station}_daily_discharge.csv")
    rows = fetch_wsc_daily_means(station)
    write_wsc_csv(rows, output)

    with_discharge = sum(row["discharge_cms"] is not None for row in rows)
    print(
        f"Wrote {len(rows)} days ({with_discharge} with discharge) "
        f"for {rows[0]['station_name']} ({station}) "
        f"from {rows[0]['date']} to {rows[-1]['date']} -> {output}"
    )


if __name__ == "__main__":
    main()
