#!/usr/bin/env python3
"""Download Chena-area SWE and snow depth; write tidy CSV with values in mm.

Call download_snow from a notebook, or run this file as a script.
Python 3.10+. See README.txt for installation, schema, and source limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

AWDB = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"
NOHRSC = "https://www.nohrsc.noaa.gov"
HISTORY = NOHRSC + "/snowsurvey/historical.html"
LOG = logging.getLogger("snow")


@dataclass(frozen=True)
class Station:
    triplet: str
    name: str
    ids: tuple[str, ...]


# Manual courses are separate records from the automated instruments.
COURSES = [
    Station("46Q01:AK:SNOW", "Mt. Ryan", ("NRCS-AK-46Q01",)),
    Station("45Q02:AK:SNOW", "Monument Creek", ("NRCS-AK-45Q02",)),
    Station("46P01:AK:SNOW", "Munson Ridge", ("NRCS-AK-46P01",)),
    Station("44Q01:AK:SNOW", "Upper Chena", ("NRCS-AK-44Q01",)),
    Station("45P03:AK:SNOW", "Teuchet Creek", ("NRCS-AK-45P03",)),
]
SNOTEL = [
    Station("948:AK:SNTL", "Mt. Ryan", ("SNOTEL-948",)),
    Station("949:AK:SNTL", "Monument Creek", ("SNOTEL-949",)),
    Station("950:AK:SNTL", "Munson Ridge", ("SNOTEL-950",)),
    Station("951:AK:SNTL", "Teuchet Creek", ("SNOTEL-951",)),
    # 44Q07/44Q07S identifies this automated site, not another snow course.
    Station("952:AK:SNTL", "Upper Chena", ("SNOTEL-952", "NRCS-AK-44Q07")),
]
FLIGHT_LINES = ("AK234", "AK240", "AK248", "AK233", "AK239", "AK241", "AK242")
VARIABLES = {"WTEQ": "swe", "SNWD": "snow_depth"}
SOURCES = ("snotel", "nrcs", "nohrsc")
COLUMNS = [
    "date", "source", "station_id", "station_name", "requested_ids", "variable",
    "value_mm", "status", "lower_bound_mm", "qc_flag", "qa_flag",
    "reporting_year", "reporting_month", "month_part", "source_unit",
    "latitude", "longitude", "source_url",
]


def to_mm(value, unit):
    """Convert only known length units; never guess units or replace missing with 0."""
    factors = {"in": 25.4, "inch": 25.4, "inches": 25.4,
               "cm": 10.0, "mm": 1.0, "m": 1000.0}
    if value is None or pd.isna(value):
        return None, "missing"
    key = str(unit).strip().lower()
    if key not in factors:
        raise ValueError(f"Unsupported length unit: {unit!r}")
    number = float(value)
    if number < 0:
        return None, "invalid_or_missing"
    return number * factors[key], "available"


class Client:
    """Retry transient HTTP errors and keep original responses for reproducibility."""
    def __init__(self, cache: Path, refresh=False):
        self.cache, self.refresh = cache, refresh
        cache.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers["User-Agent"] = "snow-data-downloader/1.0"

    def get(self, url, params=None):
        full_url = requests.Request("GET", url, params=params).prepare().url
        key = hashlib.sha256(full_url.encode()).hexdigest()
        path = self.cache / (key + ".txt")
        if path.exists() and not self.refresh:
            return path.read_text(), full_url
        response = self.session.get(full_url, timeout=(15, 90))
        response.raise_for_status()
        path.write_text(response.text)
        (self.cache / (key + ".url")).write_text(full_url)
        return response.text, full_url


def base_row(station, source, variable, day, unit, value, url):
    mm, status = to_mm(value, unit)
    return dict(date=day, source=source, station_id=station.triplet,
                station_name=station.name, requested_ids=";".join(station.ids),
                variable=variable, value_mm=mm, status=status,
                source_unit=unit, source_url=url)


def fetch_nrcs(client, station, start, end, elements):
    """SEMIMONTHLY values use collectionDate, not an invented reporting date."""
    # Padding captures observations assigned to a neighbouring reporting month.
    begin = start - timedelta(days=45)
    finish = end + timedelta(days=45)
    text, url = client.get(AWDB, dict(
        stationTriplets=station.triplet, elements=",".join(elements),
        duration="SEMIMONTHLY", beginDate=begin.isoformat(), endDate=finish.isoformat(),
        periodRef="END", returnFlags="true", returnSuspectData="false"))
    payload = json.loads(text)
    rows = []
    for result in payload:
        if result["stationTriplet"] != station.triplet:
            raise ValueError("AWDB returned an unexpected station")
        for series in result.get("data", []):
            if series.get("error"):
                raise ValueError(series["error"])
            meta = series["stationElement"]
            unit = meta["storedUnitCode"]
            for item in series.get("values", []):
                if not item.get("collectionDate"):
                    if item.get("value") is not None:
                        raise ValueError("Manual observation has no collectionDate")
                    continue
                day = date.fromisoformat(item["collectionDate"][:10])
                if start <= day <= end:
                    row = base_row(station, "NRCS_manual", VARIABLES[meta["elementCode"]],
                                   day.isoformat(), unit, item.get("value"), url)
                    row.update(qc_flag=item.get("qcFlag"), qa_flag=item.get("qaFlag"),
                               reporting_year=item.get("year"), reporting_month=item.get("month"),
                               month_part=item.get("monthPart"))
                    rows.append(row)
    return rows


def fetch_snotel(station, start, end, elements):
    from metloom.pointdata import SnotelPointData
    point = SnotelPointData(station.triplet, station.name)
    variables = {"WTEQ": point.ALLOWED_VARIABLES.SWE,
                 "SNWD": point.ALLOWED_VARIABLES.SNOWDEPTH}
    result = point.get_daily_data(
        start_date=datetime.combine(start, datetime.min.time()),
        end_date=datetime.combine(end, datetime.min.time()),
        variables=[variables[e] for e in elements])
    if result is None or result.empty:
        return []
    rows = []
    # metloom returns UTC timestamps; recover the station's observation date.
    for _, item in result.reset_index().iterrows():
        stamp = pd.Timestamp(item["datetime"])
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert(point.tzinfo)
        day = stamp.date()
        if not start <= day <= end:
            continue
        for element in elements:
            name = variables[element].name
            if name not in item:
                continue
            unit_column = next((c for c in item.index if c.lower() == f"{name}_units".lower()), None)
            if unit_column is None:
                raise ValueError(f"metloom omitted units for {name}")
            row = base_row(station, "SNOTEL_metloom", VARIABLES[element], day.isoformat(),
                           item[unit_column], item[name],
                           f"https://wcc.sc.egov.usda.gov/nwcc/site?sitenum={station.triplet.split(':')[0]}")
            rows.append(row)
    return rows


def parse_nohrsc_history(html, url, start, end):
    """Parse official year tables; their headings specify SWE units explicitly."""
    soup = BeautifulSoup(html, "html.parser")
    rows, tables = [], 0
    for table in soup.find_all("table"):
        heading = table.get_text(" ", strip=True)
        match = re.search(r"SWE\s*\(\s*(cm|in|mm)\s*\)", heading, re.I)
        if not match or "Flight Line" not in heading:
            continue
        # Ignore outer layout tables containing nested observation tables.
        if table.find("table"):
            continue
        tables += 1
        for tr in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in tr.find_all("td", recursive=False)]
            if not cells or cells[0] not in FLIGHT_LINES:
                continue
            if len(cells) != 5:
                raise ValueError(f"Unexpected NOHRSC row: {cells}")
            day = date.fromisoformat(cells[1])
            if start <= day <= end:
                row = nohrsc_row(cells[0], day, cells[2], match[1].lower(), url)
                row.update(latitude=float(cells[3]), longitude=float(cells[4]))
                rows.append(row)
    if not tables:
        raise ValueError("No recognised NOHRSC historical observation table")
    return rows


def nohrsc_row(line, day, value, unit, url):
    station = Station(line, f"Airborne flight line {line}", (f"NOHRSC-{line}",))
    row = base_row(station, "NOHRSC_airborne", "swe", day.isoformat(), unit, float(value), url)
    if float(value) == -900:
        row.update(value_mm=None, status="above_measurement_range", lower_bound_mm=1000.0)
    return row


def parse_shef(text, url, start, end):
    if not re.search(r"SWE\s+SWE", text) or not re.search(r"\(in\)", text):
        raise ValueError(f"Unrecognised SHEF units/format: {url}")
    rows = []
    pattern = r"^\s*(AK\d+)\s+DY(\d{6}|\d{8})\s*/\s*[^/]+/\s*(-?\d+(?:\.\d+)?)\s*:"
    for line in text.splitlines():
        match = re.match(pattern, line)
        if match and match[1] in FLIGHT_LINES:
            day = datetime.strptime(match[2], "%y%m%d" if len(match[2]) == 6 else "%Y%m%d").date()
            if start <= day <= end:
                rows.append(nohrsc_row(match[1], day, match[3], "in", url))
        elif any(re.match(rf"\s*{code}\b", line) for code in FLIGHT_LINES):
            raise ValueError(f"Unparsed target flight line in {url}: {line}")
    return rows


def fetch_nohrsc(client, start, end):
    """Use yearly tables when published; discover APRFC surveys for newer seasons."""
    index, _ = client.get(HISTORY)
    soup = BeautifulSoup(index, "html.parser")
    years, seasons = {}, {}
    for a in soup.find_all("a", href=True):
        url = urljoin(HISTORY, a["href"])
        query = parse_qs(urlparse(url).query)
        if "year" in query and query["year"][0].isdigit():
            years[int(query["year"][0])] = url
        if "season" in query:
            seasons[query["season"][0]] = url
    rows = []
    # A year table can include December of the previous calendar year.
    for year in range(start.year, end.year + 2):
        if year in years:
            html, url = client.get(years[year])
            rows.extend(parse_nohrsc_history(html, url, start, end))
        else:
            season = f"{year - 1}-{year}"
            if season not in seasons:
                if year > end.year:
                    continue
                raise ValueError(f"No published NOHRSC archive for season {season}")
            html, season_url = client.get(seasons[season])
            links = BeautifulSoup(html, "html.parser").find_all("a", href=True)
            surveys = {urljoin(season_url, a["href"]) for a in links
                       if "APRFC" in a.get_text() and "surveys.html" in a["href"]}
            if not surveys:
                LOG.warning("No APRFC survey listed for %s", season)
            for survey_url in sorted(surveys):
                html, _ = client.get(survey_url)
                links = BeautifulSoup(html, "html.parser").find_all("a", href=True)
                products = {urljoin(survey_url, a["href"]) for a in links
                            if re.search(r"/gamma/.*\.atw$", a["href"], re.I)}
                if not products:
                    raise ValueError(f"No SHEF product links found at {survey_url}")
                for product in sorted(products):
                    text, url = client.get(product)
                    rows.extend(parse_shef(text, url, start, end))
    return rows


@dataclass
class SnowDownload:
    """Combined measurements, station coverage, and per-query status."""

    data: pd.DataFrame
    coverage: pd.DataFrame
    status: pd.DataFrame
    out: Path

    @property
    def ok(self) -> bool:
        return not self.status["status"].eq("error").any()


def _coerce_date(value: date | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _as_choices(value: str | list[str] | None, default: tuple[str, ...], label: str) -> list[str]:
    if value is None:
        return list(default)
    chosen = [value] if isinstance(value, str) else list(dict.fromkeys(value))
    unknown = sorted(set(chosen) - set(default))
    if unknown:
        raise ValueError(f"Unknown {label}: {', '.join(unknown)}")
    return chosen


def download_snow(
    start: date | str,
    end: date | str | None = None,
    out: Path | str = Path("snow_csv"),
    sources: str | list[str] | None = None,
    elements: str | list[str] | None = None,
    refresh: bool = False,
) -> SnowDownload:
    """Download Chena-area SWE and snow depth and write CSVs under ``out``.

    ``start`` and ``end`` are ISO dates (``YYYY-MM-DD``) or ``datetime.date``.
    ``end`` defaults to today. ``sources`` is any of ``snotel``, ``nrcs``, and
    ``nohrsc``. ``elements`` is any of ``WTEQ`` (snow water equivalent) and
    ``SNWD`` (snow depth). Set ``refresh`` to ignore cached NRCS and NOHRSC
    responses.

    Returns the combined table plus per-station coverage and per-query status.
    Those tables are also written under ``out``.
    """
    start_date = _coerce_date(start)
    end_date = date.today() if end is None else _coerce_date(end)
    if start_date > end_date or end_date > date.today():
        raise ValueError("Require start <= end <= today")
    sources = _as_choices(sources, SOURCES, "sources")
    elements = _as_choices(elements, tuple(VARIABLES), "elements")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    client = Client(out / "raw_cache", refresh)
    rows, report = [], []

    def run(label, action):
        LOG.info("Fetching %s", label)
        try:
            result = action()
            rows.extend(result)
            report.append(dict(query=label, status="ok" if result else "no_observations",
                               rows=len(result), error=""))
        except Exception as exc:
            LOG.error("%s: %s", label, exc)
            report.append(dict(query=label, status="error", rows=0, error=str(exc)))

    # Bound request sizes to calendar years for long historical downloads.
    for year in range(start_date.year, end_date.year + 1):
        year_start = max(start_date, date(year, 1, 1))
        year_end = min(end_date, date(year, 12, 31))
        if "snotel" in sources:
            for station in SNOTEL:
                run(f"{station.triplet}/{year}", lambda s=station, a=year_start, b=year_end:
                    fetch_snotel(s, a, b, elements))
        if "nrcs" in sources:
            for station in COURSES:
                run(f"{station.triplet}/{year}", lambda s=station, a=year_start, b=year_end:
                    fetch_nrcs(client, s, a, b, elements))
    if "nohrsc" in sources:
        if "WTEQ" in elements:
            run("NOHRSC airborne archive", lambda: fetch_nohrsc(client, start_date, end_date))
        else:
            LOG.warning("NOHRSC skipped: airborne surveys provide SWE, not snow depth")
    df = pd.DataFrame(rows).reindex(columns=COLUMNS)
    if not df.empty:
        # Overlapping archive pages may repeat observations. Prefer historical tables
        # (which include coordinates) over SHEF copies; retain distinct values.
        df = df.drop_duplicates(subset=["source", "station_id", "date", "variable", "value_mm"])
        df = df.sort_values(["source", "station_id", "date", "variable"])
    df.to_csv(out / "snow_all_mm.csv", index=False, float_format="%.6f")
    for source in ("SNOTEL_metloom", "NRCS_manual", "NOHRSC_airborne"):
        data = df[df["source"] == source]
        data.to_csv(out / f"{source}_mm.csv", index=False, float_format="%.6f")
    coverage = []
    expected = []
    if "snotel" in sources:
        expected.extend((s.triplet, s.ids) for s in SNOTEL)
    if "nrcs" in sources:
        expected.extend((s.triplet, s.ids) for s in COURSES)
    if "nohrsc" in sources and "WTEQ" in elements:
        expected.extend((line, (f"NOHRSC-{line}",)) for line in FLIGHT_LINES)
    for station_id, ids in expected:
        data = df[df["station_id"] == station_id]
        valid = data[data["value_mm"].notna()]
        coverage.append(dict(station_id=station_id, requested_ids=";".join(ids),
                             rows=len(data), available_values=len(valid),
                             first_date=valid["date"].min() if len(valid) else None,
                             last_date=valid["date"].max() if len(valid) else None))
    coverage_df = pd.DataFrame(coverage)
    status_df = pd.DataFrame(report, columns=["query", "status", "rows", "error"])
    coverage_df.to_csv(out / "station_coverage.csv", index=False)
    status_df.to_csv(out / "download_status.csv", index=False)
    LOG.info("Wrote %s measurement rows to %s", len(df), out)
    result = SnowDownload(data=df, coverage=coverage_df, status=status_df, out=out)
    if not result.ok:
        LOG.error("Download incomplete; see download_status.csv")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", type=date.fromisoformat, default=date.today())
    parser.add_argument("--out", type=Path, default=Path("snow_csv"))
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES),
                        default=["snotel", "nrcs", "nohrsc"])
    parser.add_argument("--elements", nargs="+", choices=list(VARIABLES), default=["WTEQ", "SNWD"])
    parser.add_argument("--refresh", action="store_true", help="Refresh cached NRCS/NOHRSC responses")
    args = parser.parse_args()
    if args.start > args.end or args.end > date.today():
        parser.error("Require start <= end <= today")
    result = download_snow(
        start=args.start,
        end=args.end,
        out=args.out,
        sources=args.sources,
        elements=args.elements,
        refresh=args.refresh,
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
