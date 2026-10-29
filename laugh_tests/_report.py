"""Collect laugh-test comparisons and render a readable report."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASS = "PASS"
DEVIATION = "KNOWN DEVIATION"
FAIL = "FAIL"


@dataclass
class LaughRecord:
    module: str
    test: str
    label: str
    expected: str
    computed: str
    ok: bool
    reference: str
    nodeid: str


RECORDS: list[LaughRecord] = []
NODE_STATUS: dict[str, str] = {}
NODE_REASON: dict[str, str] = {}
FIGURES: list[str] = []


def format_value(value: object) -> str:
    """Render a scalar or a small array on one line."""
    if value is None:
        return "None"
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if number == 0.0:
            return "0"
        return f"{number:.6g}"
    arr = np.asarray(value)
    if arr.ndim == 0:
        return format_value(arr.item())
    if np.issubdtype(arr.dtype, np.floating):
        text = np.array2string(
            arr,
            precision=6,
            separator=", ",
            max_line_width=10_000,
            threshold=16,
        )
    else:
        text = np.array2string(arr, separator=", ", max_line_width=10_000, threshold=16)
    return " ".join(text.split())


def values_match(
    expected: object,
    computed: object,
    *,
    atol: float,
    rtol: float,
    exact: bool,
) -> bool:
    if isinstance(expected, (bool, np.bool_)) or isinstance(computed, (bool, np.bool_)):
        return bool(expected) == bool(computed)
    if expected is None or computed is None:
        return expected is computed
    if isinstance(expected, str) or isinstance(computed, str):
        return expected == computed
    exp = np.asarray(expected)
    comp = np.asarray(computed)
    if exp.shape != comp.shape:
        return False
    if exact:
        return bool(np.array_equal(exp, comp))
    return bool(
        np.allclose(
            exp.astype(float),
            comp.astype(float),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )
    )


class Laugh:
    """Record one analytical comparison, then assert it."""

    def __init__(self, nodeid: str, module: str, test: str) -> None:
        self.nodeid = nodeid
        self.module = module
        self.test = test

    def check(
        self,
        label: str,
        expected: object,
        computed: object,
        *,
        atol: float = 0.0,
        rtol: float = 1e-12,
        reference: str = "",
        exact: bool = False,
    ) -> None:
        ok = values_match(expected, computed, atol=atol, rtol=rtol, exact=exact)
        expected_text = format_value(expected)
        computed_text = format_value(computed)
        RECORDS.append(
            LaughRecord(
                module=self.module,
                test=self.test,
                label=label,
                expected=expected_text,
                computed=computed_text,
                ok=ok,
                reference=reference,
                nodeid=self.nodeid,
            )
        )
        if ok:
            return
        tolerance = "exact" if exact else f"atol={atol:g}, rtol={rtol:g}"
        raise AssertionError(
            f"{label}\n"
            f"  expected:  {expected_text}\n"
            f"  computed:  {computed_text}\n"
            f"  tolerance: {tolerance}\n"
            f"  reference: {reference}"
        )


def row_status(record: LaughRecord) -> tuple[str, str]:
    node = NODE_STATUS.get(record.nodeid, FAIL)
    reason = NODE_REASON.get(record.nodeid, "")
    if not record.ok and node == DEVIATION:
        return DEVIATION, reason
    if record.ok:
        return PASS, ""
    return FAIL, reason


def _rows() -> list[tuple[LaughRecord, str, str]]:
    return [(record, *row_status(record)) for record in RECORDS]


def _counts(rows: list[tuple[LaughRecord, str, str]]) -> dict[str, int]:
    counts = {PASS: 0, DEVIATION: 0, FAIL: 0}
    for _record, status, _reason in rows:
        counts[status] = counts.get(status, 0) + 1
    return counts


def render_text() -> str:
    rows = _rows()
    counts = _counts(rows)
    lines = [
        (
            f"{counts[PASS]} PASS, {counts[DEVIATION]} KNOWN DEVIATION, "
            f"{counts[FAIL]} FAIL"
        ),
        "",
    ]
    for record, status, reason in rows:
        lines.append(f"{status:<16} {record.module}::{record.test}")
        lines.append(f"  check      {record.label}")
        lines.append(f"  expected   {record.expected}")
        lines.append(f"  computed   {record.computed}")
        if record.reference:
            lines.append(f"  reference  {record.reference}")
        if reason and status == DEVIATION:
            lines.append(f"  note       {' '.join(reason.split())}")
        lines.append("")
    if FIGURES:
        lines.append("Figures")
        for name in FIGURES:
            lines.append(f"  laugh_tests/output/{name}")
        lines.append("")
    lines.append("Full report: laugh_tests/output/report.md")
    return "\n".join(lines)


def _markdown_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown() -> str:
    rows = _rows()
    counts = _counts(rows)
    lines = [
        "# Laugh test report",
        "",
        (
            f"{counts[PASS]} PASS, {counts[DEVIATION]} KNOWN DEVIATION, "
            f"{counts[FAIL]} FAIL."
        ),
        "",
        "A PASS row matches an analytical result. A KNOWN DEVIATION row is a",
        "textbook result the current code does not reproduce. The test is marked",
        "strict, so the suite fails if that code starts matching the textbook and",
        "the deviation note is not retired.",
        "",
        "| Status | Test | Check | Expected | Computed | Reference |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for record, status, _reason in rows:
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(part)
                for part in (
                    status,
                    f"{record.module}::{record.test}",
                    record.label,
                    record.expected,
                    record.computed,
                    record.reference,
                )
            )
            + " |"
        )
    deviations = [
        (record, reason) for record, status, reason in rows if status == DEVIATION
    ]
    if deviations:
        lines.extend(["", "## Known deviations", ""])
        for record, reason in deviations:
            lines.append(f"- **{record.module}::{record.test}** — {record.label}")
            lines.append(f"  - expected: {record.expected}")
            lines.append(f"  - computed: {record.computed}")
            if reason:
                lines.append(f"  - note: {' '.join(reason.split())}")
    if FIGURES:
        lines.extend(["", "## Figures", ""])
        for name in FIGURES:
            lines.append(f"- [{name}]({name})")
    lines.append("")
    return "\n".join(lines)
