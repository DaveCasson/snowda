"""Pytest fixtures for the laugh-test report.

Loaded only when pytest is pointed at this directory. The default
``testpaths`` setting does not include ``laugh_tests``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from laugh_tests._report import (
    DEVIATION,
    FAIL,
    NODE_REASON,
    NODE_STATUS,
    PASS,
    Laugh,
    render_markdown,
    render_text,
)

# Pytest imports this module before the test modules, so the backend is set
# before any test imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

OUTPUT = Path(__file__).resolve().parent / "output"


def pytest_configure(config: pytest.Config) -> None:
    logging.getLogger("snowda").setLevel(logging.ERROR)


@pytest.fixture
def laugh(request: pytest.FixtureRequest) -> Laugh:
    module = request.node.module.__name__.rsplit(".", 1)[-1]
    return Laugh(request.node.nodeid, module, request.node.name)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return
    wasxfail = getattr(report, "wasxfail", False)
    if wasxfail is not False and report.outcome != "failed":
        NODE_STATUS[item.nodeid] = DEVIATION
        NODE_REASON[item.nodeid] = str(wasxfail)
    elif report.passed:
        NODE_STATUS[item.nodeid] = PASS
        NODE_REASON[item.nodeid] = ""
    else:
        NODE_STATUS[item.nodeid] = FAIL
        NODE_REASON[item.nodeid] = ""


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if getattr(session.config.option, "collectonly", False):
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.md").write_text(render_markdown(), encoding="utf-8")
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    text = render_text()
    if reporter is not None:
        reporter.write_sep("=", "laugh test report")
        for line in text.splitlines():
            reporter.write_line(line)
    else:
        print(text)
