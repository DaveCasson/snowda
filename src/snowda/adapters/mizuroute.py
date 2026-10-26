from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Mapping

from snowda.utils.logging import get_logger

log = get_logger(__name__)


def _pad_field(value: str, width: int = 20) -> str:
    return f"{value:{width}}"


def read_control_value(control_file: str | Path, key: str) -> str | None:
    """Return the first whitespace-delimited value for `<key>` in a control file."""
    token = f"<{key}>"
    try:
        lines = Path(control_file).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith(token):
            continue
        tail = stripped[len(token) :].strip()
        if not tail:
            return None
        return tail.split()[0].strip().strip("'\"")
    return None


def update_control_file(
    template_path: str | Path,
    output_path: str | Path,
    settings: Mapping[str, str],
) -> Path:
    """
    Update mizuRoute control-file tokens in `<key>` form.

    Parameters
    ----------
    template_path
        Path to the template control file.
    output_path
        Path where the updated control file is written.
    settings
        Mapping from token key to replacement string value.
    """
    template = Path(template_path)
    out = Path(output_path)
    lines = template.read_text(encoding="utf-8").splitlines()

    updated_lines: list[str] = []
    for line in lines:
        rewritten = line
        stripped = line.strip()
        for key, value in settings.items():
            token = f"<{key}>"
            if not stripped.startswith(token):
                continue
            prefix_len = len(line) - len(line.lstrip(" "))
            comment = ""
            has_bang = "!" in line
            if has_bang:
                comment = line.split("!", 1)[1].strip()
            key_field = token.ljust(30)
            value_field = _pad_field(str(value))
            rewritten = f"{' ' * prefix_len}{key_field}{value_field}"
            if has_bang:
                if comment:
                    rewritten += f"! {comment}"
                else:
                    rewritten += "!"
            break
        updated_lines.append(rewritten)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    return out


def run_mizuroute(
    executable: str | Path, control_file: str | Path, *, log_stdout: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run mizuRoute and return subprocess metadata."""
    command = [str(executable), str(control_file)]
    log.info("Running mizuRoute: %s", " ".join(command))
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
    )
    # #region agent log
    try:
        import json
        import time

        payload = {
            "sessionId": "83d907",
            "hypothesisId": "A",
            "location": "mizuroute.py:run_mizuroute",
            "message": "mizuroute returned",
            "data": {
                "control": str(control_file),
                "returncode": completed.returncode,
                "stderr_tail": (
                    completed.stderr.decode("utf-8", errors="replace")[-300:]
                    if completed.stderr
                    else ""
                ),
            },
            "timestamp": int(time.time() * 1000),
        }
        with open("/Users/dcasson/GitHub/snowda/.cursor/debug-83d907.log", "a") as handle:
            handle.write(json.dumps(payload) + "\n")
    except Exception:
        pass
    # #endregion
    stderr = completed.stderr.decode("utf-8", errors="replace") if completed.stderr else ""
    stdout = completed.stdout.decode("utf-8", errors="replace") if completed.stdout else ""
    if stderr and "FATAL ERROR" in stderr:
        raise RuntimeError(f"mizuRoute reported fatal error: {stderr}")
    if stdout and log_stdout:
        log.info("mizuRoute stdout: %s", stdout)
    return completed
