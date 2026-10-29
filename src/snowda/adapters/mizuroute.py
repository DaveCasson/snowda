from __future__ import annotations

import os
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


_OUTPUT_TAIL = 4000


def _decode_output(data: bytes | str | None) -> str:
    if not data:
        return ""
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="replace")
    return data


def _tail(text: str, limit: int = _OUTPUT_TAIL) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"... ({omitted} characters omitted)\n{text[-limit:]}"


def _failure_detail(returncode: int, command: list[str], stdout: str, stderr: str) -> str:
    stdout_text = _tail(stdout) if stdout.strip() else "(empty)"
    stderr_text = _tail(stderr) if stderr.strip() else "(empty)"
    return (
        f"mizuRoute failed (exit {returncode}): {' '.join(command)}\n"
        f"stdout:\n{stdout_text}\n"
        f"stderr:\n{stderr_text}"
    )


def _mizuroute_env() -> dict[str, str]:
    """Drop the SLURM step id so Open MPI starts one local process.

    Inside a batch job, ``SLURM_NODELIST`` plus ``SLURM_STEP_ID`` makes this
    Open MPI build treat a direct ``exec`` as ``srun``. That build has no
    SLURM PMI support, so ``MPI_Init`` aborts. With the step id unset, the
    same binary starts as a single process, which is how it runs on a login
    node. ``OMPI_MCA_ess=singleton`` does not help: Open MPI overwrites it
    when ``SLURM_STEP_ID`` is set.
    """
    env = os.environ.copy()
    env.pop("SLURM_STEP_ID", None)
    env.pop("SLURM_STEPID", None)
    return env


def run_mizuroute(
    executable: str | Path, control_file: str | Path, *, log_stdout: bool = True
) -> subprocess.CompletedProcess[bytes]:
    """Run mizuRoute and return subprocess metadata.

    A non-zero exit, or ``FATAL ERROR`` on either stream, is logged and raised
    with the tail of both streams. mizuRoute writes its message to stdout.
    """
    command = [str(executable), str(control_file)]
    log.info("Running mizuRoute: %s", " ".join(command))
    completed = subprocess.run(command, capture_output=True, env=_mizuroute_env())
    stdout = _decode_output(completed.stdout)
    stderr = _decode_output(completed.stderr)
    failed = completed.returncode != 0 or "FATAL ERROR" in stdout or "FATAL ERROR" in stderr
    if failed:
        detail = _failure_detail(completed.returncode, command, stdout, stderr)
        log.error("%s", detail)
        raise RuntimeError(detail)
    if stdout and log_stdout:
        log.info("mizuRoute stdout: %s", stdout)
    return completed
