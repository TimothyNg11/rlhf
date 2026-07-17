"""Line-buffered JSONL metrics logging for the training loop.

Each row is written and flushed immediately so an external process (e.g.
``ssh`` + ``tail -f`` on a running training pod) can read metrics as they
land, without waiting for the file to be closed.
"""

from __future__ import annotations

import json
from pathlib import Path


class MetricsLogger:
    """Line-buffered JSONL writer. Use as a context manager or call
    :meth:`close` explicitly when done."""

    def __init__(self, path: str | Path, *, resume: bool = False):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        self._file = open(path, mode, encoding="utf-8")

    def log(self, row: dict) -> None:
        """Write ``row`` as one JSON line and flush so it is immediately
        visible to another process reading the file."""
        self._file.write(json.dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "MetricsLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def read_metrics(path: str | Path) -> list[dict]:
    """Parse a metrics JSONL file into a list of dicts, skipping blank lines."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
