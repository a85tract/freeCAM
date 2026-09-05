"""The generated YAML tables the runtime reads: parsed once, with libyaml when present.

Every table under ``native/pi_cam`` is read-only data -- kernel descriptors,
physics-buffer fields, view codes.  Parsing them with the pure-Python loader
costs a stage about a second per rank at its first call and the same again
for every walk that reads the same file, so this module keeps one parsed
copy per file, keyed by its path, size and modification time.  Callers must
not mutate what they get back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_CACHE: dict[tuple[str, int, int], Any] = {}


def load_table(path: str | Path) -> Any:
    """The parsed YAML at ``path``, shared with every other reader of the same file."""

    source = Path(path).resolve()
    stat = source.stat()
    key = (str(source), stat.st_size, stat.st_mtime_ns)
    try:
        return _CACHE[key]
    except KeyError:
        pass
    payload = yaml.load(source.read_text(), Loader=_LOADER)
    _CACHE[key] = payload
    return payload
