#!/usr/bin/env python3
"""Compile every source-injected adapter with its real CAM build command."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
from hashlib import sha256
import json
import os
from pathlib import Path
import shlex
import subprocess
from typing import Any, Mapping, Sequence

from freecam.pi_cam.adapter_validation import load_adapter_build_contexts


def _commands(build_root: Path) -> tuple[tuple[str, ...], ...]:
    commands: list[tuple[str, ...]] = []
    for log in sorted(build_root.glob("*.bldlog*.gz")):
        with gzip.open(log, "rt", errors="replace") as stream:
            for line in stream:
                stripped = line.strip()
                if not stripped.startswith(("ftn ", "ifort ", "ifx ")):
                    continue
                try:
                    command = tuple(shlex.split(stripped))
                except ValueError:
                    continue
                if "-c" in command:
                    commands.append(command)
    return tuple(commands)


def _source_command(
    commands: Sequence[tuple[str, ...]], source: str
) -> tuple[str, ...] | None:
    suffix = "/" + source.lstrip("/")
    exact = tuple(
        command
        for command in commands
        if any(token.endswith(suffix) for token in command)
    )
    if exact:
        return exact[-1]
    name = Path(source).name
    matches = tuple(
        command
        for command in commands
        if any(Path(token).name == name for token in command)
    )
    return matches[-1] if len(matches) == 1 else None


def _compile_one(
    item: tuple[
        str,
        tuple[Mapping[str, Any], ...],
        Path,
        Path,
        Path,
        tuple[Any, ...],
        Mapping[str, tuple[tuple[str, ...], ...]],
    ]
) -> dict[str, Any]:
    source, adapters, source_root, work_root, library_root, contexts, commands = item
    patched = source_root / source
    attempts: list[dict[str, Any]] = []
    for context in contexts:
        if not context.accepts_source(source):
            continue
        original = _source_command(commands[context.name], source)
        if original is None:
            attempts.append(
                {"context": context.name, "status": "missing_build_command"}
            )
            continue
        output = work_root / context.name / sha256(source.encode()).hexdigest()[:16]
        output.mkdir(parents=True, exist_ok=True)
        object_path = output / (patched.stem + ".o")
        command = list(original)
        replaced = False
        for index, token in enumerate(command):
            if token.endswith("/" + source) or Path(token).name == patched.name:
                command[index] = str(patched)
                replaced = True
                break
        if not replaced:
            attempts.append({"context": context.name, "status": "source_not_replaced"})
            continue
        command.extend(
            (
                "-fPIC",
                "-I" + str(context.build_root / "atm/obj"),
                "-module",
                str(output),
                "-o",
                str(object_path),
            )
        )
        completed = subprocess.run(
            command,
            cwd=output,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            attempts.append(
                {
                    "context": context.name,
                    "status": "compile_failed",
                    "diagnostic": (completed.stderr or completed.stdout)[-8000:],
                }
            )
            continue
        symbols = subprocess.run(
            ["nm", "--defined-only", str(object_path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        missing = [
            str(adapter["symbol"])
            for adapter in adapters
            if str(adapter["symbol"]) not in symbols
        ]
        status = "passed" if not missing else "missing_symbols"
        library_path: Path | None = None
        link_diagnostic = ""
        if status == "passed":
            library_root.mkdir(parents=True, exist_ok=True)
            library_path = library_root / (
                sha256(source.encode()).hexdigest()[:16] + "-" + patched.stem + ".so"
            )
            link = subprocess.run(
                [
                    original[0],
                    "-shared",
                    "-Wl,-z,notext,--allow-shlib-undefined,-Bsymbolic-functions",
                    str(object_path),
                    "-o",
                    str(library_path),
                ],
                cwd=output,
                check=False,
                capture_output=True,
                text=True,
            )
            if link.returncode:
                status = "link_failed"
                link_diagnostic = (link.stderr or link.stdout)[-8000:]
                library_path = None
        attempts.append(
            {
                "context": context.name,
                "status": status,
                "object_path": str(object_path),
                "object_sha256": sha256(object_path.read_bytes()).hexdigest(),
                "library": None if library_path is None else str(library_path),
                "library_sha256": (
                    None
                    if library_path is None
                    else sha256(library_path.read_bytes()).hexdigest()
                ),
                "missing_symbols": missing,
                "diagnostic": link_diagnostic,
            }
        )
        if status == "passed":
            return {
                "source": source,
                "adapters": len(adapters),
                "status": status,
                "selected_context": context.name,
                "object_path": str(object_path),
                "object_sha256": sha256(object_path.read_bytes()).hexdigest(),
                "library": str(library_path),
                "library_sha256": sha256(library_path.read_bytes()).hexdigest(),
                "attempts": attempts,
            }
    return {
        "source": source,
        "adapters": len(adapters),
        "status": "failed",
        "selected_context": None,
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-report", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--contexts",
        type=Path,
        default=Path("validation/pi_cam_adapter_build_contexts.yaml"),
    )
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()

    generation = json.loads(arguments.generation_report.read_text())
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for adapter in generation["generated"]:
        grouped.setdefault(str(adapter["original_source"]), []).append(adapter)
    contexts = load_adapter_build_contexts(arguments.contexts)
    command_index = {
        context.name: _commands(context.build_root)
        for context in contexts
        if context.build_root is not None
    }
    tasks = tuple(
        (
            source,
            tuple(adapters),
            arguments.source_root.resolve(),
            arguments.work_root.resolve(),
            arguments.library_root.resolve(),
            contexts,
            command_index,
        )
        for source, adapters in sorted(grouped.items())
    )
    with ThreadPoolExecutor(max_workers=max(1, arguments.workers)) as executor:
        sources = tuple(executor.map(_compile_one, tasks))
    report = {
        "schema_version": 1,
        "validator": "tools/validate_pi_cam_in_module_adapters.py",
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "generated_adapters": sum(int(item["adapters"]) for item in sources),
        "source_files": len(sources),
        "compiled_sources": sum(item["status"] == "passed" for item in sources),
        "compiled_adapters": sum(
            int(item["adapters"]) for item in sources if item["status"] == "passed"
        ),
        "passed": all(item["status"] == "passed" for item in sources),
        "sources": sources,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        f"compiled {report['compiled_adapters']}/{report['generated_adapters']} "
        f"adapters from {report['compiled_sources']}/{report['source_files']} sources"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
