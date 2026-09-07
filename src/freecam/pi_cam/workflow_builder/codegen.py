"""The generator, called from Python.

There is one code generator, written in TypeScript for the page.  Bundled
for Node by the front-end build, it is called here so the service and the
tests produce byte for byte what the browser produces.  Without the bundle
(or Node), the service still accepts the browser's own text.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .document import GeneratedWorkflowArtifact, WorkflowDocument

STATIC = Path(__file__).resolve().parent / "static"
BUNDLE = STATIC / "codegen.mjs"

ARTIFACT_FILES: Mapping[str, str] = {
    "setup": "setup.py",
    "script": "run_workflow.py",
    "notebook": "workflow.ipynb",
    "workflow": "workflow.json",
}


@dataclass(frozen=True, slots=True)
class Artifacts:
    setup: str
    script: str
    notebook: str
    workflow: str
    external_files: tuple[str, ...] = ()
    changes: tuple[str, ...] = ()

    def texts(self) -> dict[str, str]:
        return {"setup": self.setup, "script": self.script, "notebook": self.notebook, "workflow": self.workflow}


def available() -> bool:
    return BUNDLE.is_file() and shutil.which("node") is not None


def generate(document: WorkflowDocument, snapshot: Mapping[str, Any]) -> Artifacts:
    """Run the bundled generator on ``document``; raises when Node or the bundle is missing."""

    node = shutil.which("node")
    if node is None or not BUNDLE.is_file():
        raise RuntimeError(
            "the code generator bundle is not built (run `npm run build` under web/) or node is missing"
        )
    import tempfile

    with tempfile.TemporaryDirectory(prefix="freecam-codegen-") as directory:
        document_path = Path(directory) / "document.json"
        snapshot_path = Path(directory) / "snapshot.json"
        document_path.write_text(json.dumps(document.to_payload()))
        snapshot_path.write_text(json.dumps(snapshot, default=str))
        result = subprocess.run(
            [node, str(BUNDLE), str(document_path), str(snapshot_path)],
            capture_output=True, text=True, check=False, timeout=120,
        )
    if result.returncode != 0:
        raise RuntimeError(f"the code generator failed: {result.stderr.strip()}")
    payload = json.loads(result.stdout)
    return Artifacts(
        setup=str(payload["setup"]), script=str(payload["script"]), notebook=str(payload["notebook"]),
        workflow=str(payload["workflow"]), external_files=tuple(payload.get("external_files", ())),
        changes=tuple(payload.get("changes", ())),
    )


def browser_hash(document: WorkflowDocument) -> str:
    """The hash the page computes for ``document``, through the same bundle."""

    node = shutil.which("node")
    if node is None or not BUNDLE.is_file():
        raise RuntimeError("the code generator bundle is not built or node is missing")
    import tempfile

    with tempfile.TemporaryDirectory(prefix="freecam-codegen-") as directory:
        document_path = Path(directory) / "document.json"
        document_path.write_text(json.dumps(document.to_payload()))
        result = subprocess.run([node, str(BUNDLE), "--hash", str(document_path)],
                                capture_output=True, text=True, check=True, timeout=60)
    return result.stdout.strip()


def write_artifacts(directory: Path, workflow_hash: str, texts: Mapping[str, str]) -> GeneratedWorkflowArtifact:
    """Write the generated texts under ``directory/<hash>/`` and describe them."""

    target = Path(directory) / workflow_hash[:12]
    target.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for key, text in texts.items():
        name = ARTIFACT_FILES.get(key)
        if name is None:
            continue
        path = target / name
        path.write_text(text)
        files[key] = path
    manifest = {"workflow_hash": workflow_hash, "files": {k: str(v) for k, v in files.items()}}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return GeneratedWorkflowArtifact(
        name=workflow_hash[:12], directory=target, files=files, workflow_hash=workflow_hash, manifest=manifest,
    )


__all__ = ["ARTIFACT_FILES", "Artifacts", "BUNDLE", "STATIC", "available", "browser_hash", "generate", "write_artifacts"]
