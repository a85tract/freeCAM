"""The source a new Python process starts from."""

from __future__ import annotations


def python_process_template(name: str, after: str | None = None) -> str:
    """An ``fc.Physics`` class the user edits in the Inspector.

    The class is what the generated script and notebook carry verbatim, so
    it is complete and runnable as written: a name, its place in the step,
    the fields it reads and writes, one tunable, and a ``run`` that does
    something visible.
    """

    class_name = "".join(part.capitalize() for part in name.split("_")) or "Process"
    anchor = after or "dry_adjustment"
    return (
        f"class {class_name}(fc.Physics):\n"
        f'    """A rank-local Python process; edit freely."""\n'
        f"\n"
        f'    name = "{name}"\n'
        f'    after = "{anchor}"\n'
        f'    reads = ("T",)\n'
        f'    writes = ("T",)\n'
        f"    rate = fc.Property(0.0, doc=\"heating rate in K per step\")\n"
        f"\n"
        f"    def run(self, state, context):\n"
        f"        state.T += self.rate\n"
    )


__all__ = ["python_process_template"]
