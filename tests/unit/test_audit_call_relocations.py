"""The relocation audit reads objects the way the hook build does and classifies what it finds."""
import importlib.util
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit_relocations", REPO / "tools/audit_pi_cam_call_relocations.py")
audit = importlib.util.module_from_spec(spec)
sys.modules["audit_relocations"] = audit
spec.loader.exec_module(audit)

SYMBOLS = """
Symbol table '.symtab' contains 5 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
     1: 000000000004c4b0   720 FUNC    GLOBAL DEFAULT    5 uwshcu_mp_fluxbelowinv_
     2: 0000000000000000     0 NOTYPE  GLOBAL DEFAULT  UND cloud_fraction_mp_cldfrc_fice_
     3: 000000000001c1e0 0x1c1e0 FUNC    LOCAL  DEFAULT    5 uwshcu_mp_compute_uwshcu_
     4: 0000000000000010    12 OBJECT  GLOBAL DEFAULT    9 uwshcu_mp_mkx_
"""

RELOCATIONS = """
Relocation section '.rela.text' at offset 0x100 contains 3 entries:
    Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
000000000000a1f0  000000d500000004 R_X86_64_PLT32         000000000004c4b0 uwshcu_mp_fluxbelowinv_ - 4
000000000000b2f0  000000d500000004 R_X86_64_PLT32         000000000004c4b0 uwshcu_mp_fluxbelowinv_ - 4
000000000000c3f0  000000e100000002 R_X86_64_PC32          0000000000000000 .rodata + 10

Relocation section '.rela.trace' at offset 0x200 contains 1 entry:
    Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
0000000000000010  000000d500000001 R_X86_64_64            000000000004c4b0 uwshcu_mp_compute_alpha_ + 0
"""


def test_readelf_output_is_parsed_into_definitions_and_text_references(monkeypatch) -> None:
    outputs = {"-sW": SYMBOLS, "-rW": RELOCATIONS}
    monkeypatch.setattr(audit, "_run", lambda command, cwd=None: outputs[command[1]])
    functions = audit.defined_functions(Path("uwshcu.o"))
    assert functions == {"uwshcu_mp_fluxbelowinv_": {"binding": "GLOBAL", "size": 720},
                         "uwshcu_mp_compute_uwshcu_": {"binding": "LOCAL", "size": 0x1C1E0}}
    references = audit.text_references(Path("uwshcu.o"))
    assert references == {"uwshcu_mp_fluxbelowinv_": Counter({"R_X86_64_PLT32": 2}), ".rodata": Counter({"R_X86_64_PC32": 1})}
    # the .rela.trace reference to compute_alpha is not a call


def test_procedure_symbols_follow_ifort_naming() -> None:
    definitions = {"uwshcu_mp_fluxbelowinv_": {}, "dadadj_": {}, "mcica_subcol_gen_lw_mp_kissvec_ip_low_byte_": {}}
    assert audit.procedure_symbols({"name": "fluxbelowinv", "module": "uwshcu", "host": None}, definitions) == ["uwshcu_mp_fluxbelowinv_"]
    assert audit.procedure_symbols({"name": "dadadj", "module": None, "host": None}, definitions) == ["dadadj_"]
    hosted = {"name": "low_byte", "module": "mcica_subcol_gen_lw", "host": "mcica_subcol_gen_lw::kissvec"}
    assert audit.procedure_symbols(hosted, definitions) == ["mcica_subcol_gen_lw_mp_kissvec_ip_low_byte_"]
    assert audit.procedure_symbols({**hosted, "name": "m"}, definitions) == []


def _record(defined_in, references):
    return [{"symbol": "s", "defined_in": {obj: {} for obj in defined_in},
             "references": [{"object": obj, "relocations": 1, "types": ["R_X86_64_PLT32"]} for obj in references]}]


def test_classification_matches_the_two_redirection_modes() -> None:
    assert audit.classify(_record(["cloud_fraction.o"], ["zm_conv.o", "macrop_driver.o"])) == "rename-references"
    assert audit.classify(_record(["uwshcu.o"], ["uwshcu.o"])) == "weaken-definition"
    assert audit.classify(_record(["uwshcu.o"], ["uwshcu.o", "convect_shallow.o"])) == "weaken-definition"
    assert audit.classify(_record(["uwshcu.o"], [])) == "no-call-relocation"
    assert audit.classify([]) == "not-in-archive"
    assert audit.classify(_record([], [])) == "not-in-archive"
