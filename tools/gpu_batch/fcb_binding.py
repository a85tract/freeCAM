"""Hand the batched-GPU plugin (tools/gpu_batch/fcb_plugin.c, built by build.sh) to freeCAM's CLI:

    --kernel-plugin compute_uwshcu_inv=/path/fcb_binding.py:plugin    (or --shadow-kernel-plugin)

Environment: FCB_PLUGIN_SO (the built fcb_plugin.so), FCB_MODEL (the TorchScript file),
FCB_GPUS_PER_NODE (groups: ranks sharing a GPU, default 4), FCB_DEVICE_INDEX (default 0: the
one device the rank wrapper leaves visible).  fcb_init runs here, once per rank, before the
model initialises: the node-shared window is allocated and the group leader loads the model
on its GPU.
"""
import ctypes
import os
from pathlib import Path

from mpi4py import MPI

from freecam.physics.native_model import NativePlugin

_so = Path(os.environ["FCB_PLUGIN_SO"]).resolve()
_model = Path(os.environ["FCB_MODEL"]).resolve()
_lib = ctypes.CDLL(str(_so), mode=ctypes.RTLD_GLOBAL)
_lib.fcb_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
_lib.fcb_init.restype = ctypes.c_int
_status = _lib.fcb_init(MPI.COMM_WORLD.py2f(), int(os.environ.get("FCB_GPUS_PER_NODE", "4")), str(_model).encode(),
                        int(os.environ.get("FCB_DEVICE_INDEX", "0")))
if _status != 0:
    raise RuntimeError(f"fcb_init failed with status {_status} on rank {MPI.COMM_WORLD.Get_rank()}")

plugin = NativePlugin.from_library(_lib, "fcb_plugin", kernel="compute_uwshcu_inv")

# at exit, every rank reports where its time inside the plugin went (means in ms over its calls):
# gather, wait for the group, forward (leader only), wait for the result, scatter, total, calls
import atexit


def _report_stats() -> None:
    stats = (ctypes.c_double * 7)()
    _lib.fcb_stats.argtypes = [ctypes.POINTER(ctypes.c_double)]
    _lib.fcb_stats(stats)
    _lib.fcb_group_rank.restype = ctypes.c_int
    print(f"fcb plugin stats rank {MPI.COMM_WORLD.Get_rank()} grank {_lib.fcb_group_rank()}: gather {stats[0]:.3f} wait_in {stats[1]:.3f} "
          f"forward {stats[2]:.3f} wait_out {stats[3]:.3f} scatter {stats[4]:.3f} total {stats[5]:.3f} ms over {int(stats[6])} calls", flush=True)


atexit.register(_report_stats)
