# This package shares a name with the Python stdlib 'queue' module.
# To prevent shadowing, we load the stdlib queue directly and re-export
# all its public symbols, so that `from queue import Queue` works correctly
# whether Python resolves 'queue' to this package or to stdlib.
import importlib.util as _util
import sysconfig as _sysconfig
import os as _os
import sys as _sys
import types as _types

_stdlib_queue_path = _os.path.join(_sysconfig.get_path("stdlib"), "queue.py")
_spec = _util.spec_from_file_location("_stdlib_queue", _stdlib_queue_path)
_mod = _util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export all public names from stdlib queue
Queue = _mod.Queue
LifoQueue = _mod.LifoQueue
PriorityQueue = _mod.PriorityQueue
SimpleQueue = _mod.SimpleQueue
Empty = _mod.Empty
Full = _mod.Full

__all__ = ["Queue", "LifoQueue", "PriorityQueue", "SimpleQueue", "Empty", "Full"]
