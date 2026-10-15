#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## turbo_worker.py
##
"""
    Runs one turbo solve in a process of its own, for ``CPM_turbo`` with
    ``isolate=True`` (the default). Not meant to be imported: ``CPM_turbo``
    starts it as ``python turbo_worker.py`` and talks to it through pipes.

    Why a process per solve: on its first solve, turbo reserves the CUDA
    device-malloc heap (90% of the GPU's memory with the default
    'barebones' architecture), and CUDA keeps it until the process exits.
    In the caller's process, that memory would stay taken for as long as
    the caller runs. A CUDA error in turbo also calls ``exit()``, and turbo
    prints solutions and warnings straight to stdout.

    Protocol: one JSON request on stdin, ``{"argv": [...]}``; one JSON
    reply on the original stdout, ``{"stats": {...} | null, "best": {...} |
    null, "error": {"type", "message"} | absent}``. Everything turbo prints
    goes to stderr, which the caller collects separately.

    Deliberately depends on nothing but the standard library and
    ``turbo_python``, so the child starts fast and never loads cpmpy.
"""

import json
import os
import signal
import sys

_PR_SET_PDEATHSIG = 1


def _die_with_parent(parent_pid: int) -> None:
    """Ask the kernel to SIGKILL this process when its parent dies, so a
    caller that is killed mid-solve doesn't leave a child holding the GPU.
    Linux only; elsewhere this is a no-op."""
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL)
    except (OSError, AttributeError):
        return
    if os.getppid() != parent_pid:  # the parent died before prctl took effect
        os._exit(1)


def main() -> None:
    # The signal mask survives exec: a caller that blocks SIGTERM (an MCP
    # server taking it with sigwait, say) would leave this child unkillable
    # by SIGTERM.
    signal.pthread_sigmask(signal.SIG_SETMASK, [])

    # Keep the real stdout for the reply, and point fd 1 at stderr before
    # turbo is loaded: everything turbo prints from C++ lands there.
    reply_channel = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)

    request = json.load(sys.stdin)
    _die_with_parent(int(request["parent_pid"]))

    reply = {"stats": None, "best": None}
    try:
        import turbo_python
        solver = turbo_python.Turbo(request["argv"])
        try:
            solver.solve()
        except Exception as e:
            reply["error"] = {"type": type(e).__name__, "message": str(e)}
        # stats/best stay readable after an interrupted solve
        for key, read in (("stats", solver.stats), ("best", solver.best)):
            try:
                reply[key] = dict(read())
            except Exception:
                pass
    except Exception as e:  # import or construction failed
        reply["error"] = {"type": type(e).__name__, "message": str(e)}

    sys.stdout.flush()
    reply_channel.write(json.dumps(reply))
    reply_channel.flush()


if __name__ == "__main__":
    main()
