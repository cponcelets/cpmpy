# <b>CPMpy-turbo</b>: Integrating the <a href="https://github.com/ptal/turbo">turbo</a> solver into <a href="https://github.com/CPMpy/cpmpy">CPMpy</a>

![turbo tests passing](https://img.shields.io/badge/turbo%20tests-14%2C330%20passing-brightgreen)
![turbo tests failing](https://img.shields.io/badge/turbo%20tests-149%20failing-red)
![turbo tests skipped](https://img.shields.io/badge/turbo%20tests-24%20skipped-lightgrey)

<sub>`pytest --forked tests/ --solver=turbo`, 2026-09-16 — update after the next full rerun, these are manually maintained, not CI-linked.</sub>
<sub>- Fail reasons: Incremental sovling and solveAll not supported yet.<sub>

## Content

This is a fork of [CPMpy](https://github.com/CPMpy/cpmpy) that adds `CPM_turbo`, a solver interface to [turbo](https://github.com/ptal/turbo): an open-source research project aiming to be a constraint solver that runs entirely on GPUs.

For CPMpy general usage see the upstream [CPMpy README](https://github.com/CPMpy/cpmpy) and [documentation](https://cpmpy.readthedocs.io/).

## Turbo solver interface

`CPM_turbo` lets you solve CPMpy models with turbo like any other CPMpy solver.

### How it works

Turbo accepts [XCSP3](https://xcsp.org/specifications/) and [FlatZinc](https://docs.minizinc.dev/en/stable/flattening.html) has input format, (for the moment) we use MiniZinc module as a translator:

1. the model is built through CPMpy's MiniZinc interface (`CPM_minizinc`, using the `turbo.gpu.release` solver configuration, which selects the right globals library / FlatZinc dialect)
2. just before solving, it is compiled to FlatZinc
3. the FlatZinc text is handed to `turbo_python`, which performs the actual search on the GPU — MiniZinc itself never runs a search

### Installation

Requires the `turbo_python` package and MiniZinc:
-  `turbo_python` is a CUDA/CMake build, and
- MiniZinc needs its binary bundle plus a `turbo.gpu.release.msc` solver configuration registered so it can find turbo. [thirdparty/install_turbo.sh](thirdparty/install_turbo.sh) automates all of it (see [thirdparty/README.md](thirdparty/README.md)):

First, set up a virtual environment with [uv](https://docs.astral.sh/uv/) (install it via `curl -LsSf https://astral.sh/uv/install.sh | sh` if you don't have it yet):

```
uv venv
source .venv/bin/activate
uv pip install "cpmpy[turbo] @ git+https://github.com/cponcelets/cpmpy@turbo"
```

Then build/install `turbo_python` and MiniZinc into that venv:

```
./thirdparty/install_turbo.sh
source thirdparty/env.sh  # puts MiniZinc's bundle on PATH; do this in every new shell
```

### Usage

```python
import cpmpy as cp

x = cp.intvar(0, 10, shape=3)
model = cp.Model([cp.AllDifferent(x), cp.sum(x) == 10])
model.solve(solver="turbo")
print(x.value())
```

## Configuration

Any keyword argument passed to `solve()`/`solveAll()` is translated 1-to-1 into a turbo command-line flag, so turbo's own options (documented at https://github.com/ptal/turbo) can be passed straight from CPMpy: a single-character key becomes `-x`, a longer one becomes `--xxx`; booleans are passed as bare flags (only when `True`), anything else as `flag value`.

```python
model.solve(solver="turbo", time_limit=10, arch="gpu", p=4, verbose=True)
# -> turbo -a -t 10000 --arch gpu -p 4 --verbose <model.fzn>
```

A few flags are handled for you:
- `time_limit` (seconds) is converted to turbo's `-t`/`--timeout` in milliseconds; only pass `-t`/`--timeout` yourself if you need a different unit/behaviour.
- `verbose=True` (or `v=True`) also makes CPMpy print the MiniZinc and FlatZinc text at every `compile()` call.

`solveAll(solution_limit=...)` is accepted for API compatibility but ignored with a warning: turbo does not support a solution limit, and its Python API only exposes the best/last solution, so `display` is invoked at most once rather than once per solution.

See [cpmpy/solvers/turbo.py](cpmpy/solvers/turbo.py) for the full interface documentation.

## License

Same as upstream CPMpy: [Apache 2.0](https://github.com/cpmpy/cpmpy/blob/master/LICENSE).
