# <b>CPMpy-turbo</b>: Integrating the <a href="https://github.com/ptal/turbo">turbo</a> solver into <a href="https://github.com/CPMpy/cpmpy">CPMpy</a>

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

Requires the `turbo_python` package and MiniZinc. See detailed installation instructions at https://github.com/cponcelets/turbo/tree/turbo_python

```
pip install "cpmpy[turbo] @ git+https://github.com/cponcelets/cpmpy@turbo"
```

You need to set the `turbo.gpu.release.msc` for configuring MiniZinc flattening function. (Install scripts to be added)

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
