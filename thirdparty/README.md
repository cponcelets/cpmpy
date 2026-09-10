# Third-party install scripts

`CPM_turbo` has two direct dependencies:

- **MiniZinc**: needs both the `minizinc` python package and the MiniZinc binary bundle (used only as a MiniZinc -> FlatZinc translator, see [cpmpy/solvers/turbo.py](../cpmpy/solvers/turbo.py))
- **turbo_python**: a CUDA/CMake build of [turbo](https://github.com/cponcelets/turbo/tree/turbo_python), the actual GPU solver

`install_turbo.sh` automates both, plus registering the `turbo.gpu.release.msc` MiniZinc solver configuration. See the script's own `--help` (or the header comment) for options and what it deliberately leaves to you (installing CUDA/cmake/doxygen themselves - those are system/hardware-sensitive and not something worth guessing at).

`wsl.patch` disables concurrent managed memory in turbo's CMake config, which WSL2's CUDA doesn't support; the script detects WSL and applies it automatically.

```
./thirdparty/install_turbo.sh
```
