#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## turbo.py
##
"""
    Interface to turbo's API

    Turbo is a open-source research project, aiming to be a constraint solver entirely on GPUs. 
    Turbo is part of a larger project called Lattice Land.
    https://github.com/ptal/turbo

    The turbo interface is text-based and uses the MiniZinc package:
    - the model is constructed via CPMpy's MiniZinc interface (``CPM_minizinc``)
    - just before solving, a FlatZinc version of the model is compiled and handed to turbo

    .. note::
        MiniZinc is used *only* as a MiniZinc -> FlatZinc translator here.
        The MiniZinc solver configuration (``turbo.gpu.release``) is needed for the
        globals library / FlatZinc dialect it selects, but MiniZinc never runs a search:
        ``turbo_python`` solves the compiled FlatZinc model independently.

    ============
    Installation
    ============

    Requires that the 'turbo_python' python package is installed:
    See detailed installation instructions at:
    https://github.com/cponcelets/turbo/tree/turbo_python

    ===============
    List of classes
    ===============

    .. autosummary::
        :nosignatures:

        CPM_turbo
"""

import re
import time
import warnings
from typing import Any, Optional

from .solver_interface import SolverInterface, SolverStatus, ExitStatus, Callback
from ..expressions.core import Expression, NestedBoolExprLike
from ..expressions.variables import _NumVarImpl, intvar
from ..expressions.utils import is_num, is_any_list, argvals


class _Unknown:
    """Sentinel: 'no value could be found for this name' (distinct from None/0/False)."""
    __slots__ = ()
    def __repr__(self):
        return "<unknown>"


_UNKNOWN = _Unknown()

class CPM_turbo(SolverInterface):
    """
    Interface to turbo's API

    Creates the following attributes (see parent constructor for more):

    - ``mzn_cpm``: object, the minizinc cpmpy instance (used as MiniZinc -> FlatZinc translator)
    - ``turbo_solver``: object, the turbo solver instance
    - ``turbo_best``: dict, the best solution of the previous solve (FlatZinc names -> values)
    - ``my_stats``: dict, the statistics of the previous solve
    - ``mzn_fzn``: str, the FlatZinc text handed to turbo during the last compile
    - ``mzn_ozn``: list[str], the accompanying MiniZinc output model (.ozn), used to recover the value of variables that the MiniZinc compiler removed from the FlatZinc model
    - ``flat_constants``: dict, values recovered from the .ozn model

    Documentation of the solver's own Python API:
    https://github.com/cponcelets/turbo/tree/turbo_python
    """

    ## Since using MiniZinc -> Let CPMpy flatten?
    supported_global_constraints = frozenset({#"alldifferent", "alldifferent_except0", "allequal",
                                              "inverse", "ite", "xor", "table", "InDomain", "negative_table", "mdd", "regular", "cumulative", "circuit", "gcc",
                                              "increasing", "decreasing",
                                              "strictly_increasing", "strictly_decreasing", "lex_lesseq", "lex_less",
                                              "lex_chain_less","lex_chain_lesseq",
                                              "precedence", "no_overlap",
                                              "min", "max", "abs", "mul", "div", "mod", "pow", "element", "count", "nvalue", "among", "nd_element"})
    supported_reified_global_constraints = supported_global_constraints - {"circuit", "precedence", "regular"}

    required_version = (2, 8, 2)

    #: MiniZinc solver configuration used to *translate* the model (never to solve it)
    default_flattener = "turbo.gpu.release"

    #: turbo currently needs an objective; this dummy one is injected for satisfaction problems.
    #: It is added to a throw-away copy of the MiniZinc model, so the solver object is never mutated.
    fake_objective_name = "turbo_fake_objective"

    #: turbo's '-t' option is expressed in milliseconds (`Configuration::timeout_ms`),
    #: while CPMpy's `time_limit` is in seconds. Set to False if your build expects seconds.
    time_limit_in_ms = True

    _OZN_ARRAY = re.compile(r"^\s*array\s*\[[^\]]*\]\s*of\s+(bool|int|float|string)\s*:\s*([A-Za-z_]\w*)\s*=\s*(.+?);\s*$")
    _OZN_SCALAR = re.compile(r"^\s*(bool|int|float|string)\s*:\s*([A-Za-z_]\w*)\s*=\s*(.+?);\s*$")
    
    @staticmethod
    def supported():
        # try to import the package
        try:
            import cpmpy
            import turbo_python as turbo
            from cpmpy.solvers.minizinc import CPM_minizinc
            return CPM_minizinc.supported()
        except ImportError:  # covers ModuleNotFoundError, and broken/partial installs
            return False

    @classmethod
    def version(cls) -> Optional[str]:
        """
        Returns the installed version of the solver's Python API.
        """
        from importlib.metadata import version, PackageNotFoundError
        return version("turbo_python")

    @staticmethod
    def solvernames(installed:bool=True):
        """
            Returns solvers supported by TEMPLATE (on your system).

            Arguments:
                installed (boolean): whether to filter the solvernames to those installed on your system (default True)

            Returns:
                list of solver names
        """
        return []

    @classmethod
    def solverversion(cls, subsolver: str) -> Optional[str]:
        """
        Returns the version of the requested subsolver (MiniZinc translator configuration).

        Arguments:
            subsolver (str): name of the subsolver

        Returns:
            Version number of the subsolver if installed, else None
        """
        import minizinc
        try:
            return minizinc.Solver.lookup(subsolver).version
        except LookupError:
            raise ValueError(f"Unknown subsolver '{subsolver}', "
                             f"expected one of {cls.solvernames(installed=False)}")

    def __init__(self, cpm_model=None, subsolver: Optional[str] = None, verbose: bool = False):
        """
        Constructor of the native solver object

        Arguments:
            cpm_model: Model(), a CPMpy Model() (optional)
            subsolver: str, name of the MiniZinc solver configuration used as
                       MiniZinc -> FlatZinc translator (default: 'turbo.gpu.release').
                       turbo always performs the search itself.
            verbose: bool, print the MiniZinc and FlatZinc models on every compile
        """
        if not self.supported():
            raise ModuleNotFoundError("CPM_turbo: Install the python package 'cpmpy[turbo]' to use this solver interface.")

        import cpmpy
        import turbo_python
        assert subsolver is None # unless you support subsolvers, see pysat or minizinc

        # initialise the native solver object
        self.mzn_cpm = cpmpy.SolverLookup.get("minizinc:turbo.gpu.release")
        self.turbo_solver = None
        self.turbo_best = None
        self.my_stats = None
        self.verbose = verbose

        # compilation artifacts of the last `compile()` call
        self.mzn_fzn = None          # FlatZinc text handed to turbo
        self.mzn_ozn = None          # MiniZinc output model (.ozn), as a list of lines
        self.flat_constants = None   # values recovered from the .ozn model

        # whether the *user* model had an objective during the last solve
        self._had_objective = False

        # initialise everything else and post the constraints/objective
        super().__init__(name="turbo", cpm_model=cpm_model)


    @property
    def native_model(self):
        """
            Returns the solver's underlying native model (for direct solver access).

            This is the MiniZinc model object; `native_model` is a property on
            CPM_minizinc, so it must not be called.
        """
        return self.mzn_cpm.native_model

    def _flatten(self, add_fake_objective: bool = False, **flat_kwargs):
        """
            Compile the MiniZinc model to FlatZinc.

            Works on a throw-away copy of the MiniZinc model (like CPM_minizinc._pre_solve does),
            so `add_fake_objective` never leaks into the solver object, 
            and both the .fzn and the .ozn are kept (CPM_minizinc.flatzinc_string() discards the .ozn).

            Returns:
                (fzn_text: str, ozn_lines: list[str])
        """
        import minizinc

        mzn = self.mzn_cpm
        copy_model = mzn.mzn_model.__copy__()  # it is implemented
        if add_fake_objective:
            # turbo's `best()` only reports correct variable values when the objective
            # is a variable the search actually branches on with a real domain; a
            # freshly declared var with a trivial/constraint-fixed domain makes it
            # silently return sentinel values for every variable instead (verified:
            # `var 0..0: dummy; solve minimize dummy;` and even a wider-domain dummy
            # constrained to a constant both reproduce it). So reuse an existing user
            # variable as a no-op objective rather than declare a new one.
            user_vars = list(mzn.user_vars)
            if user_vars:
                dummy = mzn.solver_var(user_vars[0])
                copy_model.add_string(f"solve minimize {dummy};\n")
            else:  # no variable to reuse (a constant-only model) - fall back
                copy_model.add_string(f"var 0..0: {self.fake_objective_name};\n")
                copy_model.add_string(f"solve minimize {self.fake_objective_name};\n")
        else:
            copy_model.add_string(mzn.mzn_txt_solve)

        inst = minizinc.Instance(mzn.mzn_solver, copy_model)
        with inst.flat(**flat_kwargs) as (fzn, ozn, statistics):
            with open(fzn.name) as f:
                fzn_text = f.read()
            ozn_lines = []
            if ozn is not None:
                with open(ozn.name) as f:
                    ozn_lines = f.readlines()
        return fzn_text, ozn_lines

    def compile(self, add_fake_objective: Optional[bool] = None, **flat_kwargs) -> str:
        """
            Returns the FlatZinc model (and caches it, together with its .ozn).

            Arguments:
                add_fake_objective: whether to inject the dummy objective turbo needs for satisfaction problems.
                                    Default: only when the user model has no objective.
        """
        if add_fake_objective is None:
            add_fake_objective = not self.has_objective()

        if self.verbose:
            print(f"MiniZinc: =====\n{self.mzn_cpm.minizinc_string()}")

        fzn_text, ozn_lines = self._flatten(add_fake_objective, **flat_kwargs)

        # cache, and invalidate anything derived from a previous compile
        self.mzn_fzn = fzn_text
        self.mzn_ozn = ozn_lines
        self.flat_constants = None

        if self.verbose:
            print(f"FlatZinc: =====\n{fzn_text}")

        return fzn_text

    def minizinc_model(self) -> str:
        """
            Returns the model in the MiniZinc language.

            Never contains the dummy objective: it is only added to the throw-away copy
            that is compiled to FlatZinc.
        """
        return self.mzn_cpm.minizinc_string()

    def flatzinc_model(self, cached: bool = True) -> str:
        """
            Returns the model in the FlatZinc language.

            Arguments:
                cached: if True (default) and a solve already happened, returns the exact FlatZinc text that was handed to turbo 
                        (dummy objective included for satisfaction problems).
                        If False, recompiles the user model as-is.
        """
        if cached and self.mzn_fzn is not None:
            return self.mzn_fzn
        return self.compile(add_fake_objective=False)

    def ozn_model(self) -> Optional[str]:
        """
            Returns the MiniZinc output model (.ozn) of the last compile, or None.
        """
        if self.mzn_ozn is None:
            return None
        return "".join(self.mzn_ozn)

    def turbo_stats(self):
        return self.my_stats

    # ------------------------------------------------------------------ #
    #  Solving                                                           #
    # ------------------------------------------------------------------ #

    _stat_aliases = {
        "num_solutions": ("num_solutions", "solutions", "nb_solutions"),
        "exhaustive":    ("exhaustive", "search_exhaustive", "is_exhaustive", "complete"),
        "solve_time":    ("solveTime", "solve_time", "solveTime_s"),
        "objective":     ("optimization", "objective", "best_objective", "best_bound"),
    }

    def _stat(self, key: str, default=None):
        """Read a statistic from turbo's stats() dict, tolerating key renamings."""
        if not self.my_stats:
            return default
        for alias in self._stat_aliases.get(key, (key,)):
            if alias in self.my_stats:
                return self.my_stats[alias]
        return default

    def _build_argv(self, time_limit, kwargs) -> list:
        """Translate CPMpy kwargs into a turbo command line."""
        argv = []
        for key, value in kwargs.items():
            flag = f"-{key}" if len(key) == 1 else f"--{key}"
            if isinstance(value, bool):
                if not value:
                    continue  # a False flag is simply not passed
                argv.append(flag)
                if key in ("verbose", "v"):
                    self.verbose = True
            else:  # non-boolean parameters get a value (this branch was unreachable before)
                argv.extend([flag, str(value)])

        # turbo needs '-a' to report intermediate solutions; don't pass it twice
        if not any(a in ("-a", "--all", "--print-intermediate-solutions") for a in argv):
            argv.insert(0, "-a")

        if time_limit is not None:
            if time_limit <= 0:
                raise ValueError("Time limit must be positive")
            if not any(a in ("-t", "--timeout") for a in argv):
                # turbo expects an integer number of milliseconds (Configuration::timeout_ms)
                t = int(round(time_limit * 1000)) if self.time_limit_in_ms else int(round(time_limit))
                argv.extend(["-t", str(max(t, 1))])

        return argv

    def solve(self, time_limit:Optional[float]=None, **kwargs):
        """
            Call the turbo solver (the instance is created from the flatzinc )

            Arguments:
            - time_limit:  maximum solve time in seconds (float, optional)
            - display:     generic solution callback for use during optimization.
                           either a list of CPMpy expressions, OR a callback function which
                           gets called after the variable-value mapping of the intermediate solution.
                           default/None: nothing is displayed
            - kwargs:      any keyword argument, sets parameters of solver object

            Arguments that correspond to solver parameters:
            https://github.com/ptal/turbo
        """

        import turbo_python
        from pathlib import Path
        import tempfile

        had_objective = self.has_objective()
        self._had_objective = had_objective

        # ensure all vars are known to solver
        self.user_vars = self.mzn_cpm.user_vars
        self.solver_vars(list(self.user_vars))
        # ensure all vars are known to the minizinc translator
        self.mzn_cpm.solver_vars(list(self.mzn_cpm.user_vars))

         # command line
        argv = self._build_argv(time_limit, kwargs)

        # write turbo's input to a temporary file
        flat_model = self.compile(add_fake_objective=not had_objective)
        tmp = tempfile.NamedTemporaryFile(suffix=".fzn", mode="w", delete=False)
        tmp.write(flat_model)
        tmp.close()
        path = Path(tmp.name)
        argv.append(str(path))

        if self.verbose:
            print(f"turbo argv: {argv}")

        self.turbo_solver = turbo_python.Turbo(argv)

        # fresh status for this run
        self.cpm_status = SolverStatus(self.name)
        self.my_stats = None
        self.turbo_best = None
        self.objective_value_ = None

        t0 = time.time()
        try:
            # call the solver, with parameters
            self.turbo_solver.solve()
            self.my_stats = self.turbo_solver.stats()
            self.cpm_status.runtime = self._stat("solve_time", time.time() - t0)

            num_solutions = self._stat("num_solutions", 0)
            exhaustive = bool(self._stat("exhaustive", False))

            # CSP:                         COP:
            # ├─ sat -> FEASIBLE           ├─ optimal -> OPTIMAL
            # ├─ unsat -> UNSATISFIABLE    ├─ sub-optimal -> FEASIBLE
            # └─ timeout -> UNKNOWN        ├─ unsat -> UNSATISFIABLE
            #                              └─ timeout -> UNKNOWN
            if num_solutions is None:
                raise NotImplementedError(f"turbo did not report a solution count: {self.my_stats}")
            elif num_solutions > 0:
                if had_objective and exhaustive:
                    self.cpm_status.exitstatus = ExitStatus.OPTIMAL
                else:
                    # a satisfaction problem, or an optimisation run that was cut short
                    self.cpm_status.exitstatus = ExitStatus.FEASIBLE
            elif exhaustive:
                # search space explored without a solution -> unsatisfiable
                self.cpm_status.exitstatus = ExitStatus.UNSATISFIABLE
            else:
                # no solution *and* the search was interrupted (timeout, memory, ...)
                self.cpm_status.exitstatus = ExitStatus.UNKNOWN

        except turbo_python.Timeout:
            self.cpm_status.exitstatus = ExitStatus.UNKNOWN
            try:  # stats may still be available after a timeout
                self.my_stats = self.turbo_solver.stats()
            except Exception:
                pass
            self.cpm_status.runtime = self._stat("solve_time", time.time() - t0)

        except turbo_python.ParseError as e:
            self.cpm_status.exitstatus = ExitStatus.ERROR
            self.cpm_status.runtime = time.time() - t0
            print(f"Parse error: {e}")
            raise e

        except Exception as e:
            self.cpm_status.exitstatus = ExitStatus.ERROR
            self.cpm_status.runtime = time.time() - t0
            print(f"Exception: {e}")
            raise e

        finally:
            path.unlink(missing_ok=True)

        # True/False depending on self.cpm_status
        has_sol = self._solve_return(self.cpm_status)

        # translate solution values (of user specified variables only)
        if has_sol:
            self.turbo_best = dict(self.turbo_solver.best())
            # fill in variable values
            for cpm_var in self.mzn_cpm.user_vars:
                value = self._value_of(cpm_var.name)
                if value is _UNKNOWN:
                    self._raise_unknown(cpm_var.name)
                # FlatZinc booleans may come back as 0/1
                cpm_var._value = bool(value) if cpm_var.is_bool() else int(value)

            # translate objective, for optimisation problems only
            if had_objective:
                self.objective_value_ = self._objective_value()

        else:  # clear values of variables
            for cpm_var in self.mzn_cpm.user_vars:
                cpm_var.clear()

        return has_sol

    def _objective_value(self):
        """
            Value of the objective of the last solve.

            Preferred: evaluate the CPMpy objective expression on the returned solution
            (exact, and independent of how turbo names/reports its bound).
            Fallback: turbo's statistics.
        """
        obj = getattr(self.mzn_cpm, "objective_", None)
        if obj is not None:
            try:
                if isinstance(obj, Expression):
                    val = obj.value()
                elif is_num(obj):
                    val = obj
                else:  # e.g. FloatSum
                    val = obj.value()
                if val is not None:
                    return val
            except Exception:
                pass
        return self._stat("objective")

    # ------------------------------------------------------------------ #
    #  Value recovery (turbo output + .ozn model)                        #
    # ------------------------------------------------------------------ #

    def _value_of(self, var_name: str, _seen: Optional[set] = None):
        """
            Value of a variable, by name. Returns `_UNKNOWN` if it cannot be found.

            A name can be:
            - a FlatZinc name reported by turbo
            - a CPMpy/user name, mapped to its MiniZinc name by CPM_minizinc._varmap
            - a name the MiniZinc compiler removed from the FlatZinc model, whose value
              is then recovered from the .ozn output model
        """
        if _seen is None:
            _seen = set()
        if var_name in _seen:
            return _UNKNOWN
        _seen.add(var_name)

        if self.turbo_best and var_name in self.turbo_best:
            return self.turbo_best[var_name]  # it was a fzn name, present in the result

        # it was a user var name: follow the CPMpy name -> MiniZinc name mapping
        mzn_name = self.mzn_cpm._varmap.get(var_name)
        if mzn_name is not None and mzn_name != var_name:
            value = self._value_of(mzn_name, _seen)
            if value is not _UNKNOWN:
                return value

        # otherwise it may have been optimised away by the MiniZinc compiler
        if self.flat_constants is None:
            self.parse_ozn()
        if var_name in self.flat_constants:
            return self.flat_constants[var_name]

        return _UNKNOWN

    def _raise_unknown(self, var_name: str):
        print(f"turbo output:{self.turbo_best}")
        print(f"self.mzn_cpm._varmap:{self.mzn_cpm._varmap}")
        print(f"self.flat_constants:{self.flat_constants}")
        raise ValueError(f"Var {var_name} is unknown to the MiniZinc-turbo solver, "
                         f"this is unexpected - please report on github...")

    def get_value_with_ozn(self, var_name: str, is_from_ozn: bool = False):
        """
            Backwards-compatible wrapper around `_value_of()`.

            Returns None instead of raising when `is_from_ozn` is True (the caller then
            knows the value may appear later in the .ozn file).
        """
        value = self._value_of(var_name)
        if value is _UNKNOWN:
            if is_from_ozn:
                return None
            self._raise_unknown(var_name)
        return value

    def parse_ozn(self):
        """
            Parse the MiniZinc output model (.ozn) for the values of variables that the
            compiler removed from the FlatZinc model, into `self.flat_constants`.

            Handles scalar and array declarations, and declarations whose right-hand side
            refers to another name (resolved recursively, in any order).
        """
        self.flat_constants = dict()

        lines = self.mzn_ozn
        if lines is None:  # tolerate a patched CPM_minizinc that exposes the .ozn itself
            lines = getattr(self.mzn_cpm, "mzn_ozn", None)
        if lines is None:
            return
        if isinstance(lines, str):
            lines = lines.splitlines()

        # Scan every line: a declaration with a value is unambiguous, and depending on
        # the MiniZinc version/output-mode the declarations can sit before *or* after the
        # output statement (keying on a '["dzn"]' marker line silently finds nothing on
        # MiniZinc 2.8.x).
        raw = dict()  # name -> (type, right-hand side)
        for line in lines:
            for pattern in (self._OZN_ARRAY, self._OZN_SCALAR):
                m = pattern.match(line)
                if m:
                    typ, name, value = m.groups()
                    raw[name] = (typ, value.strip())
                    break

        for name in raw:
            value = self._resolve_ozn(name, raw, set())
            if value is not _UNKNOWN:
                self.flat_constants[name] = value

    def _resolve_ozn(self, name: str, raw: dict, seen: set):
        """Resolve one .ozn declaration to a Python value (or `_UNKNOWN`)."""
        if name in self.flat_constants:
            return self.flat_constants[name]
        if self.turbo_best and name in self.turbo_best:
            return self.turbo_best[name]
        if name in seen or name not in raw:
            return _UNKNOWN
        seen.add(name)

        typ, text = raw[name]
        value = self._parse_ozn_value(text, typ, raw, seen)
        if value is not _UNKNOWN:
            self.flat_constants[name] = value
        return value

    def _parse_ozn_value(self, text: str, typ: str, raw: dict, seen: set):
        """Parse a .ozn right-hand side: literal, identifier, or (nested) array literal."""
        text = text.strip()

        # arrayNd(1..3, 1..2, [ ... ]) -> keep the array literal
        m = re.match(r"^array\d*d\s*\((.*)\)$", text, flags=re.DOTALL)
        if m:
            inner = m.group(1)
            start = inner.find("[")
            if start != -1:
                text = inner[start:inner.rfind("]") + 1]

        if text.startswith("[") and text.endswith("]"):
            elements = self._split_top_level(text[1:-1])
            values = [self._parse_ozn_value(e, typ, raw, seen) for e in elements if e.strip()]
            if any(v is _UNKNOWN for v in values):
                return _UNKNOWN
            return values

        # literals
        if text in ("true", "false"):
            return text == "true"
        try:
            if typ == "int":
                return int(text)
            if typ == "float":
                return float(text)
            if typ == "bool":
                return bool(int(text))
        except ValueError:
            pass
        if typ == "string":
            return text.strip('"')

        # an identifier: another .ozn declaration, a turbo variable, or a user var
        if re.match(r"^[A-Za-z_]\w*$", text):
            value = self._resolve_ozn(text, raw, seen)
            if value is not _UNKNOWN:
                return value
            mzn_name = self.mzn_cpm._varmap.get(text)
            if mzn_name is not None and mzn_name != text:
                return self._resolve_ozn(mzn_name, raw, seen)
            return _UNKNOWN

        return _UNKNOWN

    @staticmethod
    def _split_top_level(text: str) -> list:
        """Split on commas that are not nested inside brackets or parentheses."""
        parts, depth, current = [], 0, []
        for ch in text:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(ch)
        if current:
            parts.append("".join(current))
        return parts

    # ------------------------------------------------------------------ #
    #  Model building (delegated to the MiniZinc translator)             #
    # ------------------------------------------------------------------ #

    def solver_var(self, cpm_var):
        """
            Creates solver variable for cpmpy variable
            or returns from cache if previously created
            or returns a constant if the variable is a constant
        """
        revar = self.mzn_cpm.solver_var(cpm_var)
        # reverse map, for the variables the MiniZinc compiler may rename/remove
        if revar is not None and isinstance(cpm_var, _NumVarImpl):
            self._varmap[revar] = cpm_var
        return revar

    def objective(self, expr, minimize=True):
        """
            Post the given expression to the solver as objective to minimize/maximize

            'objective()' can be called multiple times, only the last one is stored

            (technical side note: any constraints created during conversion of the objective
            are permanently posted to the solver)
        """
        self.mzn_cpm.objective(expr, minimize)

    def has_objective(self):
        return self.mzn_cpm.has_objective()

    # `add()` first calls `transform()`
    def transform(self, cpm_expr: NestedBoolExprLike) -> list[Expression]:
        """
            Transform arbitrary CPMpy expressions to constraints the solver supports

            Implemented through chaining multiple solver-independent **transformation functions** from
            the `cpmpy/transformations/` directory.

            See the 'Adding a new solver' docs on readthedocs for more information.

            Arguments:
                cpm_expr (NestedBoolExprLike): CPMpy expression, or list thereof

            Returns:
                list[Expression]: transformed constraints
        """
        return self.mzn_cpm.transform(cpm_expr)

    def add(self, cpm_expr: NestedBoolExprLike) -> "CPM_turbo":
        """
            Eagerly add a constraint to the underlying solver.

            Any CPMpy expression given is immediately transformed (through `transform()`)
            and then posted to the solver in this function.

            This can raise 'NotImplementedError' for any constraint not supported after transformation

            The variables used in expressions given to add are stored as 'user variables'. Those are the only ones
            the user knows and cares about (and will be populated with a value after solve). All other variables
            are auxiliary variables created by transformations.

            Arguments:
                cpm_expr (NestedBoolExprLike): CPMpy expression, or list thereof

            Returns:
                self
        """
        self.mzn_cpm.add(cpm_expr)
        # a new constraint invalidates the cached compilation
        self.mzn_fzn = self.mzn_ozn = self.flat_constants = None
        return self
    __add__ = add  # avoid redirect in superclass

    # Other functions from SolverInterface that you can overwrite:
    # solveAll, solution_hint, get_core

    def solveAll(self, display: Optional[Callback] = None, time_limit: Optional[float] = None,
                 solution_limit: Optional[int] = None, call_from_model=False, **kwargs):
        """
            A shorthand to (efficiently) compute all (optimal) solutions, map them to CPMpy
            and optionally display the solutions.

            .. warning::
                turbo's Python API only exposes the *best* solution (`best()`), so this
                cannot enumerate solutions the way other CPMpy solvers do: it returns the
                number of solutions turbo reports and, if `display` is given, displays the
                best/last one only. For an optimisation problem, '-a' makes turbo report
                *improving* solutions, so the count is a number of improvements, not a
                number of distinct optimal solutions.

            Arguments:
                - display: either a list of CPMpy expressions, OR a callback function, called with the variables after value-mapping
                        default/None: nothing displayed
                - time_limit: stop after this many seconds (default: None)
                - solution_limit: stop after this many solutions (default: None)
                - call_from_model: whether the method is called from a CPMpy Model instance or not
                - any other keyword argument

            Returns: number of solutions found
        """
        if solution_limit is not None:
            warnings.warn("CPM_turbo: 'solution_limit' is not supported by turbo, ignoring it.")

        kwargs.setdefault("a", True)  # report intermediate solutions
        self.solve(time_limit=time_limit, **kwargs)  # note: **kwargs, not kwargs

        if display is not None:
            warnings.warn("CPM_turbo: turbo only exposes its best solution, "
                          "'display' is called once, on that solution.")
            if self._solve_return(self.cpm_status):
                self.print_display(display)

        return self._stat("num_solutions", 0)