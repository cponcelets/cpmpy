"""CPM_turbo.set_search: the branching order, as a MiniZinc search annotation."""

import pytest

import cpmpy as cp
from cpmpy.solvers.turbo import CPM_turbo


class _Names:
    """Stands in for the MiniZinc translator: only names variables."""

    def solver_var(self, v):
        return v.name.replace("[", "_").replace("]", "").replace(",", "_")


def _bare():
    """A CPM_turbo without turbo installed, enough to build annotations."""
    s = CPM_turbo.__new__(CPM_turbo)
    s.mzn_cpm = _Names()
    s._varmap = {}
    return s


def test_one_strategy_is_an_int_search():
    x = cp.intvar(0, 5, shape=3, name="x")
    s = _bare()
    s.set_search((x, "first_fail", "indomain_min"))
    assert s._annotated("solve minimize obj;\n") == \
        "solve :: int_search([x_0, x_1, x_2], first_fail, indomain_min) minimize obj;\n"


def test_strategies_and_bool_runs_make_a_seq_search():
    x = cp.intvar(0, 5, shape=3, name="x")
    b = cp.boolvar(shape=2, name="b")
    s = _bare()
    s.set_search(([b, x[0], x[1]], "input_order", "indomain_split"),
                 (x[2], "largest", "indomain_max"))
    assert s._annotated("solve satisfy;") == (
        "solve :: seq_search([bool_search([b_0, b_1], input_order, indomain_min), "
        "int_search([x_0, x_1], input_order, indomain_split), "
        "int_search([x_2], largest, indomain_max)]) satisfy;")


def test_no_strategy_leaves_the_solve_item_alone():
    s = _bare()
    s.set_search()
    assert s._annotated("solve satisfy;") == "solve satisfy;"


@pytest.mark.parametrize("strategy", [
    (cp.intvar(0, 5, shape=2, name="y") + 1, "input_order", "indomain_min"),
    (cp.intvar(0, 5, shape=2, name="y"), "fastest", "indomain_min"),
    (cp.intvar(0, 5, shape=2, name="y"), "input_order", "indomain_random"),
])
def test_bad_strategies_are_rejected(strategy):
    with pytest.raises(ValueError):
        _bare().set_search(strategy)


@pytest.mark.skipif(not CPM_turbo.supported(), reason="turbo is not installed")
def test_the_annotation_survives_flattening():
    x = cp.intvar(0, 5, shape=3, name="x")
    s = CPM_turbo(cp.Model(cp.AllDifferent(x), cp.sum(x) >= 6))
    s.set_search((x, "input_order", "indomain_max"))
    assert "int_search" in s.compile()
