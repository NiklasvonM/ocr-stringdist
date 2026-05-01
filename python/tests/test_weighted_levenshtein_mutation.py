from collections.abc import Callable
from typing import Any

import pytest
from ocr_stringdist import WeightedLevenshtein


@pytest.mark.parametrize(
    ["costs_attribute", "costs", "source", "target", "expected"],
    [
        ("substitution_costs", {("a", "b"): 0.2}, "a", "b", 0.2),
        ("insertion_costs", {"b": 0.3}, "", "b", 0.3),
        ("deletion_costs", {"a": 0.4}, "a", "", 0.4),
    ],
)
def test_cost_property_reassignment_updates_calculator(
    costs_attribute: str,
    costs: dict[Any, float],
    source: str,
    target: str,
    expected: float,
) -> None:
    wl = WeightedLevenshtein.unweighted()
    assert wl.distance(source, target) == pytest.approx(1.0)

    setattr(wl, costs_attribute, costs)

    assert wl.distance(source, target) == pytest.approx(expected)
    assert wl.batch_distance(source, [target]) == pytest.approx([expected])


@pytest.mark.parametrize(
    ["costs_attribute", "key", "cost", "source", "target", "expected"],
    [
        ("substitution_costs", ("a", "b"), 0.2, "a", "b", 0.2),
        ("insertion_costs", "b", 0.3, "", "b", 0.3),
        ("deletion_costs", "a", 0.4, "a", "", 0.4),
    ],
)
def test_in_place_cost_assignment_updates_calculator(
    costs_attribute: str,
    key: Any,
    cost: float,
    source: str,
    target: str,
    expected: float,
) -> None:
    wl = WeightedLevenshtein.unweighted()
    costs = getattr(wl, costs_attribute)
    assert wl.distance(source, target) == pytest.approx(1.0)

    costs[key] = cost

    assert wl.distance(source, target) == pytest.approx(expected)
    assert wl.batch_distance(source, [target]) == pytest.approx([expected])


@pytest.mark.parametrize(
    ["costs_attribute", "costs", "remove_cost", "source", "target"],
    [
        ("substitution_costs", {("a", "b"): 0.2}, lambda costs: costs.pop(("a", "b")), "a", "b"),
        ("insertion_costs", {"b": 0.3}, lambda costs: costs.pop("b"), "", "b"),
        ("deletion_costs", {"a": 0.4}, lambda costs: costs.pop("a"), "a", ""),
        ("substitution_costs", {("a", "b"): 0.2}, lambda costs: costs.clear(), "a", "b"),
        ("insertion_costs", {"b": 0.3}, lambda costs: costs.clear(), "", "b"),
        ("deletion_costs", {"a": 0.4}, lambda costs: costs.clear(), "a", ""),
    ],
)
def test_in_place_cost_removal_updates_calculator(
    costs_attribute: str,
    costs: dict[Any, float],
    remove_cost: Callable[[dict[Any, float]], Any],
    source: str,
    target: str,
) -> None:
    wl = WeightedLevenshtein.unweighted()
    setattr(wl, costs_attribute, costs)
    assert wl.distance(source, target) < 1.0

    remove_cost(getattr(wl, costs_attribute))

    assert wl.distance(source, target) == pytest.approx(1.0)
    assert wl.batch_distance(source, [target]) == pytest.approx([1.0])


@pytest.mark.parametrize(
    ["default_cost_attribute", "source", "target"],
    [
        ("default_substitution_cost", "a", "b"),
        ("default_insertion_cost", "", "b"),
        ("default_deletion_cost", "a", ""),
    ],
)
def test_default_cost_setters_update_calculator(
    default_cost_attribute: str, source: str, target: str
) -> None:
    wl = WeightedLevenshtein.unweighted()
    assert wl.distance(source, target) == pytest.approx(1.0)

    setattr(wl, default_cost_attribute, 0.25)

    assert wl.distance(source, target) == pytest.approx(0.25)
    assert wl.batch_distance(source, [target]) == pytest.approx([0.25])


def test_symmetric_substitution_setter_updates_calculator() -> None:
    wl = WeightedLevenshtein(
        substitution_costs={("a", "b"): 0.2},
        symmetric_substitution=False,
    )
    assert wl.distance("b", "a") == pytest.approx(1.0)

    wl.symmetric_substitution = True

    assert wl.distance("b", "a") == pytest.approx(0.2)
    assert wl.batch_distance("b", ["a"]) == pytest.approx([0.2])


@pytest.mark.parametrize(
    ["costs_attribute", "invalid_costs"],
    [
        ("substitution_costs", {("a", "b"): -0.1}),
        ("substitution_costs", {("a",): 0.1}),
        ("insertion_costs", {"b": -0.1}),
        ("insertion_costs", {("b",): 0.1}),
        ("deletion_costs", {"a": -0.1}),
        ("deletion_costs", {("a",): 0.1}),
    ],
)
def test_invalid_cost_property_reassignment_keeps_existing_calculator(
    costs_attribute: str, invalid_costs: dict[Any, float]
) -> None:
    wl = WeightedLevenshtein.unweighted()

    with pytest.raises((TypeError, ValueError)):
        setattr(wl, costs_attribute, invalid_costs)

    assert getattr(wl, costs_attribute) == {}
    assert wl.distance("a", "b") == pytest.approx(1.0)


@pytest.mark.parametrize(
    ["costs_attribute", "key", "cost"],
    [
        ("substitution_costs", ("a", "b"), -0.1),
        ("substitution_costs", ("a",), 0.1),
        ("insertion_costs", "b", -0.1),
        ("insertion_costs", ("b",), 0.1),
        ("deletion_costs", "a", -0.1),
        ("deletion_costs", ("a",), 0.1),
    ],
)
def test_invalid_in_place_cost_assignment_keeps_existing_calculator(
    costs_attribute: str, key: Any, cost: float
) -> None:
    wl = WeightedLevenshtein.unweighted()

    with pytest.raises((TypeError, ValueError)):
        getattr(wl, costs_attribute)[key] = cost

    assert getattr(wl, costs_attribute) == {}
    assert wl.distance("a", "b") == pytest.approx(1.0)
