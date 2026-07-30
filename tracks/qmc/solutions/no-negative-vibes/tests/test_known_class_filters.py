from __future__ import annotations

import sympy as sp

from oracle.exterior_category_search import (
    TypedEdge,
    TypedExteriorGraph,
    has_induced_diagonal_charts,
)
from oracle.known_class_filters import (
    apply_object_permutations,
    find_signed_permutation_tn_coboundary,
)


def _apparent_grade_specific_candidate() -> TypedExteriorGraph:
    matrices = tuple(
        sp.ImmutableMatrix(entries)
        for entries in (
            [
                [0, 0, 0, 2],
                [1, 0, 0, 0],
                [-2, 0, 1, 0],
                [0, 1, 0, 0],
            ],
            [
                [0, 0, 0, 2],
                [1, 0, 0, 4],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
            ],
            [
                [0, 1, 0, 0],
                [0, 0, 2, 1],
                [0, 0, 1, 0],
                [1, 0, 0, 0],
            ],
        )
    )
    return TypedExteriorGraph(
        (
            TypedEdge("x", "a", "b", matrices[0]),
            TypedEdge("y", "a", "b", matrices[1]),
            TypedEdge("return", "b", "a", matrices[2]),
        )
    )


def test_signed_permutation_filter_catches_the_apparent_minus_two_candidate() -> None:
    graph = _apparent_grade_specific_candidate()
    assert not has_induced_diagonal_charts(graph)

    witness = find_signed_permutation_tn_coboundary(graph)

    assert witness is not None
    assert witness.tested_assignments <= 24**2
    transformed = apply_object_permutations(
        graph,
        witness.object_permutations,
    )
    assert has_induced_diagonal_charts(transformed)
    assert sp.det(sp.eye(4) + graph.word_product(("y",))) == -2


def test_signed_permutation_filter_rejects_a_negative_closed_loop() -> None:
    graph = TypedExteriorGraph(
        (
            TypedEdge(
                "loop",
                "a",
                "a",
                sp.ImmutableMatrix(sp.diag(-2, 1, 1, 1)),
            ),
        )
    )

    assert find_signed_permutation_tn_coboundary(graph) is None
