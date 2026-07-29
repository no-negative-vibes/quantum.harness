from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from oracle.exterior_category_search import (
    TypedEdge,
    TypedExteriorGraph,
    certificate_from_one_particle_charts,
    has_induced_diagonal_charts,
    solve_diagonal_grade_charts,
)


def _coboundary_control() -> tuple[
    TypedExteriorGraph,
    dict[str, sp.ImmutableMatrix],
]:
    chart_b = sp.ImmutableMatrix(
        [
            [-1, 0, 4, -2],
            [-2, 1, 4, -2],
            [2, -2, -3, 2],
            [1, 0, -2, 1],
        ]
    )
    coordinate_1 = sp.ImmutableMatrix(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 2, 1],
        ]
    )
    coordinate_2 = sp.ImmutableMatrix(
        [
            [1, 0, 0, 0],
            [3, 3, 2, 4],
            [0, 1, 1, 2],
            [0, 0, 0, 1],
        ]
    )
    identity = sp.ImmutableMatrix(sp.eye(4))
    graph = TypedExteriorGraph(
        (
            TypedEdge("forward-1", "a", "b", chart_b * coordinate_1),
            TypedEdge("forward-2", "a", "b", chart_b * coordinate_2),
            TypedEdge("return", "b", "a", chart_b.inv()),
        )
    )
    return graph, {"a": identity, "b": chart_b}


def test_typed_word_validation_rejects_a_source_target_mismatch() -> None:
    graph, _ = _coboundary_control()

    assert graph.is_legal_word(("forward-1", "return"))
    assert graph.is_closed_word(("forward-1", "return"))
    assert not graph.is_legal_word(("forward-1", "forward-2"))
    assert not graph.is_closed_word(("forward-1",))

    with pytest.raises(ValueError, match="source"):
        graph.word_product(("forward-1", "forward-2"), require_legal=True)


def test_induced_charts_prove_legal_paths_but_types_hide_an_exact_negative_word() -> None:
    graph, one_particle_charts = _coboundary_control()
    certificate = certificate_from_one_particle_charts(
        graph,
        one_particle_charts,
    )

    assert certificate.is_entrywise_nonnegative()
    assert certificate.proof_origin == "one_particle_charts"
    legal = certificate.certify_closed_word(
        ("forward-1", "return", "forward-2", "return")
    )
    assert legal.weight == sum(legal.grade_traces)
    assert legal.weight > 0

    forbidden_product = graph.word_product(("forward-1", "forward-2"))
    forbidden_weight = sp.det(sp.eye(4) + forbidden_product)
    assert forbidden_weight == -12


def test_diagonal_grade_chart_solver_returns_a_verifiable_certificate() -> None:
    sign_flip = sp.ImmutableMatrix(sp.diag(-1, -1, 1, 1))
    graph = TypedExteriorGraph(
        (
            TypedEdge("out", "a", "b", sign_flip),
            TypedEdge("back", "b", "a", sign_flip),
        )
    )

    certificate = solve_diagonal_grade_charts(graph)

    assert certificate is not None
    assert certificate.is_entrywise_nonnegative(tolerance=1e-12)
    assert has_induced_diagonal_charts(graph)
    assert certificate.has_induced_diagonal_sign_certificate
    assert certificate.certify_closed_word(("out", "back")).weight > 0


def test_diagonal_grade_chart_solver_detects_a_sign_contradiction() -> None:
    identity = sp.ImmutableMatrix(sp.eye(4))
    one_sign_changed = sp.ImmutableMatrix(sp.diag(-1, 1, 1, 1))
    graph = TypedExteriorGraph(
        (
            TypedEdge("plain", "a", "b", identity),
            TypedEdge("conflict", "a", "b", one_sign_changed),
        )
    )

    assert solve_diagonal_grade_charts(graph) is None


def test_support_aware_candidate_needs_grade_specific_diagonal_sign_charts() -> None:
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
    graph = TypedExteriorGraph(
        (
            TypedEdge("x", "a", "b", matrices[0]),
            TypedEdge("y", "a", "b", matrices[1]),
            TypedEdge("return", "b", "a", matrices[2]),
        )
    )

    certificate = solve_diagonal_grade_charts(graph)

    assert certificate is not None
    assert certificate.is_entrywise_nonnegative()
    assert not has_induced_diagonal_charts(graph)
    assert not certificate.has_induced_diagonal_sign_certificate
    assert sp.det(sp.eye(4) + graph.word_product(("y",))) == -2
    assert certificate.certify_closed_word(("x", "return")).weight == 24


def test_certificate_refuses_to_certify_an_open_path() -> None:
    graph, charts = _coboundary_control()
    certificate = certificate_from_one_particle_charts(graph, charts)

    with pytest.raises(ValueError, match="closed"):
        certificate.certify_closed_word(("forward-1",))


def test_float_screen_cannot_be_promoted_to_an_arbitrary_depth_certificate() -> None:
    epsilon = 5e-11
    graph = TypedExteriorGraph(
        (
            TypedEdge(
                "out",
                "a",
                "b",
                np.array([[0.0, 1e10], [-epsilon, 0.0]]),
            ),
            TypedEdge(
                "back",
                "b",
                "a",
                np.array([[0.0, 4e10], [-epsilon, 0.0]]),
            ),
        )
    )

    with pytest.raises(TypeError, match="exact rational"):
        certificate_from_one_particle_charts(
            graph,
            {"a": np.eye(2), "b": np.eye(2)},
        )
    with pytest.raises(TypeError, match="exact rational"):
        solve_diagonal_grade_charts(graph)


def test_sympy_float_is_not_mislabelled_as_exact() -> None:
    graph = TypedExteriorGraph(
        (
            TypedEdge(
                "loop",
                "a",
                "a",
                sp.ImmutableMatrix([[sp.Float(1.0), 0], [0, 1]]),
            ),
        )
    )

    with pytest.raises(TypeError, match="exact rational"):
        solve_diagonal_grade_charts(graph)
