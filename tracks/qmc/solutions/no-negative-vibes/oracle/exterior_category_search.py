"""Typed exterior-cone certificates for constrained determinant histories.

The edge order is chronological: for the word ``(e1, e2)`` the one-particle
product is ``B_e2 @ B_e1``.  Positivity is certified only for legal closed
paths.  Erasing the source/target labels deliberately produces a larger free
alphabet whose words are available for counterexample searches.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Hashable, Iterable, Mapping, Sequence

import numpy as np
import sympy as sp

from .compound_cones import (
    chronological_product,
    compound_matrix,
    exterior_character_sum,
    subset_basis,
)


Matrix = np.ndarray | sp.MatrixBase
ObjectLabel = Hashable


def _shape(matrix: Matrix) -> tuple[int, int]:
    return tuple(int(value) for value in matrix.shape)


def _is_exact(matrix: Matrix) -> bool:
    return isinstance(matrix, sp.MatrixBase)


def _is_exact_rational_matrix(matrix: Matrix) -> bool:
    return _is_exact(matrix) and all(
        sp.sympify(value).is_Rational is True for value in matrix
    )


def _require_exact_rational_graph(graph: "TypedExteriorGraph") -> None:
    if not all(_is_exact_rational_matrix(edge.matrix) for edge in graph.edges):
        raise TypeError(
            "theorem certificates require exact rational SymPy edge matrices"
        )


def _immutable(matrix: Matrix) -> Matrix:
    if _is_exact(matrix):
        return sp.ImmutableMatrix(matrix)
    array = np.array(matrix, copy=True)
    array.setflags(write=False)
    return array


def _inverse(matrix: Matrix) -> Matrix:
    if _is_exact(matrix):
        return sp.ImmutableMatrix(matrix.inv())
    return np.linalg.inv(np.asarray(matrix))


def _trace(matrix: Matrix):
    if _is_exact(matrix):
        return sp.trace(matrix)
    return np.trace(np.asarray(matrix))


def _entry_sign(value: object, *, tolerance: float) -> int | None:
    """Return -1, 0, or +1; ``None`` means numerically ambiguous."""

    if isinstance(value, sp.Basic):
        simplified = sp.simplify(value)
        if simplified == 0:
            return 0
        if simplified.is_positive:
            return 1
        if simplified.is_negative:
            return -1
        return None
    else:
        numerical = complex(value)
    if abs(numerical.imag) > tolerance:
        return None
    if abs(numerical.real) <= tolerance:
        return 0
    return 1 if numerical.real > 0 else -1


def _matrix_is_nonnegative(matrix: Matrix, *, tolerance: float) -> bool:
    values = matrix if _is_exact(matrix) else np.asarray(matrix).flat
    for value in values:
        sign = _entry_sign(value, tolerance=tolerance)
        if sign is None or sign < 0:
            return False
    return True


def _matrices_equal(left: Matrix, right: Matrix, *, tolerance: float) -> bool:
    if _shape(left) != _shape(right):
        return False
    if _is_exact(left) or _is_exact(right):
        difference = sp.Matrix(left) - sp.Matrix(right)
        return all(sp.simplify(value) == 0 for value in difference)
    return bool(
        np.allclose(
            np.asarray(left),
            np.asarray(right),
            rtol=tolerance,
            atol=tolerance,
        )
    )


@dataclass(frozen=True)
class TypedEdge:
    """One one-particle propagator with declared source and target objects."""

    name: str
    source: ObjectLabel
    target: ObjectLabel
    matrix: Matrix

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("edge name must be nonempty")
        shape = _shape(self.matrix)
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError("edge matrix must be square")
        object.__setattr__(self, "matrix", _immutable(self.matrix))


class TypedExteriorGraph:
    """Finite typed alphabet with strict chronological composition."""

    def __init__(self, edges: Iterable[TypedEdge]):
        edge_tuple = tuple(edges)
        if not edge_tuple:
            raise ValueError("at least one typed edge is required")
        dimensions = {_shape(edge.matrix)[0] for edge in edge_tuple}
        if len(dimensions) != 1:
            raise ValueError("all edge matrices must have the same dimension")
        names = [edge.name for edge in edge_tuple]
        if len(set(names)) != len(names):
            raise ValueError("edge names must be unique")
        self._edges = edge_tuple
        self._by_name = {edge.name: edge for edge in edge_tuple}
        self._dimension = dimensions.pop()
        self._objects = tuple(
            dict.fromkeys(
                label
                for edge in edge_tuple
                for label in (edge.source, edge.target)
            )
        )

    @property
    def edges(self) -> tuple[TypedEdge, ...]:
        return self._edges

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def objects(self) -> tuple[ObjectLabel, ...]:
        return self._objects

    def word_edges(self, word: Sequence[str]) -> tuple[TypedEdge, ...]:
        if not word:
            raise ValueError("word must contain at least one edge")
        try:
            return tuple(self._by_name[name] for name in word)
        except KeyError as error:
            raise ValueError(f"unknown edge name: {error.args[0]}") from error

    def is_legal_word(self, word: Sequence[str]) -> bool:
        try:
            edges = self.word_edges(word)
        except ValueError:
            return False
        return all(
            left.target == right.source
            for left, right in zip(edges, edges[1:])
        )

    def is_closed_word(self, word: Sequence[str]) -> bool:
        if not self.is_legal_word(word):
            return False
        edges = self.word_edges(word)
        return edges[-1].target == edges[0].source

    def word_product(
        self,
        word: Sequence[str],
        *,
        require_legal: bool = False,
    ) -> Matrix:
        edges = self.word_edges(word)
        if require_legal:
            for left, right in zip(edges, edges[1:]):
                if left.target != right.source:
                    raise ValueError(
                        f"edge {right.name!r} has source {right.source!r}, "
                        f"expected source {left.target!r}"
                    )
        return chronological_product([edge.matrix for edge in edges])


@dataclass(frozen=True)
class ClosedWordCertificate:
    word: tuple[str, ...]
    grade_traces: tuple[object, ...]
    weight: object
    direct_weight: object
    proof_level: str = "exact_rational"


@dataclass(frozen=True)
class GradeChartCertificate:
    """Simplicial charts in every exterior grade for every typed object."""

    graph: TypedExteriorGraph
    charts: Mapping[int, Mapping[ObjectLabel, Matrix]]
    proof_origin: str
    has_induced_diagonal_sign_certificate: bool

    def __post_init__(self) -> None:
        _require_exact_rational_graph(self.graph)
        if self.proof_origin not in {
            "one_particle_charts",
            "grade_diagonal_sign_charts",
        }:
            raise ValueError("unknown grade-chart proof origin")
        frozen: dict[int, Mapping[ObjectLabel, Matrix]] = {}
        for grade, object_charts in self.charts.items():
            exact_charts: dict[ObjectLabel, Matrix] = {}
            for label, chart in object_charts.items():
                if not _is_exact_rational_matrix(chart):
                    raise TypeError(
                        "theorem certificates require exact rational SymPy charts"
                    )
                exact_charts[label] = sp.ImmutableMatrix(chart)
            frozen[int(grade)] = MappingProxyType(exact_charts)
        object.__setattr__(self, "charts", MappingProxyType(frozen))

    def coordinate_edge(self, edge_name: str, grade: int) -> Matrix:
        edge = self.graph.word_edges((edge_name,))[0]
        try:
            source_chart = self.charts[grade][edge.source]
            target_chart = self.charts[grade][edge.target]
        except KeyError as error:
            raise ValueError(
                f"missing grade-{grade} chart for object {error.args[0]!r}"
            ) from error
        exterior_edge = compound_matrix(edge.matrix, grade)
        return _inverse(target_chart) @ exterior_edge @ source_chart

    def is_entrywise_nonnegative(self, *, tolerance: float = 1e-10) -> bool:
        expected_grades = set(range(self.graph.dimension + 1))
        if set(self.charts) != expected_grades:
            return False
        for grade in expected_grades:
            if set(self.charts[grade]) != set(self.graph.objects):
                return False
            for edge in self.graph.edges:
                if not _matrix_is_nonnegative(
                    self.coordinate_edge(edge.name, grade),
                    tolerance=tolerance,
                ):
                    return False
        return True

    def certify_closed_word(
        self,
        word: Sequence[str],
        *,
        tolerance: float = 1e-10,
    ) -> ClosedWordCertificate:
        if not self.graph.is_closed_word(word):
            raise ValueError("positivity certificate requires a legal closed word")
        if not self.is_entrywise_nonnegative(tolerance=tolerance):
            raise ValueError("grade charts do not make every typed edge nonnegative")
        product = self.graph.word_product(word, require_legal=True)
        traces = tuple(
            _trace(compound_matrix(product, grade))
            for grade in range(self.graph.dimension + 1)
        )
        weight = exterior_character_sum(product)
        direct_weight = sp.det(sp.eye(self.graph.dimension) + product)
        if sp.simplify(weight - direct_weight) != 0:
            raise RuntimeError("exterior character does not match direct determinant")
        if any(
            _entry_sign(trace, tolerance=0.0) not in {0, 1}
            for trace in traces
        ):
            raise RuntimeError("certified closed word has a negative grade trace")
        if _entry_sign(direct_weight, tolerance=0.0) not in {0, 1}:
            raise RuntimeError("certified closed word has a negative determinant")
        return ClosedWordCertificate(
            tuple(word),
            traces,
            weight,
            direct_weight,
        )


def certificate_from_one_particle_charts(
    graph: TypedExteriorGraph,
    one_particle_charts: Mapping[ObjectLabel, Matrix],
    *,
    tolerance: float = 1e-10,
) -> GradeChartCertificate:
    """Lift object charts functorially into every exterior grade."""

    _require_exact_rational_graph(graph)
    if set(one_particle_charts) != set(graph.objects):
        raise ValueError("one-particle charts must cover exactly the graph objects")
    if not all(
        _is_exact_rational_matrix(chart)
        for chart in one_particle_charts.values()
    ):
        raise TypeError(
            "theorem certificates require exact rational SymPy charts"
        )
    charts: dict[int, dict[ObjectLabel, Matrix]] = {}
    for grade in range(graph.dimension + 1):
        charts[grade] = {
            label: compound_matrix(chart, grade)
            for label, chart in one_particle_charts.items()
        }
    certificate = GradeChartCertificate(
        graph=graph,
        charts=charts,
        proof_origin="one_particle_charts",
        has_induced_diagonal_sign_certificate=has_induced_diagonal_charts(
            graph,
            tolerance=tolerance,
        ),
    )
    if not certificate.is_entrywise_nonnegative(tolerance=tolerance):
        raise ValueError("one-particle charts do not certify every typed edge")
    return certificate


def _diagonal(signs: Sequence[int], *, exact: bool) -> Matrix:
    if exact:
        return sp.ImmutableMatrix(sp.diag(*signs))
    return np.diag(np.asarray(signs, dtype=float))


def _gf2_system_is_consistent(
    equations: Iterable[tuple[int, int]],
) -> bool:
    """Gaussian-eliminate bit-packed linear equations over GF(2)."""

    pivots: dict[int, tuple[int, int]] = {}
    for raw_mask, raw_rhs in equations:
        mask = int(raw_mask)
        rhs = int(raw_rhs) & 1
        while mask:
            pivot = (mask & -mask).bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = (mask, rhs)
                break
            pivot_mask, pivot_rhs = pivots[pivot]
            mask ^= pivot_mask
            rhs ^= pivot_rhs
        if mask == 0 and rhs:
            return False
    return True


def has_induced_diagonal_charts(
    graph: TypedExteriorGraph,
    *,
    tolerance: float = 1e-10,
) -> bool:
    """Return whether one-particle sign gauges induce every grade chart.

    Unlike comparing one arbitrary solution of the per-grade systems, this
    solves the coupled GF(2) feasibility problem directly.  The variables are
    the one-particle signs ``x_(object,mode)``; a grade-``k`` minor uses the
    XOR sum of the ``k`` corresponding variables.
    """

    _require_exact_rational_graph(graph)
    variable_index = {
        (label, mode): index
        for index, (label, mode) in enumerate(
            (label, mode)
            for label in graph.objects
            for mode in range(graph.dimension)
        )
    }
    equations: list[tuple[int, int]] = []
    for grade in range(graph.dimension + 1):
        basis = subset_basis(graph.dimension, grade)
        for edge in graph.edges:
            exterior_edge = compound_matrix(edge.matrix, grade)
            for row, target_subset in enumerate(basis):
                for column, source_subset in enumerate(basis):
                    sign = _entry_sign(
                        exterior_edge[row, column],
                        tolerance=tolerance,
                    )
                    if sign is None:
                        return False
                    if sign == 0:
                        continue
                    mask = 0
                    for mode in target_subset:
                        mask ^= 1 << variable_index[(edge.target, mode)]
                    for mode in source_subset:
                        mask ^= 1 << variable_index[(edge.source, mode)]
                    equations.append((mask, 0 if sign > 0 else 1))
    return _gf2_system_is_consistent(equations)


def solve_diagonal_grade_charts(
    graph: TypedExteriorGraph,
    *,
    tolerance: float = 1e-10,
) -> GradeChartCertificate | None:
    """Solve all compound-entry sign constraints by parity propagation.

    A variable is attached to every ``(object, subset)`` coordinate.  Every
    nonzero entry of every typed exterior edge contributes one GF(2)
    equation.  Zero-support entries impose no constraint, which is precisely
    the support-aware freedom explored by the pilot.
    """

    _require_exact_rational_graph(graph)
    charts: dict[int, dict[ObjectLabel, Matrix]] = {}
    for grade in range(graph.dimension + 1):
        basis = subset_basis(graph.dimension, grade)
        coordinate_count = len(basis)
        variables = [
            (label, index)
            for label in graph.objects
            for index in range(coordinate_count)
        ]
        adjacency: dict[
            tuple[ObjectLabel, int],
            list[tuple[tuple[ObjectLabel, int], int]],
        ] = {variable: [] for variable in variables}

        for edge in graph.edges:
            exterior_edge = compound_matrix(edge.matrix, grade)
            for row in range(coordinate_count):
                for column in range(coordinate_count):
                    sign = _entry_sign(
                        exterior_edge[row, column],
                        tolerance=tolerance,
                    )
                    if sign is None:
                        return None
                    if sign == 0:
                        continue
                    left = (edge.target, row)
                    right = (edge.source, column)
                    parity = 0 if sign > 0 else 1
                    adjacency[left].append((right, parity))
                    adjacency[right].append((left, parity))

        assignment: dict[tuple[ObjectLabel, int], int] = {}
        for root in variables:
            if root in assignment:
                continue
            assignment[root] = 0
            stack = [root]
            while stack:
                current = stack.pop()
                for neighbor, parity in adjacency[current]:
                    required = assignment[current] ^ parity
                    if neighbor in assignment:
                        if assignment[neighbor] != required:
                            return None
                    else:
                        assignment[neighbor] = required
                        stack.append(neighbor)

        charts[grade] = {
            label: _diagonal(
                [
                    -1 if assignment[(label, index)] else 1
                    for index in range(coordinate_count)
                ],
                exact=True,
            )
            for label in graph.objects
        }

    induced = has_induced_diagonal_charts(graph, tolerance=tolerance)

    certificate = GradeChartCertificate(
        graph=graph,
        charts=charts,
        proof_origin="grade_diagonal_sign_charts",
        has_induced_diagonal_sign_certificate=induced,
    )
    if not certificate.is_entrywise_nonnegative(tolerance=tolerance):
        raise RuntimeError("internal error: solved grade charts failed verification")
    return certificate
