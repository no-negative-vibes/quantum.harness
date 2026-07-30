"""Exact front-door filters for known typed positivity reductions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from types import MappingProxyType
from typing import Hashable, Mapping, Sequence

import sympy as sp

from .exterior_category_search import (
    TypedEdge,
    TypedExteriorGraph,
    has_induced_diagonal_charts,
)


@dataclass(frozen=True)
class SignedPermutationTNWitness:
    """Object-wise mode reorderings that reduce the graph to gauge-TN."""

    object_permutations: Mapping[Hashable, tuple[int, ...]]
    tested_assignments: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "object_permutations",
            MappingProxyType(dict(self.object_permutations)),
        )


def _permutation_matrix(
    permutation: Sequence[int],
) -> sp.ImmutableMatrix:
    dimension = len(permutation)
    if tuple(sorted(permutation)) != tuple(range(dimension)):
        raise ValueError("permutation must contain every mode exactly once")
    matrix = sp.zeros(dimension)
    for column, row in enumerate(permutation):
        matrix[int(row), column] = 1
    return sp.ImmutableMatrix(matrix)


def apply_object_permutations(
    graph: TypedExteriorGraph,
    object_permutations: Mapping[Hashable, Sequence[int]],
) -> TypedExteriorGraph:
    """Express each edge after independent source/target mode reorderings."""

    if set(object_permutations) != set(graph.objects):
        raise ValueError("permutations must cover exactly the graph objects")
    matrices = {
        label: _permutation_matrix(object_permutations[label])
        for label in graph.objects
    }
    return TypedExteriorGraph(
        TypedEdge(
            edge.name,
            edge.source,
            edge.target,
            matrices[edge.target].T * edge.matrix * matrices[edge.source],
        )
        for edge in graph.edges
    )


def find_signed_permutation_tn_coboundary(
    graph: TypedExteriorGraph,
    *,
    max_assignments: int | None = None,
) -> SignedPermutationTNWitness | None:
    """Exhaust object-wise permutations followed by diagonal sign gauges.

    A hit gives explicit one-particle charts of the form ``P_a D_a``.  Their
    exterior lifts make every typed edge totally nonnegative, so the candidate
    is a known state-gauge/coboundary reduction rather than a genuinely
    grade-specific mechanism.

    ``max_assignments`` is an operational cap only.  Returning ``None`` after
    a capped search is not a non-inclusion proof and must be reported as
    incomplete by the caller.
    """

    if max_assignments is not None and max_assignments < 1:
        raise ValueError("max_assignments must be positive")
    mode_permutations = tuple(permutations(range(graph.dimension)))
    tested = 0
    for assignment in product(
        mode_permutations,
        repeat=len(graph.objects),
    ):
        if max_assignments is not None and tested >= max_assignments:
            return None
        tested += 1
        object_permutations = dict(zip(graph.objects, assignment))
        transformed = apply_object_permutations(
            graph,
            object_permutations,
        )
        if has_induced_diagonal_charts(transformed):
            return SignedPermutationTNWitness(
                object_permutations=object_permutations,
                tested_assignments=tested,
            )
    return None
