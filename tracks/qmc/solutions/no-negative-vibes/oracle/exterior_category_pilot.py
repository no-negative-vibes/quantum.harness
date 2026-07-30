"""Deterministic pilot grammar and resumable runner for typed exterior search.

The pilot deliberately separates an operationally successful calculation from
its scientific outcome.  An exact negative type-erasure witness is therefore a
successful cell result, while a corrupt or mismatched manifest is recomputed.
All words use the repository's chronological convention:
``(e1, e2) -> B_e2 @ B_e1``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from .compound_cones import compound_matrix, subset_basis
from .exterior_category_search import (
    TypedEdge,
    TypedExteriorGraph,
    solve_diagonal_grade_charts,
)
from .known_class_filters import find_signed_permutation_tn_coboundary


_GRAMMARS = frozenset(
    {
        "coboundary_tn_control",
        "support_aware_sparse",
    }
)
_MANIFEST_SCHEMA = "exterior-category-pilot-cell-v1"


@dataclass(frozen=True)
class PilotCandidate:
    """One floating candidate together with its exact replay representation."""

    grammar: str
    seed: int
    candidate_id: str
    graph: TypedExteriorGraph
    exact_graph: TypedExteriorGraph
    definition: Mapping[str, Any]


@dataclass(frozen=True)
class TypeErasureSearch:
    """Accounting and exact witnesses from an untyped free-word search."""

    determinant_checks: int
    exact_replays: int
    ambiguous_checks: int
    witnesses: tuple[dict[str, Any], ...]


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _matrix_payload(matrix: sp.MatrixBase) -> list[list[int | str]]:
    def scalar(value: sp.Basic) -> int | str:
        simplified = sp.simplify(value)
        if simplified.is_Integer:
            return int(simplified)
        return str(simplified)

    return [
        [scalar(matrix[row, column]) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def _float_graph(exact_graph: TypedExteriorGraph) -> TypedExteriorGraph:
    return TypedExteriorGraph(
        TypedEdge(
            edge.name,
            edge.source,
            edge.target,
            np.asarray(edge.matrix, dtype=float),
        )
        for edge in exact_graph.edges
    )


def _inversion_parity(permutation: Sequence[int]) -> int:
    return sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    ) % 2


def _even_permutation(
    rng: random.Random,
    dimension: int,
) -> tuple[list[int], sp.ImmutableMatrix]:
    permutation = list(range(dimension))
    if dimension >= 3 and bool(rng.getrandbits(1)):
        first, second, third = rng.sample(range(dimension), 3)
        permutation[first] = second
        permutation[second] = third
        permutation[third] = first
    matrix = sp.zeros(dimension)
    for column, row in enumerate(permutation):
        matrix[row, column] = 1
    return permutation, sp.ImmutableMatrix(matrix)


def _shear(
    dimension: int,
    row: int,
    column: int,
    coefficient: int,
) -> sp.ImmutableMatrix:
    matrix = sp.eye(dimension)
    matrix[row, column] = coefficient
    return sp.ImmutableMatrix(matrix)


def _chronological_exact(factors: Sequence[sp.MatrixBase]) -> sp.ImmutableMatrix:
    if not factors:
        raise ValueError("at least one factor is required")
    dimension = factors[0].rows
    result: sp.MatrixBase = sp.eye(dimension)
    for factor in factors:
        result = factor * result
    return sp.ImmutableMatrix(result)


def _tn_coordinate(
    rng: random.Random,
    dimension: int,
) -> tuple[sp.ImmutableMatrix, dict[str, Any]]:
    """Build a product of positive Jacobi shears and a positive diagonal."""

    shear_specs: list[dict[str, int]] = []
    shear_factors: list[sp.ImmutableMatrix] = []
    for _ in range(rng.randint(1, 3)):
        lower = bool(rng.getrandbits(1))
        index = rng.randrange(dimension - 1)
        row, column = (
            (index + 1, index) if lower else (index, index + 1)
        )
        coefficient = rng.randint(1, 3)
        shear_specs.append(
            {
                "row": row,
                "column": column,
                "coefficient": coefficient,
            }
        )
        shear_factors.append(
            _shear(dimension, row, column, coefficient)
        )
    diagonal_values = [rng.randint(1, 3) for _ in range(dimension)]
    diagonal = sp.ImmutableMatrix(sp.diag(*diagonal_values))
    matrix = _chronological_exact((*shear_factors, diagonal))
    return matrix, {
        "shears": shear_specs,
        "positive_diagonal": diagonal_values,
        "coordinate_matrix": _matrix_payload(matrix),
    }


def _coboundary_tn_control(
    dimension: int,
    seed: int,
) -> PilotCandidate:
    rng = random.Random(seed)
    signs = [rng.choice((-1, 1)) for _ in range(dimension)]
    if all(sign > 0 for sign in signs):
        signs[rng.randrange(dimension)] = -1
    chart_b = sp.ImmutableMatrix(sp.diag(*signs))

    coordinate_1, definition_1 = _tn_coordinate(rng, dimension)
    coordinate_2, definition_2 = _tn_coordinate(rng, dimension)
    coordinate_return, definition_return = _tn_coordinate(rng, dimension)
    exact_edges = (
        TypedEdge(
            "forward-1",
            "a",
            "b",
            chart_b * coordinate_1,
        ),
        TypedEdge(
            "forward-2",
            "a",
            "b",
            chart_b * coordinate_2,
        ),
        TypedEdge(
            "return",
            "b",
            "a",
            coordinate_return * chart_b,
        ),
    )
    edge_definitions: list[dict[str, Any]] = []
    for edge, coordinate_definition in zip(
        exact_edges,
        (definition_1, definition_2, definition_return),
    ):
        edge_definitions.append(
            {
                "name": edge.name,
                "source": edge.source,
                "target": edge.target,
                **coordinate_definition,
                "matrix": _matrix_payload(sp.Matrix(edge.matrix)),
            }
        )
    definition = {
        "grammar": "coboundary_tn_control",
        "dimension": dimension,
        "seed": seed,
        "object_chart_signs": {"a": [1] * dimension, "b": signs},
        "edges": edge_definitions,
    }
    exact_graph = TypedExteriorGraph(exact_edges)
    return PilotCandidate(
        grammar="coboundary_tn_control",
        seed=seed,
        candidate_id=_digest(definition),
        graph=_float_graph(exact_graph),
        exact_graph=exact_graph,
        definition=definition,
    )


def _support_aware_edge(
    rng: random.Random,
    *,
    name: str,
    source: str,
    target: str,
    dimension: int,
) -> tuple[TypedEdge, dict[str, Any]]:
    permutation, permutation_matrix = _even_permutation(rng, dimension)
    shear_specs: list[dict[str, int]] = []
    shear_factors: list[sp.ImmutableMatrix] = []
    for _ in range(rng.randint(1, 3)):
        row = rng.randrange(dimension)
        column = rng.randrange(dimension - 1)
        if column >= row:
            column += 1
        coefficient = rng.choice((-2, -1, 1, 2))
        shear_specs.append(
            {
                "row": row,
                "column": column,
                "coefficient": coefficient,
            }
        )
        shear_factors.append(
            _shear(dimension, row, column, coefficient)
        )
    diagonal_values = [1] * dimension
    diagonal_values[rng.randrange(dimension)] = rng.choice((1, 2))
    positive_diagonal = sp.ImmutableMatrix(sp.diag(*diagonal_values))
    exact_matrix = _chronological_exact(
        (permutation_matrix, positive_diagonal, *shear_factors)
    )
    edge = TypedEdge(name, source, target, exact_matrix)
    definition = {
        "name": name,
        "source": source,
        "target": target,
        "factor_order": [
            "even_permutation",
            "positive_diagonal",
            "integer_shears",
        ],
        "even_permutation": permutation,
        "shears": shear_specs,
        "positive_diagonal": diagonal_values,
        "matrix": _matrix_payload(exact_matrix),
    }
    return edge, definition


def _support_aware_sparse(
    dimension: int,
    seed: int,
) -> PilotCandidate:
    rng = random.Random(seed)
    declarations = (
        ("forward-1", "a", "b"),
        ("forward-2", "a", "b"),
        ("return", "b", "a"),
    )
    generated = tuple(
        _support_aware_edge(
            rng,
            name=name,
            source=source,
            target=target,
            dimension=dimension,
        )
        for name, source, target in declarations
    )
    exact_graph = TypedExteriorGraph(item[0] for item in generated)
    definition = {
        "grammar": "support_aware_sparse",
        "dimension": dimension,
        "seed": seed,
        "edges": [item[1] for item in generated],
    }
    return PilotCandidate(
        grammar="support_aware_sparse",
        seed=seed,
        candidate_id=_digest(definition),
        graph=_float_graph(exact_graph),
        exact_graph=exact_graph,
        definition=definition,
    )


def generate_candidate(
    grammar: str,
    *,
    dimension: int,
    seed: int,
) -> PilotCandidate:
    """Generate one deterministic candidate from a declared pilot grammar."""

    if grammar not in _GRAMMARS:
        raise ValueError(f"unknown pilot grammar: {grammar!r}")
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError("dimension must be an integer")
    if dimension < 2:
        raise ValueError("dimension must be at least two")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if grammar == "coboundary_tn_control":
        return _coboundary_tn_control(dimension, seed)
    return _support_aware_sparse(dimension, seed)


def determinant_weight(
    graph: TypedExteriorGraph,
    word: Sequence[str],
) -> object:
    """Return ``det(I + B_last ... B_first)`` for one declared word."""

    matrix = graph.word_product(word)
    if isinstance(matrix, sp.MatrixBase):
        return sp.factor((sp.eye(matrix.rows) + matrix).det())
    array = np.asarray(matrix)
    return np.linalg.det(
        np.eye(array.shape[0], dtype=array.dtype) + array
    )


def _exact_sign(value: sp.Basic) -> int | None:
    simplified = sp.simplify(value)
    if simplified == 0:
        return 0
    if simplified.is_negative:
        return -1
    if simplified.is_positive:
        return 1
    return None


def _float_real(value: object, *, tolerance: float) -> float | None:
    numerical = complex(value)
    if not math.isfinite(numerical.real) or not math.isfinite(numerical.imag):
        return None
    if abs(numerical.imag) > tolerance:
        return None
    return float(numerical.real)


def find_type_erasure_witnesses(
    graph: TypedExteriorGraph,
    exact_graph: TypedExteriorGraph,
    *,
    max_depth: int,
    tolerance: float,
    max_witnesses: int,
) -> TypeErasureSearch:
    """Search illegal free words and retain only exact negative witnesses."""

    if max_depth < 2:
        raise ValueError("max_depth must be at least two")
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    if max_witnesses < 1:
        raise ValueError("max_witnesses must be positive")
    graph_signature = tuple(
        (edge.name, edge.source, edge.target)
        for edge in graph.edges
    )
    exact_signature = tuple(
        (edge.name, edge.source, edge.target)
        for edge in exact_graph.edges
    )
    if graph_signature != exact_signature:
        raise ValueError("floating and exact graphs must have identical edges")

    edge_names = tuple(edge.name for edge in graph.edges)
    determinant_checks = 0
    exact_replays = 0
    ambiguous_checks = 0
    witnesses: list[dict[str, Any]] = []
    for depth in range(1, max_depth + 1):
        for word in product(edge_names, repeat=depth):
            if graph.is_closed_word(word):
                continue
            determinant_checks += 1
            floating_weight = determinant_weight(graph, word)
            real_weight = _float_real(
                floating_weight,
                tolerance=tolerance,
            )
            needs_replay = (
                real_weight is None
                or real_weight < -tolerance
                or abs(real_weight) <= tolerance
            )
            if not needs_replay:
                continue

            exact_replays += 1
            exact_weight = sp.simplify(
                determinant_weight(exact_graph, word)
            )
            sign = _exact_sign(exact_weight)
            if sign == -1:
                witnesses.append(
                    {
                        "word": list(word),
                        "float_weight": (
                            None if real_weight is None else real_weight
                        ),
                        "exact_weight": str(exact_weight),
                        "depth": depth,
                        (
                            "primitive_real_log_impossible_by_negative_"
                            "singleton"
                        ): depth == 1,
                        "product_order": "B_last ... B_first",
                    }
                )
                if len(witnesses) >= max_witnesses:
                    return TypeErasureSearch(
                        determinant_checks=determinant_checks,
                        exact_replays=exact_replays,
                        ambiguous_checks=ambiguous_checks,
                        witnesses=tuple(witnesses),
                    )
            elif real_weight is None or real_weight <= tolerance:
                ambiguous_checks += 1

    return TypeErasureSearch(
        determinant_checks=determinant_checks,
        exact_replays=exact_replays,
        ambiguous_checks=ambiguous_checks,
        witnesses=tuple(witnesses),
    )


def _candidate_seed(
    *,
    grammar: str,
    dimension: int,
    base_seed: int,
    index: int,
) -> int:
    payload = f"{grammar}|{dimension}|{base_seed}|{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _positive_int(
    settings: Mapping[str, Any],
    field: str,
    default: int,
) -> int:
    value = settings.get(field, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def float_diagonal_grade_screen(
    graph: TypedExteriorGraph,
    *,
    tolerance: float,
) -> bool:
    """Cheap GF(2) sign-consistency screen with no certificate semantics.

    Tolerance can erase a small negative entry, so a ``True`` result is only a
    reason to rebuild the integer definition and call the exact solver.
    """

    if any(isinstance(edge.matrix, sp.MatrixBase) for edge in graph.edges):
        raise TypeError("the floating screen requires NumPy edge matrices")
    passed = True
    for grade in range(graph.dimension + 1):
        coordinate_count = len(subset_basis(graph.dimension, grade))
        variables = [
            (label, index)
            for label in graph.objects
            for index in range(coordinate_count)
        ]
        adjacency: dict[
            tuple[Any, int],
            list[tuple[tuple[Any, int], int]],
        ] = {variable: [] for variable in variables}
        for edge in graph.edges:
            exterior_edge = np.asarray(
                compound_matrix(edge.matrix, grade)
            )
            for row in range(coordinate_count):
                for column in range(coordinate_count):
                    numerical = complex(exterior_edge[row, column])
                    if (
                        not math.isfinite(numerical.real)
                        or not math.isfinite(numerical.imag)
                        or abs(numerical.imag) > tolerance
                    ):
                        passed = False
                        continue
                    if abs(numerical.real) <= tolerance:
                        continue
                    left = (edge.target, row)
                    right = (edge.source, column)
                    parity = 0 if numerical.real > 0 else 1
                    adjacency[left].append((right, parity))
                    adjacency[right].append((left, parity))

        assignment: dict[tuple[Any, int], int] = {}
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
                            passed = False
                    else:
                        assignment[neighbor] = required
                        stack.append(neighbor)
    return passed


def _legal_closed_witness(
    certificate: Any,
    *,
    max_depth: int,
) -> dict[str, Any] | None:
    """Return the first nontrivial exact closed path certified by the charts."""

    graph = certificate.graph
    edge_names = tuple(edge.name for edge in graph.edges)
    for depth in range(2, max_depth + 1):
        for word in product(edge_names, repeat=depth):
            if not graph.is_closed_word(word):
                continue
            closed = certificate.certify_closed_word(word, tolerance=0.0)
            return {
                "word": list(word),
                "exact_weight": str(sp.simplify(closed.weight)),
                "grade_traces": [
                    str(sp.simplify(value))
                    for value in closed.grade_traces
                ],
            }
    return None


def _legal_stress(
    certificate: Any,
    *,
    depths: Sequence[int],
    words_per_depth: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Replay several deterministic legal schedules at longer word depths."""

    graph = certificate.graph
    edge_names = tuple(edge.name for edge in graph.edges)
    cycles: list[tuple[str, ...]] = []
    for word in product(edge_names, repeat=2):
        if graph.is_closed_word(word):
            cycles.append(tuple(word))
    if not cycles:
        return []
    cycles_by_start: dict[Any, list[tuple[str, ...]]] = {}
    for cycle in cycles:
        start = graph.word_edges(cycle)[0].source
        cycles_by_start.setdefault(start, []).append(cycle)
    compatible_cycles = max(
        cycles_by_start.values(),
        key=lambda group: (len(group), str(group[0])),
    )
    first = compatible_cycles[0]
    second = compatible_cycles[1] if len(compatible_cycles) > 1 else first
    rng = random.Random(seed)

    results: list[dict[str, Any]] = []
    for depth in depths:
        if depth % len(first):
            raise ValueError(
                "legal stress depth must be divisible by the shortest "
                "two-edge cycle"
            )
        cycle_count = depth // len(first)
        half = cycle_count // 2
        palindromic_choices = (
            [first] * half
            + ([second] if cycle_count % 2 else [])
            + [first] * half
        )
        strategies: list[tuple[str, tuple[str, ...]]] = [
            ("alternating_first", first * cycle_count),
            ("alternating_second", second * cycle_count),
            (
                "palindromic_mixed",
                tuple(
                    edge
                    for cycle in palindromic_choices
                    for edge in cycle
                ),
            ),
            (
                "seeded_random",
                tuple(
                    edge
                    for _ in range(cycle_count)
                    for edge in rng.choice(compatible_cycles)
                ),
            ),
        ]
        for strategy, word in strategies[:words_per_depth]:
            closed = certificate.certify_closed_word(word, tolerance=0.0)
            results.append(
                {
                    "depth": depth,
                    "strategy": strategy,
                    "word": list(word),
                    "exact_weight": str(sp.simplify(closed.direct_weight)),
                }
            )
    return results


def scan_cell(
    *,
    params: Mapping[str, Any],
    settings: Mapping[str, Any],
    provenance: Mapping[str, Any],
    spec_digest: str,
) -> dict[str, Any]:
    """Run the diagonal-chart and type-erasure funnel for one parameter cell."""

    grammar = str(params["grammar"])
    dimension = int(params["dimension"])
    base_seed = int(params["seed"])
    candidates_per_cell = _positive_int(
        settings,
        "candidates_per_cell",
        1,
    )
    max_word_depth = _positive_int(settings, "max_word_depth", 2)
    if max_word_depth < 2:
        raise ValueError("max_word_depth must be at least two")
    max_witnesses = _positive_int(
        settings,
        "max_exact_witnesses_per_candidate",
        1,
    )
    tolerance = float(settings.get("weight_tolerance", 1e-10))
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("weight_tolerance must be finite and nonnegative")
    chart_tolerance = float(settings.get("chart_tolerance", tolerance))
    if chart_tolerance < 0 or not math.isfinite(chart_tolerance):
        raise ValueError("chart_tolerance must be finite and nonnegative")
    signed_permutation_max_assignments = _positive_int(
        settings,
        "signed_permutation_max_assignments",
        1_000,
    )
    legal_stress_depths = tuple(
        int(depth) for depth in settings.get("legal_stress_depths", ())
    )
    if any(depth < 2 or depth % 2 for depth in legal_stress_depths):
        raise ValueError("legal_stress_depths must contain positive even depths")
    legal_stress_candidates = _positive_int(
        settings,
        "legal_stress_candidates_per_cell",
        1,
    )
    legal_stress_words_per_depth = _positive_int(
        settings,
        "legal_stress_words_per_depth",
        4,
    )
    if legal_stress_words_per_depth > 4:
        raise ValueError("legal_stress_words_per_depth cannot exceed four")

    counts = {
        "candidates": candidates_per_cell,
        "certificate_stage_checks": 0,
        "edge_grade_checks": (
            candidates_per_cell * 3 * (dimension + 1)
        ),
        "determinant_checks": 0,
        "exact_replays": 0,
        "exact_witnesses": 0,
        "ambiguous_checks": 0,
        "no_chart": 0,
        "typed_only_calibration": 0,
        "signed_permutation_tn_reduction": 0,
        "provisional_front_door_survivor": 0,
        "known_class_filter_incomplete": 0,
        "untyped_not_separated": 0,
        "ambiguous": 0,
        "physical_strong_candidate": 0,
        "legal_stress_word_checks": 0,
    }
    candidate_records: list[dict[str, Any]] = []
    exact_witnesses: list[dict[str, Any]] = []
    stressed_candidates = 0
    for candidate_index in range(candidates_per_cell):
        seed = _candidate_seed(
            grammar=grammar,
            dimension=dimension,
            base_seed=base_seed,
            index=candidate_index,
        )
        candidate = generate_candidate(
            grammar,
            dimension=dimension,
            seed=seed,
        )
        counts["certificate_stage_checks"] += 1
        float_screen_passed = float_diagonal_grade_screen(
            candidate.graph,
            tolerance=chart_tolerance,
        )
        record: dict[str, Any] = {
            "candidate_index": candidate_index,
            "candidate_seed": seed,
            "candidate_id": candidate.candidate_id,
            "grammar": candidate.grammar,
            "definition": candidate.definition,
            "float_chart_screen_passed": float_screen_passed,
            "has_exact_grade_chart": False,
            "determinant_checks": 0,
            "exact_replays": 0,
            "ambiguous_checks": 0,
            "exact_witnesses": [],
            "minimum_exact_negative_word_depth": None,
            "primitive_real_log_audit": {
                "all_edges_pass": False,
                "status": "not_run",
                "reason": (
                    "a product-of-elementary-factors is not a single real "
                    "logarithm"
                ),
            },
            "physical_strong_candidate": False,
            "signed_permutation_filter_complete": None,
            "signed_permutation_tn_witness": None,
            "novelty_status": "not_reached",
            "legal_stress": [],
        }
        if not float_screen_passed:
            classification = "no_chart"
            record["chart_screen"] = "float_fail"
            counts[classification] += 1
        else:
            counts["certificate_stage_checks"] += 1
            exact_certificate = solve_diagonal_grade_charts(
                candidate.exact_graph,
                tolerance=0.0,
            )
            if exact_certificate is None:
                classification = "no_chart"
                record["chart_screen"] = "float_pass_exact_fail"
                counts[classification] += 1
            else:
                record["has_exact_grade_chart"] = True
                legal_closed_witness = _legal_closed_witness(
                    exact_certificate,
                    max_depth=max_word_depth,
                )
                record.update(
                    {
                        "chart_screen": "exact_pass",
                        # This says only that the independently solved
                        # grade-sign charts are the exterior lifts of one
                        # diagonal sign chart.  It is not a claim about a
                        # general one-particle coboundary reduction.
                        "induced_by_diagonal_sign_charts": (
                            exact_certificate
                            .has_induced_diagonal_sign_certificate
                        ),
                        "legal_closed_witness": legal_closed_witness,
                    }
                )
                if legal_closed_witness is None:
                    classification = "untyped_not_separated"
                    counts[classification] += 1
                else:
                    if (
                        legal_stress_depths
                        and stressed_candidates < legal_stress_candidates
                    ):
                        record["legal_stress"] = _legal_stress(
                            exact_certificate,
                            depths=legal_stress_depths,
                            words_per_depth=legal_stress_words_per_depth,
                            seed=seed,
                        )
                        counts["legal_stress_word_checks"] += len(
                            record["legal_stress"]
                        )
                        stressed_candidates += 1
                    search = find_type_erasure_witnesses(
                        candidate.graph,
                        candidate.exact_graph,
                        max_depth=max_word_depth,
                        tolerance=tolerance,
                        max_witnesses=max_witnesses,
                    )
                    record.update(
                        {
                            "determinant_checks": search.determinant_checks,
                            "exact_replays": search.exact_replays,
                            "ambiguous_checks": search.ambiguous_checks,
                            "exact_witnesses": list(search.witnesses),
                            "minimum_exact_negative_word_depth": (
                                min(
                                    witness["depth"]
                                    for witness in search.witnesses
                                )
                                if search.witnesses
                                else None
                            ),
                        }
                    )
                    counts["determinant_checks"] += (
                        search.determinant_checks
                    )
                    counts["exact_replays"] += search.exact_replays
                    counts["ambiguous_checks"] += search.ambiguous_checks
                    for witness in search.witnesses:
                        exact_witnesses.append(
                            {
                                "candidate_id": candidate.candidate_id,
                                "candidate_seed": seed,
                                **witness,
                            }
                        )

                    if not search.witnesses:
                        if search.ambiguous_checks:
                            classification = "ambiguous"
                            record["novelty_status"] = (
                                "ambiguous_type_erasure_weight"
                            )
                        else:
                            classification = "untyped_not_separated"
                            record["novelty_status"] = (
                                "no_exact_type_erasure_witness_within_depth"
                            )
                    elif grammar == "coboundary_tn_control":
                        classification = "typed_only_calibration"
                        record["novelty_status"] = (
                            "known_one_particle_coboundary_control"
                        )
                    elif exact_certificate.has_induced_diagonal_sign_certificate:
                        classification = "typed_only_calibration"
                        record["novelty_status"] = (
                            "known_diagonal_sign_tn_coboundary"
                        )
                    else:
                        total_assignments = math.factorial(dimension) ** len(
                            candidate.exact_graph.objects
                        )
                        filter_complete = (
                            signed_permutation_max_assignments
                            >= total_assignments
                        )
                        signed_permutation_witness = (
                            find_signed_permutation_tn_coboundary(
                                candidate.exact_graph,
                                max_assignments=min(
                                    signed_permutation_max_assignments,
                                    total_assignments,
                                ),
                            )
                        )
                        record["signed_permutation_filter_complete"] = (
                            filter_complete
                            or signed_permutation_witness is not None
                        )
                        if signed_permutation_witness is not None:
                            record["signed_permutation_tn_witness"] = {
                                "object_permutations": {
                                    str(label): list(permutation)
                                    for label, permutation in (
                                        signed_permutation_witness
                                        .object_permutations.items()
                                    )
                                },
                                "tested_assignments": (
                                    signed_permutation_witness
                                    .tested_assignments
                                ),
                            }
                            record["novelty_status"] = (
                                "known_signed_permutation_tn_coboundary"
                            )
                            classification = (
                                "signed_permutation_tn_reduction"
                            )
                        elif not filter_complete:
                            record["novelty_status"] = (
                                "signed_permutation_filter_incomplete"
                            )
                            classification = "known_class_filter_incomplete"
                        else:
                            record["novelty_status"] = (
                                "provisional_after_front_door_filters"
                            )
                            classification = (
                                "provisional_front_door_survivor"
                            )
                    if (
                        classification == "provisional_front_door_survivor"
                        and not record["legal_stress"]
                        and legal_stress_depths
                    ):
                        record["legal_stress"] = _legal_stress(
                            exact_certificate,
                            depths=legal_stress_depths,
                            words_per_depth=legal_stress_words_per_depth,
                            seed=seed,
                        )
                        counts["legal_stress_word_checks"] += len(
                            record["legal_stress"]
                        )
                    counts[classification] += 1

        record["classification"] = classification
        candidate_records.append(record)

    counts["exact_witnesses"] = len(exact_witnesses)
    return {
        "schema_version": 1,
        "manifest_schema": _MANIFEST_SCHEMA,
        "completed": True,
        "compute_success": True,
        "params": dict(params),
        "settings": dict(settings),
        "provenance": dict(provenance),
        "spec_digest": spec_digest,
        "counts": counts,
        "candidate_records": candidate_records,
        "exact_witnesses": exact_witnesses,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _matching_success_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    cell_id: str,
    params: Mapping[str, Any],
    settings: Mapping[str, Any],
    provenance: Mapping[str, Any],
    spec_digest: str,
) -> bool:
    return bool(
        manifest is not None
        and manifest.get("completed") is True
        and manifest.get("compute_success") is True
        and manifest.get("manifest_schema") == _MANIFEST_SCHEMA
        and manifest.get("schema_version") == 1
        and manifest.get("cell_id") == cell_id
        and manifest.get("params") == dict(params)
        and manifest.get("settings") == dict(settings)
        and manifest.get("provenance") == dict(provenance)
        and manifest.get("spec_digest") == spec_digest
        and isinstance(manifest.get("counts"), dict)
        and isinstance(manifest.get("candidate_records"), list)
        and isinstance(manifest.get("exact_witnesses"), list)
    )


def _derived_spec_digest(spec: Mapping[str, Any]) -> str:
    frozen = {
        key: value
        for key, value in spec.items()
        if key not in {"run_dir", "spec_digest"}
    }
    return _digest(frozen)


def run_spec(
    path: str | Path,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> dict[str, int]:
    """Run or resume one deterministic modulo shard of a parameter scan."""

    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < count")
    spec_path = Path(path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    declared_run_dir = Path(spec.get("run_dir", spec_path.parent))
    run_dir = (
        declared_run_dir
        if declared_run_dir.is_absolute()
        else spec_path.parent
    )
    shared_settings = spec.get("settings", {})
    shared_provenance = spec.get("provenance", {})
    spec_digest = str(spec.get("spec_digest") or _derived_spec_digest(spec))
    completed = 0
    reused = 0
    failed = 0

    for zero_index, cell in enumerate(spec.get("cells", ())):
        if zero_index % shard_count != shard_index:
            continue
        cell_id = str(cell["cell_id"])
        params = cell.get("params", {})
        settings = {**shared_settings, **cell.get("settings", {})}
        manifest_path = run_dir / "cells" / cell_id / "manifest.json"
        existing = _read_manifest(manifest_path)
        if _matching_success_manifest(
            existing,
            cell_id=cell_id,
            params=params,
            settings=settings,
            provenance=shared_provenance,
            spec_digest=spec_digest,
        ):
            reused += 1
            continue

        started = time.perf_counter()
        try:
            manifest = scan_cell(
                params=params,
                settings=settings,
                provenance=shared_provenance,
                spec_digest=spec_digest,
            )
            manifest.update(
                {
                    "cell_id": cell_id,
                    "params": dict(params),
                    "settings": dict(settings),
                    "provenance": dict(shared_provenance),
                    "spec_digest": spec_digest,
                    "runtime_seconds": time.perf_counter() - started,
                }
            )
            if manifest.get("compute_success") is True:
                completed += 1
            else:
                failed += 1
        except Exception as error:  # retain a retryable, attributable failure
            failed += 1
            manifest = {
                "schema_version": 1,
                "manifest_schema": _MANIFEST_SCHEMA,
                "completed": False,
                "compute_success": False,
                "cell_id": cell_id,
                "params": dict(params),
                "settings": dict(settings),
                "provenance": dict(shared_provenance),
                "spec_digest": spec_digest,
                "runtime_seconds": time.perf_counter() - started,
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        _atomic_json(manifest_path, manifest)

    return {
        "completed": completed,
        "reused": reused,
        "failed": failed,
    }


def verify_run(path: str | Path) -> dict[str, Any]:
    """Strictly verify every planned cell and aggregate integer counters."""

    spec_path = Path(path)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    declared_run_dir = Path(spec.get("run_dir", spec_path.parent))
    run_dir = (
        declared_run_dir
        if declared_run_dir.is_absolute()
        else spec_path.parent
    )
    shared_settings = spec.get("settings", {})
    shared_provenance = spec.get("provenance", {})
    spec_digest = str(spec.get("spec_digest") or _derived_spec_digest(spec))
    seen_ids: set[str] = set()
    seen_params: set[str] = set()
    aggregate_counts: dict[str, int] = {}
    expected_ids: set[str] = set()
    violations: list[str] = []

    for cell in spec.get("cells", ()):
        cell_id = str(cell["cell_id"])
        params = cell.get("params", {})
        settings = {**shared_settings, **cell.get("settings", {})}
        if cell_id in seen_ids:
            violations.append(f"{cell_id}: duplicate cell_id")
        seen_ids.add(cell_id)
        expected_ids.add(cell_id)
        params_key = _canonical_json(params)
        if params_key in seen_params:
            violations.append(f"{cell_id}: duplicate params")
        seen_params.add(params_key)

        manifest_path = run_dir / "cells" / cell_id / "manifest.json"
        manifest = _read_manifest(manifest_path)
        if not _matching_success_manifest(
            manifest,
            cell_id=cell_id,
            params=params,
            settings=settings,
            provenance=shared_provenance,
            spec_digest=spec_digest,
        ):
            violations.append(f"{cell_id}: missing, stale, or incomplete manifest")
            continue
        assert manifest is not None
        for field, value in manifest["counts"].items():
            if isinstance(value, int) and not isinstance(value, bool):
                aggregate_counts[field] = (
                    aggregate_counts.get(field, 0) + value
                )

    cells_dir = run_dir / "cells"
    if cells_dir.is_dir():
        orphan_ids = {
            child.name for child in cells_dir.iterdir() if child.is_dir()
        } - expected_ids
        for orphan_id in sorted(orphan_ids):
            violations.append(f"{orphan_id}: orphan cell directory")
    if violations:
        raise RuntimeError("run verification failed: " + "; ".join(violations))
    return {
        "verified_cells": len(expected_ids),
        "spec_digest": spec_digest,
        "aggregate_counts": aggregate_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_spec", help="path to parameter-scan run_spec.json")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify_only:
        summary = verify_run(arguments.run_spec)
    else:
        summary = run_spec(
            arguments.run_spec,
            shard_index=arguments.shard_index,
            shard_count=arguments.shard_count,
        )
    print(json.dumps(summary, sort_keys=True), flush=True)


__all__ = [
    "PilotCandidate",
    "TypeErasureSearch",
    "determinant_weight",
    "find_type_erasure_witnesses",
    "float_diagonal_grade_screen",
    "generate_candidate",
    "main",
    "run_spec",
    "scan_cell",
    "verify_run",
]


if __name__ == "__main__":
    main()
