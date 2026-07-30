from __future__ import annotations

import json
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
import sympy as sp

import oracle.exterior_category_pilot as pilot
from oracle.exterior_category_search import (
    TypedEdge,
    TypedExteriorGraph,
    solve_diagonal_grade_charts,
)


def _inversion_parity(permutation: list[int]) -> int:
    return sum(
        permutation[left] > permutation[right]
        for left in range(len(permutation))
        for right in range(left + 1, len(permutation))
    ) % 2


def _is_identity_or_three_cycle(permutation: list[int]) -> bool:
    moved = [
        index
        for index, image in enumerate(permutation)
        if image != index
    ]
    if not moved:
        return True
    if len(moved) != 3:
        return False
    start = moved[0]
    return (
        permutation[start] in moved
        and permutation[permutation[start]] in moved
        and permutation[permutation[permutation[start]]] == start
    )


def _known_typed_only_graphs() -> tuple[TypedExteriorGraph, TypedExteriorGraph]:
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
    exact_edges = (
        TypedEdge("forward-1", "a", "b", chart_b * coordinate_1),
        TypedEdge("forward-2", "a", "b", chart_b * coordinate_2),
        TypedEdge("return", "b", "a", chart_b.inv()),
    )
    float_edges = tuple(
        TypedEdge(
            edge.name,
            edge.source,
            edge.target,
            np.asarray(edge.matrix, dtype=float),
        )
        for edge in exact_edges
    )
    return TypedExteriorGraph(float_edges), TypedExteriorGraph(exact_edges)


def test_coboundary_control_always_has_a_diagonal_grade_certificate() -> None:
    candidate = pilot.generate_candidate(
        "coboundary_tn_control",
        dimension=4,
        seed=1701,
    )

    assert [
        (edge.name, edge.source, edge.target)
        for edge in candidate.graph.edges
    ] == [
        ("forward-1", "a", "b"),
        ("forward-2", "a", "b"),
        ("return", "b", "a"),
    ]
    certificate = solve_diagonal_grade_charts(candidate.exact_graph)
    assert certificate is not None
    assert certificate.is_entrywise_nonnegative(tolerance=1e-12)


def test_support_aware_sparse_grammar_is_structured_and_seed_deterministic() -> None:
    first = pilot.generate_candidate(
        "support_aware_sparse",
        dimension=5,
        seed=9917,
    )
    replay = pilot.generate_candidate(
        "support_aware_sparse",
        dimension=5,
        seed=9917,
    )
    different = pilot.generate_candidate(
        "support_aware_sparse",
        dimension=5,
        seed=9918,
    )

    assert first.definition == replay.definition
    assert first.candidate_id == replay.candidate_id
    assert first.definition != different.definition
    assert [
        (edge["source"], edge["target"])
        for edge in first.definition["edges"]
    ] == [("a", "b"), ("a", "b"), ("b", "a")]
    for edge in first.definition["edges"]:
        assert _inversion_parity(edge["even_permutation"]) == 0
        assert _is_identity_or_three_cycle(edge["even_permutation"])
        assert 1 <= len(edge["shears"]) <= 3
        assert set(edge["positive_diagonal"]) <= {1, 2}
        assert edge["positive_diagonal"].count(2) <= 1
        assert edge["factor_order"] == [
            "even_permutation",
            "positive_diagonal",
            "integer_shears",
        ]

        dimension = len(edge["positive_diagonal"])
        permutation = sp.zeros(dimension)
        for column, row in enumerate(edge["even_permutation"]):
            permutation[row, column] = 1
        expected = sp.diag(*edge["positive_diagonal"]) * permutation
        for shear in edge["shears"]:
            factor = sp.eye(dimension)
            factor[shear["row"], shear["column"]] = shear["coefficient"]
            expected = factor * expected
        assert sp.Matrix(edge["matrix"]) == expected


def test_type_erasure_search_uses_chronological_products_and_exact_replay() -> None:
    float_graph, exact_graph = _known_typed_only_graphs()

    result = pilot.find_type_erasure_witnesses(
        float_graph,
        exact_graph,
        max_depth=2,
        tolerance=1e-12,
        max_witnesses=1,
    )

    assert result.determinant_checks == 5
    assert result.exact_replays == 1
    assert result.ambiguous_checks == 0
    assert len(result.witnesses) == 1
    assert result.witnesses[0]["word"] == ["forward-1", "forward-2"]
    assert np.isclose(result.witnesses[0]["float_weight"], -12.0)
    assert result.witnesses[0]["exact_weight"] == "-12"
    assert result.witnesses[0]["depth"] == 2
    assert (
        result.witnesses[0][
            "primitive_real_log_impossible_by_negative_singleton"
        ]
        is False
    )
    assert result.witnesses[0]["product_order"] == "B_last ... B_first"
    assert not float_graph.is_legal_word(result.witnesses[0]["word"])
    assert exact_graph.word_product(
        result.witnesses[0]["word"]
    ) == exact_graph.word_product(("forward-1", "forward-2"))


def test_a_negative_singleton_is_retained_but_flagged_as_nonphysical() -> None:
    exact_graph = TypedExteriorGraph(
        (
            TypedEdge(
                "bad",
                "a",
                "b",
                sp.ImmutableMatrix(sp.diag(-2, 1)),
            ),
            TypedEdge(
                "return",
                "b",
                "a",
                sp.ImmutableMatrix(sp.eye(2)),
            ),
        )
    )
    float_graph = TypedExteriorGraph(
        TypedEdge(
            edge.name,
            edge.source,
            edge.target,
            np.asarray(edge.matrix, dtype=float),
        )
        for edge in exact_graph.edges
    )

    result = pilot.find_type_erasure_witnesses(
        float_graph,
        exact_graph,
        max_depth=2,
        tolerance=1e-12,
        max_witnesses=1,
    )

    assert result.witnesses[0]["word"] == ["bad"]
    assert result.witnesses[0]["depth"] == 1
    assert (
        result.witnesses[0][
            "primitive_real_log_impossible_by_negative_singleton"
        ]
        is True
    )


def test_scan_cell_records_a_complete_candidate_funnel() -> None:
    params = {
        "grammar": "coboundary_tn_control",
        "dimension": 4,
        "seed": 4100,
    }
    settings = {
        "candidates_per_cell": 3,
        "max_word_depth": 2,
        "weight_tolerance": 1e-12,
        "max_exact_witnesses_per_candidate": 1,
        "legal_stress_depths": [2, 4, 8],
        "legal_stress_candidates_per_cell": 1,
        "legal_stress_words_per_depth": 4,
    }
    provenance = {"protocol": "exterior-category-pilot-test-v1"}

    manifest = pilot.scan_cell(
        params=params,
        settings=settings,
        provenance=provenance,
        spec_digest="spec-test",
    )

    counts = manifest["counts"]
    assert manifest["compute_success"] is True
    assert manifest["params"] == params
    assert manifest["settings"] == settings
    assert manifest["provenance"] == provenance
    assert manifest["spec_digest"] == "spec-test"
    assert counts["candidates"] == 3
    assert counts["certificate_stage_checks"] == 6
    assert counts["edge_grade_checks"] == 3 * 3 * (4 + 1)
    assert counts["legal_stress_word_checks"] == 12
    assert counts["typed_only_calibration"] == 2
    assert counts["ambiguous"] == 1
    assert counts["no_chart"] == 0
    assert counts["provisional_front_door_survivor"] == 0
    assert counts["determinant_checks"] == sum(
        record["determinant_checks"]
        for record in manifest["candidate_records"]
    )
    assert counts["exact_witnesses"] == len(manifest["exact_witnesses"])
    for record in manifest["candidate_records"]:
        assert "induced_by_one_particle_charts" not in record
        assert "induced_by_diagonal_sign_charts" in record
        assert len(record["legal_closed_witness"]["word"]) >= 2
        candidate = pilot.generate_candidate(
            record["grammar"],
            dimension=params["dimension"],
            seed=record["candidate_seed"],
        )
        assert candidate.exact_graph.is_closed_word(
            record["legal_closed_witness"]["word"]
        )
    legal_stress = manifest["candidate_records"][0]["legal_stress"]
    assert len(legal_stress) == 12
    assert sorted({item["depth"] for item in legal_stress}) == [2, 4, 8]
    assert {item["strategy"] for item in legal_stress} == {
        "alternating_first",
        "alternating_second",
        "palindromic_mixed",
        "seeded_random",
    }
    assert all(
        int(item["exact_weight"]) >= 0
        for item in manifest["candidate_records"][0]["legal_stress"]
    )
    assert (
        counts["no_chart"]
        + counts["typed_only_calibration"]
        + counts["signed_permutation_tn_reduction"]
        + counts["provisional_front_door_survivor"]
        + counts["known_class_filter_incomplete"]
        + counts["untyped_not_separated"]
        + counts["ambiguous"]
        == counts["candidates"]
    )


def test_a_float_only_chart_never_survives_a_failed_exact_solver(
    monkeypatch,
) -> None:
    backend_calls: list[str] = []
    screen_calls: list[str] = []

    def accept_float_screen(graph, **kwargs):
        assert all(
            isinstance(edge.matrix, np.ndarray)
            for edge in graph.edges
        )
        screen_calls.append("float")
        return True

    def reject_exact(graph, **kwargs):
        assert all(
            isinstance(edge.matrix, sp.MatrixBase)
            for edge in graph.edges
        )
        backend_calls.append("exact")
        return None

    monkeypatch.setattr(
        pilot,
        "float_diagonal_grade_screen",
        accept_float_screen,
    )
    monkeypatch.setattr(pilot, "solve_diagonal_grade_charts", reject_exact)
    manifest = pilot.scan_cell(
        params={
            "grammar": "coboundary_tn_control",
            "dimension": 4,
            "seed": 17,
        },
        settings={
            "candidates_per_cell": 1,
            "max_word_depth": 2,
            "weight_tolerance": 1e-12,
        },
        provenance={"protocol": "exact-gate-test"},
        spec_digest="exact-gate",
    )

    assert screen_calls == ["float"]
    assert backend_calls == ["exact"]
    assert manifest["counts"]["typed_only_calibration"] == 0
    assert manifest["counts"]["provisional_front_door_survivor"] == 0
    assert manifest["counts"]["no_chart"] == 1
    assert manifest["candidate_records"][0]["classification"] == "no_chart"
    assert (
        manifest["candidate_records"][0]["chart_screen"]
        == "float_pass_exact_fail"
    )


def test_float_screen_counts_every_edge_grade_even_after_a_contradiction(
    monkeypatch,
) -> None:
    graph = TypedExteriorGraph(
        (
            TypedEdge("plain", "a", "b", np.eye(2)),
            TypedEdge(
                "conflict",
                "a",
                "b",
                np.diag([-1.0, 1.0]),
            ),
        )
    )
    calls: list[int] = []
    real_compound = pilot.compound_matrix

    def counted_compound(matrix, grade):
        calls.append(grade)
        return real_compound(matrix, grade)

    monkeypatch.setattr(pilot, "compound_matrix", counted_compound)

    assert not pilot.float_diagonal_grade_screen(graph, tolerance=1e-12)
    assert len(calls) == len(graph.edges) * (graph.dimension + 1)


def test_apparent_support_survivor_is_absorbed_by_signed_permutation_filter(
    monkeypatch,
) -> None:
    candidate = pilot.generate_candidate(
        "support_aware_sparse",
        dimension=4,
        seed=103,
    )
    monkeypatch.setattr(
        pilot,
        "generate_candidate",
        lambda *args, **kwargs: candidate,
    )

    manifest = pilot.scan_cell(
        params={
            "grammar": "support_aware_sparse",
            "dimension": 4,
            "seed": 1,
        },
        settings={
            "candidates_per_cell": 1,
            "max_word_depth": 5,
            "weight_tolerance": 1e-10,
        },
        provenance={"protocol": "support-survivor-test"},
        spec_digest="support-survivor",
    )

    record = manifest["candidate_records"][0]
    assert record["classification"] == "signed_permutation_tn_reduction"
    assert record["induced_by_diagonal_sign_charts"] is False
    assert record["signed_permutation_filter_complete"] is True
    assert record["signed_permutation_tn_witness"] is not None
    assert len(record["legal_closed_witness"]["word"]) >= 2
    assert record["exact_witnesses"][0]["exact_weight"] == "-16"
    assert record["exact_witnesses"][0]["depth"] == 5
    assert (
        record["exact_witnesses"][0][
            "primitive_real_log_impossible_by_negative_singleton"
        ]
        is False
    )
    assert record["minimum_exact_negative_word_depth"] == 5
    assert record["primitive_real_log_audit"] == {
        "all_edges_pass": False,
        "status": "not_run",
        "reason": (
            "a product-of-elementary-factors is not a single real logarithm"
        ),
    }
    assert record["physical_strong_candidate"] is False
    assert manifest["counts"]["provisional_front_door_survivor"] == 0
    assert manifest["counts"]["signed_permutation_tn_reduction"] == 1


def test_filter_clean_support_branch_is_only_a_provisional_survivor(
    monkeypatch,
) -> None:
    candidate = pilot.generate_candidate(
        "support_aware_sparse",
        dimension=4,
        seed=103,
    )
    monkeypatch.setattr(
        pilot,
        "generate_candidate",
        lambda *args, **kwargs: candidate,
    )
    monkeypatch.setattr(
        pilot,
        "find_signed_permutation_tn_coboundary",
        lambda *args, **kwargs: None,
    )

    manifest = pilot.scan_cell(
        params={
            "grammar": "support_aware_sparse",
            "dimension": 4,
            "seed": 1,
        },
        settings={
            "candidates_per_cell": 1,
            "max_word_depth": 5,
            "weight_tolerance": 1e-10,
        },
        provenance={"protocol": "filter-clean-branch-test"},
        spec_digest="filter-clean-branch",
    )

    record = manifest["candidate_records"][0]
    assert record["signed_permutation_filter_complete"] is True
    assert record["signed_permutation_tn_witness"] is None
    assert record["classification"] == "provisional_front_door_survivor"
    assert record["novelty_status"] == "provisional_after_front_door_filters"


def test_an_induced_diagonal_sign_solution_without_a_negative_word_is_not_separated(
    monkeypatch,
) -> None:
    candidate = pilot.generate_candidate(
        "support_aware_sparse",
        dimension=4,
        seed=676,
    )
    monkeypatch.setattr(
        pilot,
        "generate_candidate",
        lambda *args, **kwargs: candidate,
    )

    manifest = pilot.scan_cell(
        params={
            "grammar": "support_aware_sparse",
            "dimension": 4,
            "seed": 2,
        },
        settings={
            "candidates_per_cell": 1,
            "max_word_depth": 2,
            "weight_tolerance": 1e-10,
        },
        provenance={"protocol": "diagonal-calibration-test"},
        spec_digest="diagonal-calibration",
    )

    record = manifest["candidate_records"][0]
    assert record["induced_by_diagonal_sign_charts"] is True
    assert record["classification"] == "untyped_not_separated"
    assert manifest["counts"]["typed_only_calibration"] == 0
    assert manifest["counts"]["untyped_not_separated"] == 1
    assert manifest["counts"]["provisional_front_door_survivor"] == 0


def _write_run_spec(path: Path) -> dict[str, object]:
    run_dir = path.parent
    spec: dict[str, object] = {
        "run_id": "pilot-resume-test",
        "run_dir": str(run_dir),
        "spec_digest": "digest-v1",
        "settings": {
            "candidates_per_cell": 1,
            "max_word_depth": 2,
            "weight_tolerance": 1e-12,
        },
        "provenance": {"protocol": "pilot-resume-test-v1"},
        "cells": [
            {
                "cell_id": "cell-0001",
                "params": {
                    "grammar": "coboundary_tn_control",
                    "dimension": 4,
                    "seed": 7,
                },
            }
        ],
    }
    path.write_text(json.dumps(spec), encoding="utf-8")
    return spec


def test_run_spec_resumes_only_an_exactly_matching_success_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    spec_path = tmp_path / "run" / "run_spec.json"
    spec_path.parent.mkdir()
    spec = _write_run_spec(spec_path)
    calls: list[dict[str, object]] = []

    def fake_scan_cell(**kwargs):
        calls.append(kwargs)
        return {
            "schema_version": 1,
            "manifest_schema": "exterior-category-pilot-cell-v1",
            "completed": True,
            "compute_success": True,
            "params": kwargs["params"],
            "settings": kwargs["settings"],
            "provenance": kwargs["provenance"],
            "spec_digest": kwargs["spec_digest"],
            "counts": {"candidates": 1},
            "candidate_records": [],
            "exact_witnesses": [],
        }

    monkeypatch.setattr(pilot, "scan_cell", fake_scan_cell)

    assert pilot.run_spec(spec_path) == {
        "completed": 1,
        "reused": 0,
        "failed": 0,
    }
    assert pilot.run_spec(spec_path) == {
        "completed": 0,
        "reused": 1,
        "failed": 0,
    }
    assert len(calls) == 1

    manifest_path = (
        spec_path.parent / "cells" / "cell-0001" / "manifest.json"
    )
    baseline = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches = (
        ("completed", False),
        ("compute_success", False),
        ("manifest_schema", "old-schema"),
        ("schema_version", 2),
        ("params", {"grammar": "wrong"}),
        ("settings", {"candidates_per_cell": 999}),
        ("provenance", {"protocol": "wrong"}),
        ("spec_digest", "wrong"),
        ("counts", None),
        ("candidate_records", None),
        ("exact_witnesses", None),
    )
    for field, wrong_value in mismatches:
        damaged = dict(baseline)
        damaged[field] = wrong_value
        manifest_path.write_text(json.dumps(damaged), encoding="utf-8")
        assert pilot.run_spec(spec_path)["completed"] == 1
        restored = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert restored["compute_success"] is True
        assert restored["params"] == spec["cells"][0]["params"]
        assert restored["settings"] == spec["settings"]
        assert restored["provenance"] == spec["provenance"]
        assert restored["spec_digest"] == spec["spec_digest"]

    manifest_path.write_text("{broken", encoding="utf-8")
    assert pilot.run_spec(spec_path)["completed"] == 1
    assert len(calls) == 1 + len(mismatches) + 1
    assert not tuple(manifest_path.parent.glob("*.tmp"))


def test_verify_run_rejects_missing_or_stale_cells(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "run" / "run_spec.json"
    spec_path.parent.mkdir()
    _write_run_spec(spec_path)
    pilot.run_spec(spec_path)

    verified = pilot.verify_run(spec_path)

    assert verified["verified_cells"] == 1
    assert verified["aggregate_counts"]["candidates"] == 1

    manifest_path = (
        spec_path.parent / "cells" / "cell-0001" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["spec_digest"] = "stale"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="cell-0001"):
        pilot.verify_run(spec_path)


def test_every_illegal_word_checked_by_the_fixture_has_exact_backend_parity() -> None:
    float_graph, exact_graph = _known_typed_only_graphs()

    for word in product(
        [edge.name for edge in float_graph.edges],
        repeat=2,
    ):
        if float_graph.is_legal_word(word):
            continue
        float_weight = pilot.determinant_weight(float_graph, word)
        exact_weight = pilot.determinant_weight(exact_graph, word)
        assert np.isclose(float_weight, float(exact_weight))


def test_cli_runs_one_spec_and_prints_the_resume_summary(
    monkeypatch,
    capsys,
) -> None:
    calls: list[tuple[str, int, int]] = []

    def fake_run_spec(path, *, shard_index, shard_count):
        calls.append((path, shard_index, shard_count))
        return {"completed": 2, "reused": 1, "failed": 0}

    monkeypatch.setattr(pilot, "run_spec", fake_run_spec)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "exterior_category_pilot",
            "run_spec.json",
            "--shard-index",
            "1",
            "--shard-count",
            "3",
        ],
    )

    pilot.main()

    assert calls == [("run_spec.json", 1, 3)]
    assert json.loads(capsys.readouterr().out) == {
        "completed": 2,
        "failed": 0,
        "reused": 1,
    }
